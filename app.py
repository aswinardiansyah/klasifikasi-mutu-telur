from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from werkzeug.utils import secure_filename

# -----------------------------
# KONFIGURASI FLASK
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# LOAD MODEL DAN SCALER
# -----------------------------
MODEL_PATH = 'model_baru.pkl'
SCALER_PATH = 'scaler-telurfix.pkl'

# Pastikan kedua file ada
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model_telur.pkl tidak ditemukan di folder project.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("❌ scaler-telur.pkl tidak ditemukan di folder project.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# FUNGSI PENDUKUNG
# -----------------------------
def load_image(image_path):
    """Baca gambar dan ubah ke RGB"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_and_detect_telur(image):
    """Resize dan deteksi area telur berdasarkan warna HSV"""
    image = cv2.resize(image, (256, 256))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Rentang warna telur coklat
    lower_telur = np.array([5, 60, 40])
    upper_telur = np.array([25, 255, 255])

    mask = cv2.inRange(hsv_image, lower_telur, upper_telur)

    # Operasi morfologi untuk membersihkan noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    masked_rgb = cv2.bitwise_and(image, image, mask=mask)
    gray_result = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)

    # Hitung luas area terdeteksi (dalam persentase)
    mask_percentage = (np.count_nonzero(mask) / (256 * 256)) * 100
    return masked_rgb, gray_result, mask_percentage


def extract_glcm_features(gray_image):
    """Ekstraksi fitur tekstur menggunakan GLCM"""
    if gray_image.max() == 0:
        return [0, 0, 0, 0]
    gray_image = (gray_image / gray_image.max() * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_image,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True,
        levels=256
    )
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    return [contrast, correlation, energy, homogeneity]


def extract_hsv_features(masked_rgb):
    """Ekstraksi fitur warna HSV"""
    hsv_image = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2HSV)
    h = hsv_image[:, :, 0][hsv_image[:, :, 0] > 0]
    s = hsv_image[:, :, 1][hsv_image[:, :, 1] > 0]
    v = hsv_image[:, :, 2][hsv_image[:, :, 2] > 0]
    hue = np.mean(h) if len(h) > 0 else 0
    saturation = np.mean(s) if len(s) > 0 else 0
    value = np.mean(v) if len(v) > 0 else 0
    return [hue, saturation, value]


def process_image(image_path):
    """Proses gambar dan prediksi mutu telur"""
    img = load_image(image_path)
    if img is None:
        return None, None, "⚠️ Gagal memuat gambar."

    masked_rgb, gray_result, mask_percentage = preprocess_and_detect_telur(img)

    # Jika telur tidak terdeteksi
    if mask_percentage < 5:
        return None, None, "⚠️ Telur tidak terdeteksi pada gambar."

    # Ekstraksi fitur
    hsv_features = extract_hsv_features(masked_rgb)
    glcm_features = extract_glcm_features(gray_result)
    all_features = hsv_features + glcm_features

    # Normalisasi (pakai scaler dari Colab)
    features_scaled = scaler.transform([all_features])

    # Prediksi model
    prediction = model.predict(features_scaled)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        confidence = np.max(probs) * 100
    else:
        confidence = 0

    feature_dict = {
        "Hue": round(hsv_features[0], 3),
        "Saturation": round(hsv_features[1], 3),
        "Value": round(hsv_features[2], 3),
        "Contrast": round(glcm_features[0], 3),
        "Correlation": round(glcm_features[1], 3),
        "Energy": round(glcm_features[2], 3),
        "Homogeneity": round(glcm_features[3], 3)
    }

    return prediction, feature_dict, f"Akurasi estimasi: {confidence:.2f}%"


# -----------------------------
# ROUTING
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error="⚠️ Tidak ada file yang diunggah.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', error="⚠️ File belum dipilih.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        prediction, features, confidence = process_image(file_path)

        if prediction is None:
            return render_template('predict.html', error=confidence, filename=filename)

        return render_template(
            'predict.html',
            result=prediction,
            confidence=confidence,
            filename=filename,
            features=features
        )

    return render_template('predict.html')


# -----------------------------
# JALANKAN SERVER
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)