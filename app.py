import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops
import os

# -----------------------------
# LOAD MODEL DAN SCALER
# -----------------------------
MODEL_PATH = 'model_baru.pkl'
SCALER_PATH = 'scaler-telurfix.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# FUNGSI PENDUKUNG
# -----------------------------
def load_image(image_file):
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_and_detect_telur(image):
    image = cv2.resize(image, (256, 256))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_telur = np.array([5, 60, 40])
    upper_telur = np.array([25, 255, 255])
    mask = cv2.inRange(hsv_image, lower_telur, upper_telur)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    masked_rgb = cv2.bitwise_and(image, image, mask=mask)
    gray_result = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
    mask_percentage = (np.count_nonzero(mask) / (256 * 256)) * 100
    return masked_rgb, gray_result, mask_percentage

def extract_glcm_features(gray_image):
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
    hsv_image = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2HSV)
    h = hsv_image[:, :, 0][hsv_image[:, :, 0] > 0]
    s = hsv_image[:, :, 1][hsv_image[:, :, 1] > 0]
    v = hsv_image[:, :, 2][hsv_image[:, :, 2] > 0]
    return [np.mean(h) if len(h) > 0 else 0,
            np.mean(s) if len(s) > 0 else 0,
            np.mean(v) if len(v) > 0 else 0]

def process_image(image):
    masked_rgb, gray_result, mask_percentage = preprocess_and_detect_telur(image)
    if mask_percentage < 5:
        return None, None, "⚠️ Telur tidak terdeteksi pada gambar."
    
    hsv_features = extract_hsv_features(masked_rgb)
    glcm_features = extract_glcm_features(gray_result)
    all_features = hsv_features + glcm_features
    features_scaled = scaler.transform([all_features])
    prediction = model.predict(features_scaled)[0]
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(features_scaled)[0]) * 100
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
# STREAMLIT APP
# -----------------------------
st.title("🥚 Deteksi Mutu Telur Menggunakan Machine Learning")

uploaded_file = st.file_uploader("Unggah gambar telur di sini", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Gambar Telur", use_column_width=True)

    if st.button("🔍 Prediksi"):
        prediction, features, confidence = process_image(image)
        
        if prediction is None:
            st.error(confidence)
        else:
            st.success(f"✅ Hasil Prediksi: {prediction}")
            st.info(confidence)
            st.subheader("📊 Fitur Ekstraksi")
            st.json(features)
