# Liveness Detector Script #

# Required Packages
import cv2
import numpy as np
import tensorflow as tf
import pywt
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Liveness Detector Class
class LivenessDetector:
    def __init__(self, feature_extractor_path, fusion_model_path):
        # Load models
        self.feature_extractor = load_model(feature_extractor_path)
        self.fusion_model = load_model(fusion_model_path)
        
        # Standard scaler for 'HAAR' features
        self.scaler = StandardScaler()
        
        # Feature categories for explanation
        self.feature_categories = [
            {"name": "Texture patterns", "description": "Natural skin texture patterns"},
            {"name": "Color distribution", "description": "Natural color variations across facial regions"},
            {"name": "Edge sharpness", "description": "Natural edge transitions in the face"},
            {"name": "Lighting consistency", "description": "Consistent lighting across the face"},
            {"name": "Reflection patterns", "description": "Natural light reflection on skin surface"},
            {"name": "Detail preservation", "description": "Presence of fine facial details"},
            {"name": "3D structure", "description": "Consistent 3D facial structure"}
        ]

# Extract Haar Features #
def extract_haar_features(self, image, wavelet='haar', level=3):
    # Apply Haar Wavelet Transform to the entire image and return flattened feature vector
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    except:
        # If already grayscale
        gray = image
        
    gray = cv2.resize(gray, (64, 64))  # Resize to a fixed size

    coeffs = pywt.wavedec2(gray, wavelet, level=level)  # Apply Haar Wavelet Transform
    features = []

    # Flatten wavelet coefficients
    for coeff in coeffs:
        if isinstance(coeff, tuple):  # If detail coefficients
            for subband in coeff:
                features.extend(subband.flatten())
        else:  # If approximation coefficients
            features.extend(coeff.flatten())

    # Ensure fixed feature vector length (e.g., 4096)
    target_size = 4096
    if len(features) < target_size:
        features = np.pad(features, (0, target_size - len(features)), 'constant')
    elif len(features) > target_size:
        features = features[:target_size]

    return np.array(features).reshape(1, -1)  # Return as (1, 4096)

# Preprocess Image for CNN #
def preprocess_image_for_cnn(self, img):
    # Preprocess an image for InceptionV3 feature extraction
    img = cv2.resize(img, (299, 299))
    img = img.astype("float32") / 255.0
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Predict Liveness #
def predict_liveness(self, image_path):
    # Predict whether the given image contains a real or fake face
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "unable to load image. check the file path."}

    # Extract CNN Features (Full Image)
    cnn_input = self.preprocess_image_for_cnn(img)
    cnn_features = self.feature_extractor.predict(cnn_input, verbose=0)

    # Extract Haar Features (Full Image)
    haar_features = self.extract_haar_features(img)

    # Scale Haar Features
    haar_features_scaled = self.scaler.fit_transform(haar_features)

    # Predict using fusion model
    prediction = self.fusion_model.predict([cnn_features, haar_features_scaled], verbose=0)
    prob = float(prediction[0][0])
    is_real = prob > 0.5
    
    # Calculate confidence
    confidence = prob if is_real else 1 - prob
    
    # Return the results
    result = {
        "isReal": bool(is_real),
        "confidence": float(confidence) * 100,
        "imagePath": image_path
    }
    
    return result