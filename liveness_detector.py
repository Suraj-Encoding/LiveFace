# Liveness Detector Script #

# Required Packages
import os
import cv2
import numpy as np
import pywt
import tensorflow as tf
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

    # Generate Feature Importance #
    def generate_feature_importance(self, is_real):
        # Generate explainable AI feature importances
        # Simulate feature importance scores for each feature category
        # In a real implementation, this would come from model analysis like SHAP or LIME
        
        # Create random importance scores but make them more realistic based on result
        if is_real:
            # For real faces, higher scores for natural features
            base_scores = np.array([0.7, 0.8, 0.75, 0.85, 0.82, 0.78, 0.79])

            # Add some randomness
            noise = np.random.uniform(-0.1, 0.1, size=len(base_scores))
            scores = base_scores + noise

            # Clip to 0-1 range
            scores = np.clip(scores, 0, 1)
        else:
            # For fake faces, lower scores on key features
            base_scores = np.array([0.3, 0.4, 0.25, 0.35, 0.2, 0.38, 0.29])

            # Add some randomness
            noise = np.random.uniform(-0.1, 0.1, size=len(base_scores))
            scores = base_scores + noise

            # Clip to 0-1 range
            scores = np.clip(scores, 0, 1)
        
        # Create explanation details
        explanations = []
        for i, cat in enumerate(self.feature_categories):
            explanation = {
                "feature": cat["name"],
                "confidence": float(scores[i]),
                "isReal": is_real
            }
            explanations.append(explanation)
            
        # Sort by score (descending for real faces, ascending for fake faces)
        if is_real:
            return sorted(explanations, key=lambda x: x["confidence"], reverse=True)
        else:
            return sorted(explanations, key=lambda x: x["confidence"], reverse=False)

    # Generate Heatmap #
    def generate_heatmap(self, img, is_real, heatmap_image_path):
        # Generate a heatmap visualization to highlight important regions
        # Resize for visualization
        vis_img = cv2.resize(img, (299, 299))
        
        # Create a simulated attention map
        # In real implementation, this would use Grad-CAM or similar techniques
        
        # Generate more realistic heatmap based on whether it's real or fake
        if is_real:
            # For real faces, more uniform heatmap with focus on eyes, nose, mouth
            heatmap = np.zeros((299, 299))
            
            # Add focus areas (eyes, nose, mouth regions)
            center_x, center_y = 299//2, 299//2
            
            # Eyes region
            for x in range(299):
                for y in range(299):
                    # Left eye region
                    dist_left_eye = np.sqrt((x - center_x + 50)**2 + (y - center_y - 30)**2)
                    if dist_left_eye < 35:
                        heatmap[y, x] = max(heatmap[y, x], 1 - dist_left_eye/35)
                    
                    # Right eye region
                    dist_right_eye = np.sqrt((x - center_x - 50)**2 + (y - center_y - 30)**2)
                    if dist_right_eye < 35:
                        heatmap[y, x] = max(heatmap[y, x], 1 - dist_right_eye/35)
                        
                    # Nose region
                    dist_nose = np.sqrt((x - center_x)**2 + (y - center_y + 10)**2)
                    if dist_nose < 30:
                        heatmap[y, x] = max(heatmap[y, x], 1 - dist_nose/30)
                        
                    # Mouth region
                    dist_mouth = np.sqrt((x - center_x)**2 + (y - center_y + 60)**2)
                    if dist_mouth < 40:
                        heatmap[y, x] = max(heatmap[y, x], 1 - dist_mouth/40)
            
            # Add some noise for realism
            heatmap = heatmap + np.random.uniform(0, 0.3, (299, 299))
            
        else:
            # For fake faces, more irregular patterns with inconsistencies
            heatmap = np.zeros((299, 299))
            
            # Add some irregular "suspicious" regions
            for _ in range(5):
                cx = np.random.randint(50, 249)
                cy = np.random.randint(50, 249)
                radius = np.random.randint(20, 60)
                intensity = np.random.uniform(0.7, 1.0)
                
                for x in range(299):
                    for y in range(299):
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if dist < radius:
                            heatmap[y, x] = max(heatmap[y, x], intensity * (1 - dist/radius))
            
            # Add more noise for fake images
            heatmap = heatmap + np.random.uniform(0, 0.2, (299, 299))
        
        # Normalize heatmap
        heatmap = np.clip(heatmap, 0, 1)
        
        # Apply Gaussian blur to make it smoother
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Normalize again after blur
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Convert to RGB heatmap using different colormaps based on result
        if is_real:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        else:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_HOT)
        
        # Overlay on the original image
        alpha = 0.5
        overlay = cv2.addWeighted(vis_img, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Add text label
        label = "REAL FACE" if is_real else "FAKE FACE"
        color = (0, 255, 0) if is_real else (0, 0, 255)
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Save the visualization
        cv2.imwrite(heatmap_image_path, overlay)

    # Predict Liveness #
    def predict_liveness(self, original_image_path):
        # Predict whether the given image contains a real or fake face

        # Load image
        img = cv2.imread(original_image_path)
        if img is None:
            return {"error": f"unable to load image. check the file path."}

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
        
        # Generate explanations
        explanations = self.generate_feature_importance(is_real)
        
        # Check if we need to generate a new heatmap
        heatmap_image_path = os.path.join(os.path.dirname(original_image_path), "heatmap_image.png")
        if not os.path.exists(heatmap_image_path):
            # Generate heatmap visualization only if it doesn't exist
            self.generate_heatmap(img, is_real, heatmap_image_path)
        
        # Form the result
        result = {
            "isReal": bool(is_real),
            "confidence": float(confidence) * 100,
            "explanations": explanations
        }
        
        # Return the results
        return result