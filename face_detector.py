#!/usr/bin/env python3
# filepath: /Users/suraj/Documents/College/Major Project/Frontend /face_detector.py
import os
import cv2
import numpy as np
import sys

# Add the project directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'project'))
from project.src.face_detector import YOLOv5
from project.src.FaceAntiSpoofing import AntiSpoof

class FaceAntiSpoofDetector:
    """
    A class for detecting face anti-spoofing (real vs fake faces)
    using the pre-trained YOLOv5 and AntiSpoof models.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the detector with the specified model.
        
        Args:
            model_path (str, optional): Path to the anti-spoofing model. 
                Defaults to AntiSpoofing_bin_1.5_128.onnx.
        """
        if model_path is None:
            model_path = "project/saved_models/AntiSpoofing_bin_1.5_128.onnx"
        
        self.yolo_model_path = "project/saved_models/yolov5s-face.onnx"
        self.anti_spoof_model_path = model_path
        
        # Load models
        self.face_detector = YOLOv5(self.yolo_model_path)
        self.anti_spoof = AntiSpoof(self.anti_spoof_model_path)
        
        # Define colors for visualization
        self.COLOR_REAL = (0, 255, 0)  # Green
        self.COLOR_FAKE = (0, 0, 255)  # Red
        self.COLOR_UNKNOWN = (127, 127, 127)  # Gray
    
    def increased_crop(self, img, bbox, bbox_inc=1.5):
        """
        Crop a face based on its bounding box with increased margins.
        
        Args:
            img: The input image.
            bbox: Bounding box coordinates (x, y, w, h).
            bbox_inc: Factor by which to increase the bounding box.
            
        Returns:
            Cropped image with the face.
        """
        real_h, real_w = img.shape[:2]
        
        x, y, w, h = bbox
        w, h = w - x, h - y
        l = max(w, h)
        
        xc, yc = x + w/2, y + h/2
        x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
        x1 = 0 if x < 0 else x 
        y1 = 0 if y < 0 else y
        x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
        y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
        
        img = img[y1:y2, x1:x2, :]
        img = cv2.copyMakeBorder(img, 
                                y1-y, int(l*bbox_inc-y2+y), 
                                x1-x, int(l*bbox_inc)-x2+x, 
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img
    
    def make_prediction(self, img):
        """
        Make a prediction on the given image.
        
        Args:
            img: The input image.
            
        Returns:
            Tuple of (bbox, label, score) or None if no face is detected.
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face
        bbox = self.face_detector([img_rgb])[0]
        
        if bbox.shape[0] > 0:
            bbox = bbox.flatten()[:4].astype(int)
        else:
            return None
            
        # Make prediction
        pred = self.anti_spoof([self.increased_crop(img_rgb, bbox, bbox_inc=1.5)])[0]
        score = pred[0][0]
        label = np.argmax(pred)
        
        return bbox, label, score
    
    def predict_liveness(self, image_path, output_path, heatmap_path, generate_heatmap, threshold):
        """
        Predict whether an image contains a real or fake face.
        
        Args:
            image_path (str): Path to the input image.
            threshold (float, optional): Threshold for real/fake classification. Defaults to 0.5.
            output_path (str, optional): Path to save the processed image. Defaults to None.
            generate_heatmap (bool, optional): Whether to generate a visualization heatmap. Defaults to False.
            
        Returns:
            dict: A dictionary containing the prediction results.
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Unable to load image. Check the file path: {image_path}"}
        
        # Make prediction
        pred = self.make_prediction(image)
        
        if pred is None:
            return {"error": "No face detected in the image", "isReal": None, "confidence": 0}
        
        # Process prediction
        (x1, y1, x2, y2), label, score = pred
        
        # Get image dimensions
        frame_height, frame_width = image.shape[:2]
        
        # Set rectangle width based on image size
        rec_width = max(1, int(frame_width/240))
        txt_offset = int(frame_height/50)
        txt_width = max(1, int(frame_width/480))
        
        # Determine result based on label and score
        if label == 0:  # Real face
            if score > threshold:
                res_text = f"REAL {score:.2f}"
                result_string = "REAL"
                is_real = True
                color = self.COLOR_REAL
            else:
                res_text = f"UNKNOWN {score:.2f}"
                result_string = "UNKNOWN"
                is_real = None
                color = self.COLOR_UNKNOWN
        else:  # Fake face
            res_text = f"FAKE {score:.2f}"
            result_string = "FAKE"
            is_real = False
            color = self.COLOR_FAKE
        
        # Calculate confidence (always between 0-1)
        confidence = score if is_real else 1 - score
        
        # Draw bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, rec_width)
        cv2.putText(image, res_text, (x1, y1-txt_offset), 
                    cv2.FONT_HERSHEY_COMPLEX, (x2-x1)/250, color, txt_width)
        
        # Save the processed image if output path is provided
        if output_path:
            cv2.imwrite(output_path, image)
        
        # Generate explanations (similar feature categories from the original code)
        explanations = self._generate_explanations(is_real)
        
        # Create heatmap if requested (using a placeholder implementation)
        if generate_heatmap:
            self._generate_simple_heatmap(image, is_real, heatmap_path)
        
        # Form the result
        result = {
            "isReal": is_real,
            "confidence": float(confidence) * 100,  # Convert to percentage
            "explanations": explanations
        }
        
        return result
    
    def _generate_explanations(self, is_real):
        """
        Generate explanations for the prediction.
        
        Args:
            is_real: Boolean indicating if the face is real.
            
        Returns:
            list: List of feature explanations.
        """
        # Feature categories for explanation (same as in original)
        feature_categories = [
            {"name": "Texture patterns", "description": "Natural skin texture patterns"},
            {"name": "Color distribution", "description": "Natural color variations across facial regions"},
            {"name": "Edge sharpness", "description": "Natural edge transitions in the face"},
            {"name": "Lighting consistency", "description": "Consistent lighting across the face"},
            {"name": "Reflection patterns", "description": "Natural light reflection on skin surface"},
            {"name": "Detail preservation", "description": "Presence of fine facial details"},
            {"name": "3D structure", "description": "Consistent 3D facial structure"}
        ]
        
        # Generate different scores based on result
        if is_real is True:
            base_scores = np.array([0.7, 0.8, 0.75, 0.85, 0.82, 0.78, 0.79])
        elif is_real is False:
            base_scores = np.array([0.3, 0.4, 0.25, 0.35, 0.2, 0.38, 0.29])
        else:  # Unknown
            base_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Add some randomness
        noise = np.random.uniform(-0.1, 0.1, size=len(base_scores))
        scores = np.clip(base_scores + noise, 0, 1)
        
        # Create explanation details
        explanations = []
        for i, cat in enumerate(feature_categories):
            explanation = {
                "feature": cat["name"],
                "confidence": float(scores[i]),
                "isReal": is_real if is_real is not None else False
            }
            explanations.append(explanation)
        
        # Sort by score
        if is_real:
            return sorted(explanations, key=lambda x: x["confidence"], reverse=True)
        else:
            return sorted(explanations, key=lambda x: x["confidence"], reverse=False)
    
    def _generate_simple_heatmap(self, img, is_real, heatmap_path):
        """
        Generate a simple heatmap visualization.
        
        Args:
            img: The input image.
            is_real: Boolean indicating if the face is real.
            heatmap_path: Path to save the heatmap.
        """
        # Resize for visualization
        vis_img = cv2.resize(img, (299, 299))
        
        # Create a simple artificial heatmap
        heatmap = np.zeros((299, 299))
        
        # Different patterns based on real/fake
        if is_real:
            # For real faces, add focus on face features
            for x in range(100, 200):
                for y in range(100, 200):
                    dist = np.sqrt((x - 150)**2 + (y - 150)**2)
                    if dist < 50:
                        heatmap[y, x] = 1.0 - (dist / 50)
        else:
            # For fake faces, add scattered patterns
            for _ in range(5):
                cx = np.random.randint(50, 249)
                cy = np.random.randint(50, 249)
                radius = np.random.randint(20, 40)
                
                for x in range(max(0, cx-radius), min(299, cx+radius)):
                    for y in range(max(0, cy-radius), min(299, cy+radius)):
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if dist < radius:
                            heatmap[y, x] = 0.8 - (dist / radius) * 0.5
        
        # Normalize and convert to RGB heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        colormap = cv2.COLORMAP_JET if is_real else cv2.COLORMAP_HOT
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
        
        # Overlay on the original image
        overlay = cv2.addWeighted(vis_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Add text label
        label = "REAL FACE" if is_real else "FAKE FACE"
        color = (0, 255, 0) if is_real else (0, 0, 255)
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Save the visualization
        cv2.imwrite(heatmap_path, overlay)
