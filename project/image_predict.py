#!/usr/bin/env python3
# filepath: /Users/suraj/Documents/College/Major Project/Frontend /project/test_single_image.py
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np
import argparse

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
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
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect face
    bbox = face_detector([img])[0]
    
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)   
    
    return bbox, label, score

def test_image(image_path, model_path=None, threshold=0.5, output_path=None):
    """
    Test a single image for spoofing detection
    
    Args:
        image_path (str): Path to the input image
        model_path (str, optional): Path to the anti-spoofing model. Defaults to saved_models/AntiSpoofing_bin_1.5_128.onnx
        threshold (float, optional): Threshold for real/fake classification. Defaults to 0.5
        output_path (str, optional): Path to save the result image. Defaults to None
        
    Returns:
        tuple: (result_string, label, score) where result_string is "REAL", "FAKE", or "UNKNOWN"
    """
    if model_path is None:
        model_path = "saved_models/AntiSpoofing_bin_1.5_128.onnx"
    
    # Load models
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof(model_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None, None
    
    # Get image dimensions
    frame_height, frame_width = image.shape[:2]
    
    # Make prediction
    pred = make_prediction(image, face_detector, anti_spoof)
    
    # If face is detected
    if pred is not None:
        (x1, y1, x2, y2), label, score = pred
        
        # Set rectangle width based on image size
        rec_width = max(1, int(frame_width/240))
        txt_offset = int(frame_height/50)
        txt_width = max(1, int(frame_width/480))
        
        if label == 0:
            if score > threshold:
                res_text = "REAL      {:.2f}".format(score)
                result_string = "REAL"
                color = COLOR_REAL
            else: 
                res_text = "unknown"
                result_string = "UNKNOWN"
                color = COLOR_UNKNOWN
        else:
            res_text = "FAKE      {:.2f}".format(score)
            result_string = "FAKE"
            color = COLOR_FAKE
            
        # Draw bbox with label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, rec_width)
        cv2.putText(image, res_text, (x1, y1-txt_offset), 
                    cv2.FONT_HERSHEY_COMPLEX, (x2-x1)/250, color, txt_width)
        
        # Save the processed image if output path is provided
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Processed image saved to {output_path}")
        
        # Display the result
        cv2.imshow('Face AntiSpoofing Result', image)
        print(f"Result: {result_string} (confidence: {score:.2f})")
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return result_string, label, score
    else:
        print("No face detected in the image")
        return "NO_FACE", None, None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Spoofing attack detection on a single image")
    parser.add_argument("--input", "-i", type=str, required=True, 
                       help="Path to input image for prediction")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Path to save processed image")
    parser.add_argument("--model_path", "-m", type=str, 
                       default="saved_models/AntiSpoofing_bin_1.5_128.onnx", 
                       help="Path to ONNX model")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                       help="Real face probability threshold (0-1)")
    args = parser.parse_args()
    
    # Test the image
    test_image(args.input, args.model_path, args.threshold, args.output)
