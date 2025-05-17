#!/usr/bin/env python3
# filepath: /Users/suraj/Documents/College/Major Project/Frontend /test_face_detector.py
import os
from face_detector import FaceAntiSpoofDetector

def test_detector():
    """Test the face anti-spoofing detector with a sample image."""
    # Initialize detector
    try:
        detector = FaceAntiSpoofDetector()
        print("Face Anti-Spoofing detector initialized successfully.")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Test with a sample image
    sample_folder = "project/input/images"
    if not os.path.exists(sample_folder):
        print(f"Sample folder {sample_folder} not found.")
        return
    
    # Find image files
    image_files = [f for f in os.listdir(sample_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {sample_folder}")
        return
    
    # Test the first image
    sample_image = os.path.join(sample_folder, image_files[0])
    print(f"Testing with sample image: {sample_image}")
    
    # Create output folder if it doesn't exist
    output_folder = "test_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Process image
    output_path = os.path.join(output_folder, "test_result.png")
    try:
        result = detector.predict_liveness(
            sample_image, 
            output_path=output_path,
            generate_heatmap=True
        )
        print(f"Testing successful!")
        print(f"Result: {result}")
        print(f"Output image saved to: {output_path}")
        print(f"Heatmap saved to: {output_path.replace('.', '_heatmap.')}")
    except Exception as e:
        print(f"Error testing detector: {e}")

if __name__ == "__main__":
    test_detector()
