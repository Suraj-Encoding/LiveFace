# Flask Server #

# Required Packages
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import base64
from werkzeug.utils import secure_filename
from liveness_detector import LivenessDetector
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__,
            template_folder='templates',
            static_folder='static',
            )

# Enable 'CORS' for all the routes    
CORS(app) 

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to '16MB'

# Initialize the liveness detector
model_dir = 'models'
try:
    detector = LivenessDetector(
        feature_extractor_path=os.path.join(model_dir, 'feature_extractor.h5'),
        fusion_model_path=os.path.join(model_dir, 'fusion_model.h5')
    )
    model_loaded = True
except Exception as e:
    print(f"error loading models: {e}")
    model_loaded = False

# Frontend Dashboard Route - Home Route #
@app.route('/')
def index():
    # Serve the main page
    return render_template('index.html', model_status=model_loaded)

# Upload Image API #
@app.route('/api/v1/upload/image', methods=['POST'])
def upload_image():
    # Handle image upload and return success status 
    try:
        # Check if image data is provided
        if 'imageData' not in request.json:
            return jsonify({'error': 'no image data provided'}), 400
        
        # Get image data from request
        image_data = request.json['imageData']
        
        # Remove the 'data:image/jpeg;base64' prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode 'base64' and save the image
        try:
            image_bytes = base64.b64decode(image_data)
        except:
            return jsonify({'error': 'invalid image data format'}), 400
            
        # Use static filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.png')
        
        # Clear existing heatmap when new image is uploaded
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap_image.png')
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({'success': True})
    
    except Exception as e:
        print(f"error uploading image: {e}")
        return jsonify({'error': f'an error occurred: {str(e)}'}), 500


# Analyze Image API #
@app.route('/api/v1/analyze/image', methods=['POST'])
def analyze_image():
    # Process the uploaded image and return liveness detection results
    if not model_loaded:
        return jsonify({'error': 'models not loaded. please check server logs.'}), 500
    
    try:
        # Use static filepath
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.png')
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'no image uploaded yet'}), 400
        
        # Process the image
        result = detector.predict_liveness(filepath)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
    
    except Exception as e:
        print(f"error processing image: {e}")
        return jsonify({'error': f'an error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print(f"model loading status: {'success' if model_loaded else 'failed'}")
    app.run(debug=True, host='localhost', port=3000)