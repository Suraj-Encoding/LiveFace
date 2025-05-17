# Flask Server #

# Required Packages
import os
import requests
import numpy as np
import base64
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from face_detector import FaceAntiSpoofDetector  # Import new detector

# Load environment variables from a '.env' file
load_dotenv()

# Get all the environment variables from '.env' file
SERVER_URI = os.getenv('SERVER_URI')
PORT = int(os.getenv('PORT'))

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

# Set all the environment variables in flask app config
app.config['SERVER_URI'] = SERVER_URI
app.config['PORT'] = PORT

# Initialize the face anti-spoofing detector
try:
    detector = FaceAntiSpoofDetector()
    model_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    model_loaded = False

# Home Route - Dashboard #
@app.route('/', methods=['GET'])
def home():
    # Serve the main page
    return render_template('index.html', model_status=model_loaded)

# Get Uploaded Image API #
@app.route('/api/v1/uploads/<image_name>', methods=['GET']) 
def get_uploaded_image(image_name):
    return send_from_directory(UPLOAD_FOLDER, image_name)

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
        input_image_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')
        
        # Clear existing heatmap when new image is uploaded
        heatmap_image_path = os.path.join(UPLOAD_FOLDER, 'heatmap_image.png')
        if os.path.exists(heatmap_image_path):
            os.remove(heatmap_image_path)
        
        with open(input_image_path, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({'success': True})
    
    except Exception as e:
        print(f"error uploading image: {e}")
        return jsonify({'error': f'an error occurred: {str(e)}'}), 500


# Analyze Image API #
@app.route('/api/v1/analyze/image', methods=['GET'])
def analyze_image():
    # Process the uploaded image and return liveness detection results
    if not model_loaded:
        return jsonify({'error': 'models not loaded. please check server logs.'}), 500
    
    try:
        # Get input image path
        input_image_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')
        
        # Check if file exists
        if not os.path.exists(input_image_path):
            return jsonify({'error': 'no image uploaded yet'}), 400
        
        # Define output path for processed image with bbox
        output_image_path = os.path.join(UPLOAD_FOLDER, 'output_image.png')
        
        # Define heatmap path
        heatmap_image_path = os.path.join(UPLOAD_FOLDER, 'heatmap_image.png')
        
        # Process the image with the new detector
        result = detector.predict_liveness(
            image_path=input_image_path, 
            output_path=output_image_path,
            heatmap_path=heatmap_image_path,
            generate_heatmap=True,
            threshold=0.75
        )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': f'an error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print(f"model loading status: {'success' if model_loaded else 'failed'}")
    app.run(debug=True, host=SERVER_URI, port=PORT)