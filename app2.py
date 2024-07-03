import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from flask import Flask, jsonify, request
from deepface import DeepFace
import cv2
from waitress import serve

app = Flask(__name__)

# Mapping of employee numbers to image paths
employee_images = {
    '123': 'path_to_image_123.jpg',
    '456': 'path_to_image_456.jpg',
    # Add more mappings as needed
}

def verify(employee_number):
    img1_path = employee_images.get(employee_number)
    if img1_path is None:
        raise ValueError(f"No image found for employee number {employee_number}")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(live_image_path)
    if img1 is None:
        raise ValueError(f"Image at path {img1_path} could not be loaded")
    if img2 is None:
        raise ValueError(f"Image at path {live_image_path} could not be loaded")
    try:
        result = DeepFace.verify(img1_path, live_image_path, model_name='Facenet')
        verification = result['verified']
        return verification, result
    except Exception as e:
        raise ValueError(f"Exception while processing {live_image_path}: {str(e)}")

@app.route("/verify", methods=['POST'])
def verify_image():
    if 'PSNo' not in request.form:
        return jsonify({'error': 'Please provide PSNo.'}), 400

    PSNo = request.form['PSNo']

    if PSNo not in employee_images:
        return jsonify({'error': 'Invalid PSNo or images not found.'}), 400

        #live_image_path = os.path.join('uploads', live_image_filename)
    # Handle live image
    live_image = request.files.get('live_image')
    if live_image:
        #calling live_image with proper parameters 
    else:
        return jsonify({'error': 'No live image provided.'}), 400

    try:
        verification, result = verify(PSNo)
        return jsonify({'verified': verification, 'result': result}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred during verification: {str(e)}'}), 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    # Use waitress to serve the app
    serve(app, host='0.0.0.0', port=5000)
