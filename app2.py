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

def verify(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None:
        raise ValueError(f"Image at path {img1_path} could not be loaded")
    if img2 is None:
        raise ValueError(f"Image at path {img2_path} could not be loaded")
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name='Facenet')
        verification = result['verified']
        return verification, result
    except Exception as e:
        raise ValueError(f"Exception while processing {img2_path}: {str(e)}")

@app.route("/verify", methods=['POST'])
def verify_image():
    if 'employee_number' not in request.form or 'live_image' not in request.files:
        return jsonify({'error': 'Please provide an employee number and a live_image file.'}), 400

    employee_number = request.form['employee_number']
    live_image = request.files['live_image']

    if employee_number not in employee_images:
        return jsonify({'error': 'Invalid employee number or images not found.'}), 400

    # Retrieve the pre-stored image path
    image1_path = employee_images[employee_number]

    # Save the live-captured image
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    live_image_filename = f"_{employee_number}_{timestamp}.png"
    image2_path = os.path.join('uploads', live_image_filename)
    live_image.save(image2_path)

    try:
        verification, result = verify(image1_path, image2_path)
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
