import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from flask import Flask, jsonify, request
from deepface import DeepFace
import cv2
from waitress import serve

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

def verify(img1_path, img2_paths):
    img1 = cv2.imread(img1_path)
    if img1 is None:
        raise ValueError(f"Image at path {img1_path} could not be loaded")

    results = []
    for img2_path in img2_paths:
        img2 = cv2.imread(img2_path)
        if img2 is None:
            raise ValueError(f"Image at path {img2_path} could not be loaded")
        try:
            result = DeepFace.verify(img1_path, img2_path, model_name='Facenet')
            results.append(result['verified'])
        except Exception as e:
            raise ValueError(f"Exception while processing {img2_path}: {str(e)}")

    average_verification = sum(results) / len(results)
    return average_verification, results

@app.route("/verify", methods=['POST'])
def verify_image():
    required_files = ['image1', 'image2', 'image3', 'image4', 'image5']
    if not all(file in request.files for file in required_files):
        return jsonify({'error': 'Please provide image1, image2, image3, image4 and image5 files.'}), 400

    if not os.path.exists('uploads'):
        try:
            os.makedirs('uploads')
        except OSError as e:
            app.logger.error(f"Error creating 'uploads' directory: {e}")
            return jsonify({'error': 'Failed to create uploads directory.'}), 500

    images = {}
    for file in required_files:
        image = request.files[file]
        image_path = os.path.join('uploads', image.filename)
        try:
            image.save(image_path)
            images[file] = image_path
        except FileNotFoundError as e:
            app.logger.error(f"FileNotFoundError: {e}")
            return jsonify({'error': 'Failed to save file to uploads directory.'}), 500

    try:
        average_verification, results = verify(images['image1'], [images['image2'], images['image3'], images['image4'], images['image5']])
        return jsonify({'average_verification': average_verification, 'results': results}), 200
    except ValueError as e:
        app.logger.error(f"ValueError: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Exception: {e}")
        return jsonify({'error': f'An error occurred during verification: {str(e)}'}), 500

if __name__ == "__main__":
    # Use waitress to serve the app (in production only)
    serve(app, host='0.0.0.0', port=5000)
