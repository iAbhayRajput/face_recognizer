import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from flask import Flask, jsonify, request
from deepface import DeepFace
import os
import cv2
from waitress import serve

app = Flask(__name__)

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
    required_files = ['image1', 'image2', 'image3', 'image4', 'image5']
    if not all(file in request.files for file in required_files):
        return jsonify({'error': 'Please provide image1, image2, image3, image4 and image5 files.'}), 400

    images = {}
    for file in required_files:
        image = request.files[file]
        image_path = os.path.join('uploads', image.filename)
        image.save(image_path)
        images[file] = image_path

    try:
        results = {}
        for i in range(2, 6):
            image_key = f'image{i}'
            verification, result = verify(images['image1'], images[image_key])
            results[image_key] = {'verified': verification, 'result': result}

        return jsonify(results), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred during verification: {str(e)}'}), 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    # Use waitress to serve the app (in production only)
    serve(app, host='0.0.0.0', port=5000)
