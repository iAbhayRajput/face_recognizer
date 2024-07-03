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
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please provide both image1 and image2 files.'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    image1_path = os.path.join('uploads', image1.filename)
    image2_path = os.path.join('uploads', image2.filename)

    image1.save(image1_path)
    image2.save(image2_path)

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
