from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import os
import base64
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://face-analyzer-frontend-51vs.vercel.app"])



def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.route('/analyze', methods=['POST'])

def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://face-analyzer-frontend-51vs.vercel.app"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

def analyze():
    try:
        data = request.json
        image_base64 = data.get('image')
        img = decode_base64_image(image_base64)

        analysis = DeepFace.analyze(img_path=img, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

        return jsonify({
            'emotion': analysis[0]['dominant_emotion'],
            'age': analysis[0]['age'],
            'gender': analysis[0]['dominant_gender'],
            'race': analysis[0]['dominant_race']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
