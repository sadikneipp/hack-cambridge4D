import acquisition
from flask import Flask, jsonify, request
import logic
from PIL import Image
from io import BytesIO
import base64
import pygame
import os
import requests
import logging
import json
logging.basicConfig(level=logging.DEBUG)
model = logic.model_load()

app = Flask(__name__)
medications = ['ibuprofen', 'paracetamol']

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img = request.files['media']
    img.save('image.jpg')
    prediction = logic.predict(model)
    print(prediction)
    return jsonify({'alexaReq': str(int(prediction in medications)),
                    'medication': str(prediction).replace('_', ' ')})

if __name__ == "__main__":
    app.run(host='0.0.0.0')

