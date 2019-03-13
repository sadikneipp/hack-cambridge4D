import acquisition.camera
from flask import Flask, jsonify, request, send_file
from PIL import Image
from io import BytesIO
import base64
import pygame
import os
import requests
import json

app = Flask(__name__)
CLOUD = 'http://104.197.248.132:5000/predict'

@app.route('/alexareq', methods=['GET', 'POST'])
def alexareq():
    webcam = acquisition.camera.init_camera()
    acquisition.camera.save_ss(webcam, 'image.jpg')
    acquisition.camera.stop_camera(webcam)
    files = {'media': open('image.jpg', 'rb')}

    post = requests.post(CLOUD, files=files)
    ans = json.loads(post.text)
    
    return jsonify(ans)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
