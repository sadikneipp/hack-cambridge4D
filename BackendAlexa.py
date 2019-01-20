import acquisition.camera
from flask import Flask, jsonify, request, send_file
# import logic
from PIL import Image
from io import BytesIO
import base64
import pygame
import os
import requests
import json

# model = logic.model_load()

app = Flask(__name__)
medications = ['ibuprofen', 'paracetamol']
CLOUD = 'http://104.197.248.132:5000/predict'

@app.route('/alexareq', methods=['GET', 'POST'])
def alexareq():
    webcam = acquisition.camera.init_camera()

    # img = acquisition.camera.snap_ss(webcam)
    # pil_string_image = pygame.image.tostring(img,"RGB",False)
    # im = Image.frombytes("RGB",(640, 480),pil_string_image)
    # buffered = BytesIO()
    # im.save(buffered, format="JPEG")
    # b64_im = base64.b64encode(buffered.getvalue())

    acquisition.camera.save_ss(webcam, 'image.jpg')
    acquisition.camera.stop_camera(webcam)
    files = {'media': open('image.jpg', 'rb')}

    post = requests.post(CLOUD, files=files)
    ans = json.loads(post.text)
    
    return jsonify(ans)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
