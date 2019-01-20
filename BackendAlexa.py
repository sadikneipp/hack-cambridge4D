import acquisition
from flask import Flask, jsonify, request
import logic
from PIL import Image
from io import BytesIO
import base64
import pygame
import os

model = logic.model_load()

app = Flask(__name__)
medications = ['ibuprofen', 'paracetamol']

@app.route('/alexareq', methods=['GET', 'POST'])
def alexareq():
    webcam = acquisition.camera.init_camera()
    img = acquisition.camera.snap_ss(webcam)
    pil_string_image = pygame.image.tostring(img,"RGB",False)
    im = Image.frombytes("RGB",(640, 480),pil_string_image)

    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    b64_im = base64.b64encode(buffered.getvalue())
    prediction = logic.predict(model, b64_im)
    print(prediction)
    acquisition.camera.stop_camera(webcam)

    #for demo
    if len(os.listdir('demo')) > 0:
        os.remove('demo/' + os.listdir('demo')[0])
    im.save('demo/' + prediction + '.jpg')
    
    return jsonify({'alexaReq': str(int(prediction in medications)),
                    'medication': str(prediction).replace('_', ' ')})

if __name__ == "__main__":
    app.run(host='0.0.0.0')
