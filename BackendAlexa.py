import acquisition
from flask import Flask, jsonify, request
import logic
from PIL import Image
from io import BytesIO
import base64

model = logic.model_load()

app = Flask(__name__)
medications = ['ibuprofen', 'paracetamol']


@app.route('/alexareq/', methods=['GET', 'POST'])
def alexareq():
    data = request.get_json()

    value = data['AlexaInput']
    # Open Pill Detector
    if value:

        logic.predict(logic.model_load(), )
        im = Image.open(acquisition.camera.save_ss(acquisition.camera.init_camera()))
        buffered = BytesIO()
        im.save(buffered, format="JPEG")
        b64_im = base64.b64encode(buffered.getvalue())

        return jsonify({'alexaReq': str(int(logic.predict(model, b64_im) in medications))})



if __name__ == "__main__":
    app.run(host='0.0.0.0')
