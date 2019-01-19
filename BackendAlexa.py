import acquisition
import json
import time
from flask import Flask, jsonify, request


app = Flask(__name__)

state = {
    'john': {'balance': 3000, 'credit': 999, 'rewards': 1337},
    'mary': {'balance': 200, 'credit': 700, 'rewards': 500},
    'leonard': {'balance': 0, 'credit': 100, 'rewards': 1}
}
AUTH_THRESH = 60
auth = time.time() - 60

ip_alexa = '82.132.222.99'


@app.route('/alexareq/', methods=['GET', 'POST'])
def alexareq():
    data = request.get_json()

    value = data['AlexaInput']
    # Open Pill Detector
    if value == 'Alexa, open pill detector':
        acquisition.camera.save_ss(acquisition.camera.init_camera(), '../Images')


@app.route('/alexares/', methods=['GET', 'POST'])
def alexares():
    data = request.get_json()
    value = data['AlexaOutput']
    # Get Yes or No from Alexa
    if value == 'Yes':
        return jsonify({'responseForAlexa': 'Yes'})
    elif value == 'No':
        return jsonify({'responseForAlexa': 'No'})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
