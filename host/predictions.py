from flask import Flask
from flask import request
from flask import json
from utils import load_obj
from utils import evaluate
from model import RNN
import torch as t

app = Flask(__name__)
char2index = load_obj('char2index')
index2char = load_obj('index2char')
n_chars = 67
hidden_size = 256
n_layers = 1

@app.route('/', methods=['POST'])
def index():
    payload = json.loads(request.get_data().decode('utf-8'))
    prediction = predict(payload)
    data = {}
    data['data'] = prediction
    return json.dumps(data)

def load_model():
    decoder = RNN(n_chars, hidden_size, n_chars, n_layers)
    decoder.load_state_dict(t.load('0.0011890831945547417_3.pth'))
    return decoder

def predict(data):
    startString = data['start']
    predict_len = int(data['len'])
    temperature = float(data['temperature'])
    decoder = load_model()
    
    return evaluate(startString,predict_len,temperature,index2char, char2index, decoder)