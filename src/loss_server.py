from losses.style_loss import compute_style_loss
from losses.clip_loss import clip_conv_loss
import torch
from flask import Flask, request, jsonify
import numpy as np
import io
import base64

app = Flask(__name__)
device = torch.device('cuda')

def decode_image_data(encoded_data):
    # Decode base64 string to tensor
    decoded = base64.b64decode(encoded_data)
    tensor_buffer = io.BytesIO(decoded)
    tensor = torch.load(tensor_buffer)
    return tensor.to(device)

@app.route('/compute_style_loss', methods=['POST'])
def style_loss_endpoint():
    data = request.json
    target_tensor = decode_image_data(data['target'])
    current_tensor = decode_image_data(data['current'])
    
    loss = compute_style_loss(target_tensor, current_tensor)
    
    return jsonify({'loss': loss.item()})

@app.route('/compute_clip_loss', methods=['POST'])
def clip_loss_endpoint():
    data = request.json
    target_tensor = decode_image_data(data['target'])
    current_tensor = decode_image_data(data['current'])
    
    loss = clip_conv_loss(target_tensor, current_tensor)
    
    return jsonify({'loss': loss.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6789) 