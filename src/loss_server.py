from losses.style_loss import compute_style_loss
from losses.clip_loss import clip_conv_loss, clip_text_loss, clip_fc_loss
from losses.dino_loss import dino_loss
import torch
from flask import Flask, request, jsonify
import numpy as np
import io
import base64
import clip

app = Flask(__name__)
device = torch.device('cuda')

# Load CLIP model once at startup
clip_model, _ = clip.load("ViT-B/32", device=device)

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

@app.route('/compute_clip_text_loss', methods=['POST'])
def clip_text_loss_endpoint():
    data = request.json
    image_tensor = decode_image_data(data['image'])
    text = data['text']
    num_augs = data.get('num_augs', 30)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(clip.tokenize(text).to(device))
    
    loss = clip_text_loss(image_tensor, text_features, num_augs)
    
    return jsonify({'loss': loss.item()})

@app.route('/compute_clip_fc_loss', methods=['POST'])
def clip_fc_loss_endpoint():
    data = request.json
    target_tensor = decode_image_data(data['target'])
    current_tensor = decode_image_data(data['current'])
    num_augs = data.get('num_augs', 30)
    
    loss = clip_fc_loss(target_tensor, current_tensor, num_augs)
    
    return jsonify({'loss': loss.item()})

@app.route('/compute_dino_loss', methods=['POST'])
def dino_loss_endpoint():
    data = request.json
    target_tensor = decode_image_data(data['target'])
    current_tensor = decode_image_data(data['current'])
    
    loss = dino_loss(target_tensor, current_tensor)
    
    return jsonify({'loss': loss.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6789) 