from flask import Flask, request, jsonify
import torch
import io
import base64
import numpy as np
from options import Options
from painting_optimization import optimize_painting
from painting import Painting
from brush_stroke import BrushStroke
import random

app = Flask(__name__)
device = torch.device('cuda')

def decode_tensor(encoded_data):
    decoded = base64.b64decode(encoded_data)
    tensor_buffer = io.BytesIO(decoded)
    tensor = torch.load(tensor_buffer)
    return tensor.to(device)

def encode_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def random_init_painting(opt, background_img, n_strokes, ink=False):
    gridded_brush_strokes = []
    xys = [(x,y) for x in torch.linspace(-.95,.95,int(n_strokes**0.5)) 
           for y in torch.linspace(-.95,.95,int(n_strokes**0.5))]
    random.shuffle(xys)
    for x,y in xys:
        brush_stroke = BrushStroke(opt, xt=x, yt=y, ink=ink)
        gridded_brush_strokes.append(brush_stroke)
    painting = Painting(opt, 0, background_img=background_img, 
        brush_strokes=gridded_brush_strokes).to(device)
    return painting

@app.route('/optimize_painting', methods=['POST'])
def optimize_painting_endpoint():
    data = request.json
    
    # Reconstruct options
    opt = Options()
    opt.gather_options()
    for key, value in data['options'].items():
        setattr(opt, key, value)
    
    # Get background image
    background_img = decode_tensor(data['background_img'])
    
    # Initialize painting
    painting = random_init_painting(
        opt, 
        background_img, 
        data['n_strokes'], 
        ink=data.get('ink', False)
    )
    
    # Optimize painting
    painting, color_palette = optimize_painting(
        opt, 
        painting, 
        data['optim_iter'],
        color_palette=None,
        change_color=data.get('change_color', True),
        shuffle_strokes=data.get('shuffle_strokes', True)
    )
    
    # Prepare response
    response = {
        'painting': encode_tensor(painting.get_painting()),
        'color_palette': encode_tensor(color_palette) if color_palette is not None else None
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6789) 