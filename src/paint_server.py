from flask import Flask, request, jsonify
import torch
import io
import base64
import numpy as np
from options import Options
from painting_optimization import load_objectives_data, optimize_painting
from painting import Painting
from brush_stroke import BrushStroke
import random
import datetime
from my_tensorboard import TensorBoard
from paint_utils3 import init_brush_strokes

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

def initialize_painting(opt, background_img, objective_img, n_strokes, ink, device='cuda'):
    attn = 1 - objective_img[0,:3].mean(dim=0)
    brush_strokes = init_brush_strokes(opt, attn, n_strokes, ink)
    painting = Painting(opt, 0, background_img=background_img, 
        brush_strokes=brush_strokes).to(device)
    return painting


@app.route('/optimize_painting', methods=['POST'])
def optimize_painting_endpoint():
    data = request.json
    
    # Reconstruct options
    opt = Options()
    opt.gather_options()
    for key, value in data['options'].items():
        setattr(opt, key, value)
    print(vars(opt))

    # Setup Tensorboard
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))

    # Get background image
    background_img = decode_tensor(data['background_img'])
    
    load_objectives_data(opt)

    # Initialize painting
    painting = initialize_painting(
        opt, 
        background_img, 
        opt.objective_data_loaded[0], # [0] is a hack. float
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
    
    # Get the rendered painting using forward pass
    with torch.no_grad():
        rendered_painting = painting(opt.h_render, opt.w_render, use_alpha=False)
    
    # Sort strokes by proximity
    positions = [(stroke.transformation.xt.item(), stroke.transformation.yt.item()) 
                for stroke in painting.brush_strokes]
    sorted_indices = []
    remaining_indices = list(range(len(painting.brush_strokes)))
    
    # Start with the leftmost stroke
    current_idx = min(remaining_indices, 
                     key=lambda i: positions[i][0])
    sorted_indices.append(current_idx)
    remaining_indices.remove(current_idx)
    
    # Add closest strokes one by one
    while remaining_indices:
        current_pos = positions[current_idx]
        # Find closest remaining stroke
        next_idx = min(remaining_indices,
                      key=lambda i: ((positions[i][0] - current_pos[0])**2 + 
                                   (positions[i][1] - current_pos[1])**2))
        sorted_indices.append(next_idx)
        remaining_indices.remove(next_idx)
        current_idx = next_idx
    
    # Serialize brush strokes in sorted order
    brush_strokes_data = []
    for idx in sorted_indices:
        stroke = painting.brush_strokes[idx]
        stroke_params = {
            'xt': stroke.transformation.xt.item(),
            'yt': stroke.transformation.yt.item(),
            'a': stroke.transformation.a.item(),  # rotation angle
            'length': stroke.stroke_length.item(),
            'bend': stroke.stroke_bend.item(),
            'z': stroke.stroke_z.item(),
            'alpha': stroke.stroke_alpha.item(),
            'color': stroke.color_transform.tolist() if hasattr(stroke, 'color_transform') else None,
            'ink': not hasattr(stroke, 'color_transform')  # if no color_transform, it's an ink stroke
        }
        brush_strokes_data.append(stroke_params)
    
    # Prepare response
    response = {
        'painting': encode_tensor(rendered_painting),
        'color_palette': encode_tensor(color_palette) if color_palette is not None else None,
        'brush_strokes': brush_strokes_data
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6789) 