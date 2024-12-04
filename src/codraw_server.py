from flask import Flask, request, jsonify
import torch
import io
import base64
import numpy as np
from options import Options
from painting_optimization import load_objectives_data, optimize_painting
from painting import Painting
import datetime
from my_tensorboard import TensorBoard
from cofrida import get_instruct_pix2pix_model
import random
from PIL import Image
from paint_utils3 import initialize_painting
from torchvision.transforms import Resize
from paint_utils3 import format_img
app = Flask(__name__)
device = torch.device('cuda')

# Initialize COFRIDA model at startup
print("Loading COFRIDA model...")
cofrida_model = get_instruct_pix2pix_model(
    lora_weights_path="skeeterman/CoFRIDA-Sharpie", 
    device=device)
cofrida_model.set_progress_bar_config(disable=True)
print("COFRIDA model loaded successfully")

def decode_tensor(encoded_data):
    decoded = base64.b64decode(encoded_data)
    tensor_buffer = io.BytesIO(decoded)
    tensor = torch.load(tensor_buffer)
    return tensor.to(device)

def encode_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/get_cofrida_image', methods=['POST'])
def get_cofrida_image_endpoint():
    data = request.json
    
    # Setup Tensorboard
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('./painting_log/' + run_name)

    # Get current canvas and convert to PIL
    current_canvas = decode_tensor(data['current_canvas'])
    current_canvas_pil = Image.fromarray(current_canvas.cpu().numpy().astype(np.uint8))
    writer.add_image('images/current_canvas', format_img(current_canvas.permute(2, 0, 1).unsqueeze(0)/255.), 0)
    
    # Generate multiple COFRIDA images
    target_imgs = []
    with torch.no_grad():
        for i in range(data.get('n_options', 6)):
            image = cofrida_model(
                data['prompt'],
                current_canvas_pil,  # PIL Image input
                num_inference_steps=20,
                num_images_per_prompt=1,
                image_guidance_scale=1.5 if i == 0 else random.uniform(1.01, 2.5)
            ).images[0]  # Returns PIL Image
            target_img = torch.from_numpy(np.array(image)).cpu()    
            target_imgs.append(target_img)
            writer.add_image('images/target_img', format_img(target_img.permute(2, 0, 1).unsqueeze(0)/255.), i)
    
    return jsonify({
        'target_imgs': [encode_tensor(img) for img in target_imgs]
    })

@app.route('/optimize_painting_plan', methods=['POST'])
def optimize_painting_plan_endpoint():
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
    
    # Get images
    current_canvas = decode_tensor(data['current_canvas']).permute(2, 0, 1).unsqueeze(0)/255.
    current_canvas = Resize((opt.h_render, opt.w_render), antialias=True)(current_canvas)
    target_img = decode_tensor(data['target_img']).permute(2, 0, 1).unsqueeze(0)/255.
    target_img = Resize((opt.h_render, opt.w_render), antialias=True)(target_img)

    print(f"current_canvas type: {type(current_canvas)}")
    print(f"current_canvas shape: {current_canvas.shape}")
    print(f"target_img type: {type(target_img)}")
    print(f"target_img shape: {target_img.shape}")
    print(f"Sample current_canvas value: {current_canvas[0,0,0,0].item()}")  # First element
    print(f"Sample target_img value: {target_img[0,0,0,0].item()}")  # First element

    processed_target_img = format_img(target_img)
    opt.writer.add_image('images/target_img', processed_target_img, 0)

    # Set objectives
    opt.objective = ['clip_conv_loss']
    opt.objective_data_loaded = [target_img]
    opt.objective_weight = [1.0]
    
    # Initialize and optimize painting
    painting = initialize_painting(opt, data['num_strokes'], target_img, 
                               current_canvas, opt.ink, device=device)
    
    painting, color_palette = optimize_painting(opt, painting, 
                optim_iter=opt.optim_iter, color_palette=None,
                log_title=f"{data['turn_number']}_3_plan")
    
    # Serialize brush strokes
    brush_strokes_data = []
    for stroke in painting.brush_strokes:
        stroke_params = {
            'xt': stroke.transformation.xt.item(),
            'yt': stroke.transformation.yt.item(),
            'a': stroke.transformation.a.item(),
            'length': stroke.stroke_length.item(),
            'bend': stroke.stroke_bend.item(),
            'z': stroke.stroke_z.item(),
            'alpha': stroke.stroke_alpha.item(),
            'color': stroke.color_transform.tolist() if hasattr(stroke, 'color_transform') else None,
            'ink': not hasattr(stroke, 'color_transform')
        }
        brush_strokes_data.append(stroke_params)
    
    return jsonify({
        'brush_strokes': brush_strokes_data,
        'color_palette': encode_tensor(color_palette) if color_palette is not None else None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6789) 
