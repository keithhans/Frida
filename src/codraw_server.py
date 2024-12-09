from flask import Flask, request, jsonify
import torch
import io
import base64
import numpy as np
from options import Options
from painting_optimization import optimize_painting
import datetime
from my_tensorboard import TensorBoard
from cofrida import get_instruct_pix2pix_model
import random
from PIL import Image
from paint_utils3 import initialize_painting, format_img
from torchvision.transforms import Resize
import einops
import k_diffusion as K
from omegaconf import OmegaConf

sys.path.append("./instruct-pix2pix/stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config
import torch.nn as nn
from torch.cuda.amp import autocast

app = Flask(__name__)
device = torch.device('cuda')

# Initialize both models at startup
print("Loading COFRIDA model...")
cofrida_model = get_instruct_pix2pix_model(
    lora_weights_path="skeeterman/CoFRIDA-Sharpie", 
    device=device)
cofrida_model.set_progress_bar_config(disable=True)
print("COFRIDA model loaded successfully")

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


print("Loading InstructPix2Pix model...")
config = OmegaConf.load("instruct-pix2pix/configs/generate.yaml")
instruct_model = load_model_from_config(
    config, 
    "instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt"
).to(device)
instruct_model.eval()
model_wrap = K.external.CompVisDenoiser(instruct_model)
model_wrap_cfg = CFGDenoiser(model_wrap)
null_token = instruct_model.get_learned_conditioning([""])
print("InstructPix2Pix model loaded successfully")

# Add configuration for the switch
USE_INSTRUCT_PIX2PIX = True  # Default to INSTRUCT_PIX2PIX 

def decode_tensor(encoded_data):
    decoded = base64.b64decode(encoded_data)
    tensor_buffer = io.BytesIO(decoded)
    tensor = torch.load(tensor_buffer)
    return tensor.to(device)

def encode_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/set_generation_mode', methods=['POST'])
def set_generation_mode():
    """Endpoint to toggle between COFRIDA and InstructPix2Pix"""
    global USE_INSTRUCT_PIX2PIX
    data = request.json
    USE_INSTRUCT_PIX2PIX = data.get('use_instruct_pix2pix', False)
    return jsonify({'status': 'success', 'mode': 'instruct_pix2pix' if USE_INSTRUCT_PIX2PIX else 'cofrida'})

@app.route('/get_cofrida_image', methods=['POST'])
def get_cofrida_image_endpoint():
    global USE_INSTRUCT_PIX2PIX
    data = request.json
    
    # Setup Tensorboard
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('./painting_log/' + run_name)

    # Get current canvas and convert to PIL
    current_canvas = decode_tensor(data['current_canvas'])
    writer.add_image('images/current_canvas', format_img(current_canvas.permute(2, 0, 1).unsqueeze(0)/255.), 0)
    writer.add_text('text/prompt', data['prompt'], 0)
    
    target_imgs = []
    
    if USE_INSTRUCT_PIX2PIX:
        # Use InstructPix2Pix model
        current_canvas_tensor = 2 * torch.tensor(np.array(current_canvas)).float() / 255 - 1
        current_canvas_tensor = einops.rearrange(current_canvas_tensor, "h w c -> 1 c h w").to(device)
        
        with torch.no_grad(), autocast("cuda"):
            for i in range(data.get('n_options', 6)):
                # Prepare conditioning
                cond = {}
                cond["c_crossattn"] = [instruct_model.get_learned_conditioning([data['prompt']])]
                cond["c_concat"] = [instruct_model.encode_first_stage(current_canvas_tensor).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                # Sample
                steps = 20
                sigmas = model_wrap.get_sigmas(steps)
                text_cfg = 7.5 if i == 0 else random.uniform(6.0, 9.0)
                image_cfg = 1.5 if i == 0 else random.uniform(1.2, 1.8)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": text_cfg,
                    "image_cfg_scale": image_cfg,
                }

                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                x = instruct_model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * einops.rearrange(x, "1 c h w -> h w c")
                target_img = x.type(torch.uint8).cpu()
                
                target_imgs.append(target_img)
                writer.add_image('images/target_img', format_img(target_img.permute(2, 0, 1).unsqueeze(0)/255.), i)
    else:
        # Use original COFRIDA model
        current_canvas_pil = Image.fromarray(current_canvas.cpu().numpy().astype(np.uint8))
        with torch.no_grad():
            for i in range(data.get('n_options', 6)):
                image = cofrida_model(
                    data['prompt'],
                    current_canvas_pil,
                    num_inference_steps=20,
                    num_images_per_prompt=1,
                    image_guidance_scale=1.5 if i == 0 else random.uniform(1.01, 2.5)
                ).images[0]
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
    
    # Get all strokes and their positions
    strokes = painting.brush_strokes
    positions = [(stroke.transformation.xt.item(), stroke.transformation.yt.item()) 
                for stroke in strokes]
    
    # Sort strokes by proximity
    sorted_indices = []
    remaining_indices = list(range(len(strokes)))
    
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
    
    # Reorder strokes
    sorted_strokes = [strokes[i] for i in sorted_indices]
    
    # Serialize brush strokes in sorted order
    brush_strokes_data = []
    for stroke in sorted_strokes:
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
