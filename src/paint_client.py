import requests
import torch
import io
import base64
from options import Options
import numpy as np
from camera.opencv_camera import WebCam

def encode_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_tensor(encoded_data):
    decoded = base64.b64decode(encoded_data)
    tensor_buffer = io.BytesIO(decoded)
    return torch.load(tensor_buffer)

class PaintClient:
    def __init__(self, server_url='http://localhost:6789'):
        self.server_url = server_url
        self.opt = Options()
        self.opt.gather_options()
        self.cam = WebCam()

    def optimize_painting(self, n_strokes=100, optim_iter=100, ink=False, 
                        change_color=True, shuffle_strokes=True):
        # Get background image from camera
        background_img = self.cam.get_canvas_tensor() / 255.
        
        # Prepare data for server
        data = {
            'options': {k: v for k, v in vars(self.opt).items() 
                       if not k.startswith('__') and isinstance(v, (int, float, str, bool))},
            'background_img': encode_tensor(background_img),
            'n_strokes': n_strokes,
            'optim_iter': optim_iter,
            'ink': ink,
            'change_color': change_color,
            'shuffle_strokes': shuffle_strokes
        }
        
        # Send request to server
        response = requests.post(f'{self.server_url}/optimize_painting', json=data)
        response_data = response.json()
        
        # Decode response
        painting = decode_tensor(response_data['painting'])
        color_palette = decode_tensor(response_data['color_palette']) if response_data['color_palette'] else None
        
        return painting, color_palette

def main():
    client = PaintClient()
    
    # Example usage
    painting, color_palette = client.optimize_painting(
        n_strokes=100,
        optim_iter=100,
        ink=True
    )
    
    # Save or display results
    # ... (add your visualization code here)

if __name__ == "__main__":
    main() 