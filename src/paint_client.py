import requests
import torch
import io
import base64
from options import Options
import numpy as np
from camera.opencv_camera import WebCam
import matplotlib.pyplot as plt
import cv2

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

    def _is_jsonable(self, x):
        try:
            import json
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    def _filter_options(self, options_dict):
        filtered = {}
        for k, v in options_dict.items():
            # Skip special attributes
            if k.startswith('__'):
                continue
            
            # Handle None values - include them
            if v is None:
                filtered[k] = None
            # Handle nested dictionaries
            elif isinstance(v, dict):
                filtered[k] = self._filter_options(v)
            # Handle basic types
            elif isinstance(v, (int, float, str, bool)):
                filtered[k] = v
            # Handle lists/tuples of basic types
            elif isinstance(v, (list, tuple)):
                if all(isinstance(x, (int, float, str, bool, type(None))) for x in v):
                    filtered[k] = list(v)
            # Skip non-serializable objects (like ArgumentParser)
            elif not self._is_jsonable(v):
                continue
            
        return filtered

    def optimize_painting(self, n_strokes=100, optim_iter=100, ink=False, 
                        change_color=True, shuffle_strokes=True):
        # Get background image from camera
        background_img = self.cam.get_canvas_tensor() / 255.
        
        # Get all options including the nested 'opt' dictionary
        all_options = vars(self.opt)
        
        # Filter and prepare options for JSON serialization
        filtered_options = self._filter_options(all_options)
        
        # Make sure critical options are included
        # if 'opt' in filtered_options:
        #     filtered_options.update(filtered_options['opt'])
        
        # Add render dimensions if not present
        if 'h_render' not in filtered_options and 'render_height' in filtered_options:
            filtered_options['h_render'] = filtered_options['render_height']
        if 'w_render' not in filtered_options and 'render_height' in filtered_options:
            filtered_options['w_render'] = filtered_options['render_height']
        
        data = {
            'options': filtered_options,
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
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Show original image from camera
    plt.subplot(1, 2, 1)
    original = client.cam.get_canvas_tensor().numpy().transpose(1, 2, 0) / 255.
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show optimized painting
    plt.subplot(1, 2, 2)
    result = painting.numpy().transpose(1, 2, 0)
    plt.imshow(result)
    plt.title('Optimized Painting')
    plt.axis('off')
    
    # If we have a color palette, display it below
    if color_palette is not None:
        colors = color_palette.numpy()
        n_colors = len(colors)
        
        # Create a small subplot for the color palette
        plt.figure(figsize=(8, 1))
        for i, color in enumerate(colors):
            plt.subplot(1, n_colors, i+1)
            plt.imshow([[color]])
            plt.axis('off')
        plt.suptitle('Color Palette')
    
    plt.show()
    
    # Optionally save the result
    cv2.imwrite('optimized_painting.png', result[...,::-1] * 255)  # Convert RGB to BGR for cv2

if __name__ == "__main__":
    main() 