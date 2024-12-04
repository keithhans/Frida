import requests
import torch
import io
import base64
from options import Options
import numpy as np
from camera.opencv_camera import WebCam
import matplotlib.pyplot as plt
import cv2
from brush_stroke import BrushStroke
from painter import Painter
from tqdm import tqdm
from utils import show_img, nearest_color, canvas_to_global_coordinates, get_colors

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
        self.painter = Painter(self.opt)  # Initialize robot painter
        self.opt = self.painter.opt  # Update options from painter
        
        # Set render dimensions
        self.w_render = int(self.opt.render_height * (self.opt.CANVAS_WIDTH_M/self.opt.CANVAS_HEIGHT_M))
        self.h_render = int(self.opt.render_height)
        self.opt.w_render, self.opt.h_render = self.w_render, self.h_render
        
        # Initialize painting state
        self.consecutive_paints = 0
        self.consecutive_strokes_no_clean = 0
        self.curr_color = -1
        
        self.color_palette = None
        if self.opt.use_colors_from is not None:
            self.color_palette = get_colors(
                cv2.resize(cv2.imread(self.opt.use_colors_from)[:,:,::-1], (256, 256)), 
                n_colors=self.opt.n_colors)

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
        background_img = self.painter.camera.get_canvas_tensor() / 255.
        # Save for visualization
        self.last_background = background_img.clone()
        
        # Get all options including the nested 'opt' dictionary
        all_options = vars(self.opt)
        
        # Filter and prepare options for JSON serialization
        filtered_options = self._filter_options(all_options)
        
        # Add render dimensions if not present
        if 'h_render' not in filtered_options['opt'] and 'render_height' in filtered_options['opt']:
            filtered_options['opt']['h_render'] = filtered_options['opt']['render_height']
        if 'w_render' not in filtered_options['opt'] and 'render_height' in filtered_options['opt']:
            filtered_options['opt']['w_render'] = filtered_options['opt']['render_height']

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
        
        # Reconstruct brush strokes
        brush_strokes = []
        for stroke_data in response_data['brush_strokes']:
            stroke = BrushStroke(
                self.opt,
                stroke_length=torch.tensor(stroke_data['length']),
                stroke_z=torch.tensor(stroke_data['z']),
                stroke_bend=torch.tensor(stroke_data['bend']),
                stroke_alpha=torch.tensor(stroke_data['alpha']),
                a=torch.tensor(stroke_data['a']),
                xt=torch.tensor(stroke_data['xt']),
                yt=torch.tensor(stroke_data['yt']),
                color=torch.tensor(stroke_data['color']) if stroke_data['color'] is not None else None,
                ink=stroke_data['ink']
            )
            brush_strokes.append(stroke)
        
        # Execute the plan
        if not self.painter.opt.simulate:
            show_img(self.painter.camera.get_canvas()/255., 
                    title="Initial plan complete. Ready to start painting."
                        + "Ensure mixed paint is provided and then exit this to "
                        + "start painting.")
        
        # Execute each stroke in the plan
        for stroke_ind in tqdm(range(len(brush_strokes)), desc="Executing plan"):
            stroke = brush_strokes[stroke_ind]
            
            # Handle paint/brush cleaning
            if not self.painter.opt.ink:
                color_ind, _ = nearest_color(stroke.color_transform.detach().cpu().numpy(), 
                                           color_palette.detach().cpu().numpy())
                new_paint_color = color_ind != self.curr_color
                if new_paint_color or self.consecutive_strokes_no_clean > 12:
                    self.painter.clean_paint_brush()
                    self.painter.clean_paint_brush()
                    self.consecutive_strokes_no_clean = 0
                    self.curr_color = color_ind
                    new_paint_color = True
                if self.consecutive_paints >= self.opt.how_often_to_get_paint or new_paint_color:
                    self.painter.get_paint(color_ind)
                    self.consecutive_paints = 0
            
            # Execute stroke
            x, y = stroke.transformation.xt.item()*0.5+0.5, stroke.transformation.yt.item()*0.5+0.5
            y = 1-y
            x, y = min(max(x,0.),1.), min(max(y,0.),1.)  # safety
            x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,self.painter.opt)
            stroke.execute(self.painter, x_glob, y_glob, stroke.transformation.a.item())
        
        self.painter.to_neutral()
        
        return painting, color_palette, brush_strokes

def main():
    client = PaintClient()
    client.painter.to_neutral()
    
    # Example usage
    painting, color_palette, brush_strokes = client.optimize_painting(
        n_strokes=400,
        optim_iter=400,
        ink=True
    )
    
    # Clean up at the end
    if not client.painter.opt.ink:
        client.painter.clean_paint_brush()
        client.painter.clean_paint_brush()
    
    client.painter.to_neutral()
    client.painter.robot.good_night_robot()

if __name__ == "__main__":
    main() 