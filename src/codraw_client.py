import torch
import datetime
import sys
from options import Options
from painter import Painter
from brush_stroke import BrushStroke
import requests
import io
import base64
from tqdm import tqdm
import cv2
import numpy as np
from paint_utils3 import nearest_color, canvas_to_global_coordinates

def encode_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_tensor(encoded_data):
    decoded = base64.b64decode(encoded_data)
    tensor_buffer = io.BytesIO(decoded)
    return torch.load(tensor_buffer)

class CoDrawClient:
    def __init__(self, server_url='http://localhost:6789'):
        self.server_url = server_url
        self.opt = Options()
        self.opt.gather_options()
        self.painter = Painter(self.opt)
        self.opt = self.painter.opt
        
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
    
    def _filter_options(self, options_dict):
        # Same as PaintClient._filter_options
        # ... (copy the method from PaintClient)
        pass

    def get_cofrida_image(self, current_canvas, prompt):
        data = {
            'current_canvas': encode_tensor(current_canvas),
            'prompt': prompt
        }
        
        response = requests.post(f'{self.server_url}/get_cofrida_image', json=data)
        response_data = response.json()
        
        return decode_tensor(response_data['target_img'])

    def optimize_painting_plan(self, current_canvas, target_img, num_strokes, turn_number):
        data = {
            'options': self._filter_options(vars(self.opt)),
            'current_canvas': encode_tensor(current_canvas),
            'target_img': encode_tensor(target_img),
            'num_strokes': num_strokes,
            'turn_number': turn_number
        }
        
        response = requests.post(f'{self.server_url}/optimize_painting_plan', json=data)
        response_data = response.json()
        
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
        
        color_palette = decode_tensor(response_data['color_palette']) if response_data['color_palette'] else None
        
        return brush_strokes, color_palette

    def run(self):
        self.painter.to_neutral()
        
        for i in range(9):  # Max number of turns
            # Get current canvas
            current_canvas = self.painter.camera.get_canvas_tensor() / 255.
            
            # Get user input for prompt
            prompt = input("\nWhat would you like me to draw? Type 'done' if finished.\n:")
            if prompt.lower() == 'done':
                break
            
            # Get COFRIDA image
            target_img = self.get_cofrida_image(current_canvas, prompt)
            
            # Get number of strokes
            num_strokes = int(input("How many strokes to use in this plan?\n:"))
            
            # Get optimized painting plan
            brush_strokes, color_palette = self.optimize_painting_plan(
                current_canvas, target_img, num_strokes, i)
            
            # Execute the plan
            if not self.painter.opt.simulate:
                show_img(self.painter.camera.get_canvas()/255., 
                        title="Initial plan complete. Ready to start painting."
                            + "Ensure mixed paint is provided and then exit this to "
                            + "start painting.")
            
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
                x, y = min(max(x,0.),1.), min(max(y,0.),1.)
                x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,self.painter.opt)
                stroke.execute(self.painter, x_glob, y_glob, stroke.transformation.a.item())

def main():
    client = CoDrawClient()
    client.run()

if __name__ == "__main__":
    main() 