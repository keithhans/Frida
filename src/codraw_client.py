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
from paint_utils3 import nearest_color, canvas_to_global_coordinates, show_img
import matplotlib.pyplot as plt
from PIL import Image
from my_tensorboard import TensorBoard

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
        
        date_and_time = datetime.datetime.now()
        run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
        self.opt.writer = TensorBoard('{}/{}'.format(self.opt.tensorboard_dir, run_name))
        self.opt.writer.add_text('args', str(sys.argv), 0)

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

    def _is_jsonable(self, x):
        try:
            import json
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    def get_cofrida_image(self, current_canvas, prompt, n_options=6):
        """Get COFRIDA image with user selection from multiple options"""
        data = {
            'current_canvas': encode_tensor(current_canvas),
            'prompt': prompt,
            'n_options': n_options
        }
        
        # Get multiple options from server
        response = requests.post(f'{self.server_url}/get_cofrida_image', json=data)
        response_data = response.json()
        
        # Decode all images
        target_imgs = [decode_tensor(img_data) for img_data in response_data['target_imgs']]
        
        # Show options to user
        fig, ax = plt.subplots(1, n_options, figsize=(2*n_options, 2))
        for j in range(n_options):
            ax[j].imshow(target_imgs[j].numpy())
            ax[j].set_xticks([])
            ax[j].set_yticks([])
            ax[j].set_title(str(j))
        plt.show()
        
        # Get user selection
        while True:
            try:
                target_img_ind = int(input("Type the number of the option you liked most? Type -1 if you don't like any and want more options.\n:"))
                if target_img_ind >= -1 and target_img_ind < n_options:
                    break
                print(f"Please enter a number between -1 and {n_options-1}")
            except ValueError:
                print("Please enter a valid number")
        
        # If user wants new options, recursively call this function
        if target_img_ind < 0:
            return self.get_cofrida_image(current_canvas, prompt, n_options)
        
        return target_imgs[target_img_ind]

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
            ##################################
            ########## Human Turn ###########
            ##################################
            
            # Get current canvas state
            current_canvas = self.painter.camera.get_canvas_tensor() / 255.
            current_canvas = torch.nn.functional.interpolate(
                current_canvas, 
                size=(self.h_render, self.w_render), 
                mode='bilinear',
                align_corners=False
            )
            
            # Let human draw
            try:
                input('\nFeel free to draw, then press enter when done.')
            except SyntaxError:
                pass

            # Get updated canvas after human drawing
            current_canvas = self.painter.camera.get_canvas_tensor() / 255.
            current_canvas = torch.nn.functional.interpolate(
                current_canvas, 
                size=(self.h_render, self.w_render), 
                mode='bilinear',
                align_corners=False
            )

            #################################
            ########## Robot Turn ###########
            #################################
            
            while True:  # Allow multiple attempts for robot turn
                # Get current canvas for COFRIDA
                curr_canvas = self.painter.camera.get_canvas()
                curr_canvas = cv2.cvtColor(curr_canvas, cv2.COLOR_BGR2RGB)
                curr_canvas_pil = Image.fromarray(curr_canvas.astype(np.uint8)).resize((512, 512))
                current_canvas = torch.from_numpy(np.array(curr_canvas_pil))

                # Get user input for prompt
                prompt = input("\nWhat would you like me to draw? Type 'done' if finished.\n:")
                if prompt.lower() == 'done':
                    return  # Exit the entire program
                
                # Get COFRIDA image with user selection
                target_img = self.get_cofrida_image(current_canvas, prompt)
                
                # Ask if user wants to try a different prompt
                retry = input("\nWould you like to try a different prompt? (y/n):\n")
                if retry.lower() != 'y':
                    break
                print("\nOk, let's try something else!")
            
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
        
        # Clean up at the end
        if not self.painter.opt.ink:
            self.painter.clean_paint_brush()
            self.painter.clean_paint_brush()
        
        self.painter.to_neutral()
        self.painter.robot.good_night_robot()

def main():
    client = CoDrawClient()
    client.run()

if __name__ == "__main__":
    main() 