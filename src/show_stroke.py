import torch
from param2stroke import get_param2img
from options import Options
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class StrokeVisualizer:
    def __init__(self, param2img, opt):
        self.param2img = param2img
        self.opt = opt
        
        # Create figure and axis with more width for params
        self.fig = plt.figure(figsize=(14, 8))  # Increased width from 10 to 14
        
        # Create main axis for the image with specific position and size
        self.ax = self.fig.add_axes([0.1, 0.2, 0.6, 0.7])  # [left, bottom, width, height]
        
        # Initial parameter values
        self.length = (opt.MIN_STROKE_LENGTH + opt.MAX_STROKE_LENGTH) / 2
        self.bend = 0.0
        self.depth = 0.9
        self.alpha = 0.0
        
        # Create initial plot
        self.img_display = self.update_stroke()
        self.ax.set_title('Generated Stroke')
        
        # Create sliders with adjusted positions
        length_ax = plt.axes([0.2, 0.2, 0.4, 0.03])  # Reduced width from 0.6 to 0.4
        bend_ax = plt.axes([0.2, 0.15, 0.4, 0.03])
        depth_ax = plt.axes([0.2, 0.1, 0.4, 0.03])
        alpha_ax = plt.axes([0.2, 0.05, 0.4, 0.03])
        
        self.length_slider = Slider(
            length_ax, 'Length', 
            opt.MIN_STROKE_LENGTH, opt.MAX_STROKE_LENGTH,
            valinit=self.length,
            valstep=0.001
        )
        self.bend_slider = Slider(
            bend_ax, 'Bend', 
            -opt.MAX_BEND, opt.MAX_BEND,
            valinit=self.bend,
            valstep=0.001
        )
        self.depth_slider = Slider(
            depth_ax, 'Depth', 0.1, 1.0, 
            valinit=self.depth,
            valstep=0.01
        )
        self.alpha_slider = Slider(
            alpha_ax, 'Alpha', -0.5, 0.5, 
            valinit=self.alpha,
            valstep=0.01
        )
        
        # Register update events
        self.length_slider.on_changed(self.update)
        self.bend_slider.on_changed(self.update)
        self.depth_slider.on_changed(self.update)
        self.alpha_slider.on_changed(self.update)
        
        # Add reset button with adjusted position
        reset_ax = plt.axes([0.5, 0.25, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset)
        
        # Add parameter display
        self.param_text = self.fig.text(
            0.75, 0.8, '',  # Moved to right side of figure
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        self.update_param_text()

    def update_stroke(self):
        params = torch.tensor(
            [self.length, self.bend, self.depth, self.alpha], 
            device='cpu'
        ).unsqueeze(0)
        
        with torch.no_grad():
            stroke_img = self.param2img(params, 200, 400)
        
        if hasattr(self, 'img_display'):
            self.img_display.set_data(stroke_img[0].cpu().numpy())
        else:
            self.img_display = self.ax.imshow(stroke_img[0].cpu().numpy(), cmap='gray')
        
        self.ax.set_axis_off()
        return self.img_display

    def update(self, val):
        self.length = self.length_slider.val
        self.bend = self.bend_slider.val
        self.depth = self.depth_slider.val
        self.alpha = self.alpha_slider.val
        
        self.update_stroke()
        self.update_param_text()
        self.fig.canvas.draw_idle()

    def reset(self, event):
        self.length_slider.reset()
        self.bend_slider.reset()
        self.depth_slider.reset()
        self.alpha_slider.reset()

    def update_param_text(self):
        params = f"Parameters:\n" \
                f"Length: {self.length:.3f}\n" \
                f"Bend: {self.bend:.3f}\n" \
                f"Depth: {self.depth:.3f}\n" \
                f"Alpha: {self.alpha:.3f}"
        self.param_text.set_text(params)

def main():
    opt = Options()
    opt.gather_options()

    # Initialize the param2img function
    param2img = get_param2img(opt, device='cpu')

    # Create and show the visualizer
    visualizer = StrokeVisualizer(param2img, opt)
    plt.show()

if __name__ == "__main__":
    main()
