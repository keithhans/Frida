import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import gzip
import glob
import os
import argparse

class StrokeDataViewer:
    def __init__(self, stroke_data, param_data):
        self.strokes = stroke_data
        self.params = param_data
        self.current_index = 0
        
        # Create the figure and axis with more width for stats
        self.fig = plt.figure(figsize=(14, 8))  # Increased width from 10 to 14
        
        # Create main axis for the image with specific position and size
        self.ax = self.fig.add_axes([0.1, 0.2, 0.6, 0.7])  # [left, bottom, width, height]
        
        # Create the initial plot
        self.img_display = self.ax.imshow(self.strokes[0], cmap='gray', vmin=0, vmax=1)
        self.ax.set_title(f'Stroke {self.current_index + 1}/{len(self.strokes)}')
        
        # Create slider
        slider_ax = plt.axes([0.2, 0.05, 0.4, 0.03])
        self.slider = Slider(
            slider_ax, 'Stroke Index', 
            0, len(self.strokes)-1,
            valinit=0,
            valstep=1
        )
        self.slider.on_changed(self.update_image)
        
        # Create navigation buttons
        btn_prev_ax = plt.axes([0.2, 0.1, 0.1, 0.04])
        btn_next_ax = plt.axes([0.5, 0.1, 0.1, 0.04])
        self.btn_prev = Button(btn_prev_ax, 'Previous')
        self.btn_next = Button(btn_next_ax, 'Next')
        self.btn_prev.on_clicked(self.prev_stroke)
        self.btn_next.on_clicked(self.next_stroke)
        
        # Add statistics and parameters display
        self.stats_text = self.fig.text(
            0.75, 0.8, '',  # Moved to right side of figure
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        self.update_stats()

    def update_image(self, val):
        self.current_index = int(val)
        self.img_display.set_data(self.strokes[self.current_index])
        self.ax.set_title(f'Stroke {self.current_index + 1}/{len(self.strokes)}')
        self.update_stats()
        self.fig.canvas.draw_idle()

    def prev_stroke(self, event):
        self.current_index = max(0, self.current_index - 1)
        self.slider.set_val(self.current_index)

    def next_stroke(self, event):
        self.current_index = min(len(self.strokes) - 1, self.current_index + 1)
        self.slider.set_val(self.current_index)

    def update_stats(self):
        current_stroke = self.strokes[self.current_index]
        current_params = self.params[self.current_index]
        
        # Get parameters (length, bend, depth, alpha)
        length, bend, depth, alpha = current_params
        
        stats = f"Statistics:\n" \
               f"Min: {current_stroke.min():.3f}\n" \
               f"Max: {current_stroke.max():.3f}\n" \
               f"Mean: {current_stroke.mean():.3f}\n" \
               f"Std: {current_stroke.std():.3f}\n" \
               f"Shape: {current_stroke.shape}\n" \
               f"\nParameters:\n" \
               f"Length: {length:.3f}\n" \
               f"Bend: {bend:.3f}\n" \
               f"Depth: {depth:.3f}\n" \
               f"Alpha: {alpha:.3f}"
        self.stats_text.set_text(stats)

def load_stroke_data(file_path):
    """Load stroke intensity data from .npy file"""
    with gzip.GzipFile(file_path, 'r') as f:
        strokes = np.load(f, allow_pickle=True).astype(np.float32)/255.
    return strokes

def load_param_data(file_path):
    """Load stroke parameter data from .npy file"""
    param_file = file_path.replace('intensities', 'parameters')
    return np.load(param_file, allow_pickle=True, encoding='bytes')

def main():
    parser = argparse.ArgumentParser(description='Visualize stroke intensity data')
    parser.add_argument('path', type=str, help='Path to stroke_intensities*.npy file or directory')
    args = parser.parse_args()

    # Handle both single file and directory inputs
    if os.path.isdir(args.path):
        stroke_files = glob.glob(os.path.join(args.path, 'stroke_intensities*.npy*'))
        if not stroke_files:
            print(f"No stroke intensity files found in {args.path}")
            return
        
        # Load and concatenate all stroke data and parameters
        all_strokes = []
        all_params = []
        for file in sorted(stroke_files):
            print(f"Loading {file}...")
            strokes = load_stroke_data(file)
            params = load_param_data(file)
            all_strokes.append(strokes)
            all_params.append(params)
        strokes = np.concatenate(all_strokes)
        params = np.concatenate(all_params)
    else:
        if not os.path.exists(args.path):
            print(f"File not found: {args.path}")
            return
        strokes = load_stroke_data(args.path)
        params = load_param_data(args.path)

    print(f"Loaded {len(strokes)} strokes")
    
    # Create and show the viewer
    viewer = StrokeDataViewer(strokes, params)
    plt.show()

if __name__ == "__main__":
    main() 