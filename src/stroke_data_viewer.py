import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import gzip
import glob
import os
import argparse

class StrokeDataViewer:
    def __init__(self, stroke_data):
        self.strokes = stroke_data
        self.current_index = 0
        
        # Create the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)  # Make room for slider
        
        # Create the initial plot
        self.img_display = self.ax.imshow(self.strokes[0], cmap='gray', vmin=0, vmax=1)
        self.ax.set_title(f'Stroke {self.current_index + 1}/{len(self.strokes)}')
        
        # Create slider
        slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(
            slider_ax, 'Stroke Index', 
            0, len(self.strokes)-1,
            valinit=0,
            valstep=1
        )
        self.slider.on_changed(self.update_image)
        
        # Create navigation buttons
        btn_prev_ax = plt.axes([0.2, 0.1, 0.1, 0.04])
        btn_next_ax = plt.axes([0.7, 0.1, 0.1, 0.04])
        self.btn_prev = Button(btn_prev_ax, 'Previous')
        self.btn_next = Button(btn_next_ax, 'Next')
        self.btn_prev.on_clicked(self.prev_stroke)
        self.btn_next.on_clicked(self.next_stroke)
        
        # Add statistics display
        self.stats_text = self.ax.text(
            1.05, 0.95, '', 
            transform=self.ax.transAxes,
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
        stats = f"Statistics:\n" \
               f"Min: {current_stroke.min():.3f}\n" \
               f"Max: {current_stroke.max():.3f}\n" \
               f"Mean: {current_stroke.mean():.3f}\n" \
               f"Std: {current_stroke.std():.3f}\n" \
               f"Shape: {current_stroke.shape}"
        self.stats_text.set_text(stats)

def load_stroke_data(file_path):
    """Load stroke intensity data from .npy file"""
    # if file_path.endswith('.npy.gz'):
    with gzip.GzipFile(file_path, 'r') as f:
        strokes = np.load(f, allow_pickle=True).astype(np.float32)/255.
    # else:
    #     strokes = np.load(file_path).astype(np.float32)/255.
    return strokes

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
        
        # Load and concatenate all stroke data
        all_strokes = []
        for file in sorted(stroke_files):
            print(f"Loading {file}...")
            strokes = load_stroke_data(file)
            all_strokes.append(strokes)
        strokes = np.concatenate(all_strokes)
    else:
        if not os.path.exists(args.path):
            print(f"File not found: {args.path}")
            return
        strokes = load_stroke_data(args.path)

    print(f"Loaded {len(strokes)} strokes")
    
    # Create and show the viewer
    viewer = StrokeDataViewer(strokes)
    plt.show()

if __name__ == "__main__":
    main() 