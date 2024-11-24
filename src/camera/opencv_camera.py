#! /usr/bin/env python3
import cv2
import numpy as np
import torch
from torchvision.transforms import Resize

class WebCam():
    def __init__(self, opt,webcam_id=0):
        """Initialize the webcam using OpenCV VideoCapture
        Args:
            webcam_id (int): Camera device ID (default: 0)
        """
        self.webcam = cv2.VideoCapture(webcam_id)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open webcam with ID {webcam_id}")
        self.H_canvas = None
        self.opt = opt
        
    def get_rgb_image(self):
        """Capture and return an RGB image from the webcam"""
        ret, frame = self.webcam.read()
        if not ret:
            raise RuntimeError("Failed to capture image from webcam")
        return frame

    def get_canvas(self):
        """Get the perspective-corrected canvas view"""
        if self.H_canvas is None:
            self.calibrate_canvas()
        img = self.get_rgb_image()
        canvas = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        return canvas
    
    def get_canvas_tensor(self, h=None, w=None):
        """Get the canvas as a PyTorch tensor, optionally resized
        Args:
            h (int, optional): Target height
            w (int, optional): Target width
        Returns:
            torch.Tensor: Canvas image as tensor with shape [1, C, H, W]
        """
        canvas = self.get_canvas()
        # Convert BGR to RGB and normalize to [0,1]
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = torch.from_numpy(canvas).float() / 255.0
        canvas = canvas.permute(2, 0, 1).unsqueeze(0)
        if h is not None and w is not None:
            canvas = Resize((h, w), antialias=True)(canvas)
        return canvas

    def calibrate_canvas(self):
        """Interactive calibration to find canvas corners and compute homography"""
        import matplotlib.pyplot as plt
        img = self.get_rgb_image()
        h, w = img.shape[0], img.shape[1]
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img_rgb)
        plt.title("Select corners of canvas. First is top-left, then clock-wise.")
        self.canvas_points = np.array(plt.ginput(n=4))
        true_points = np.array([[0,0], [w,0], [w,h], [0,h]])
        self.H_canvas, _ = cv2.findHomography(self.canvas_points, true_points)
        
        # Show the warped result
        img1_warp = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        img1_warp_rgb = cv2.cvtColor(img1_warp, cv2.COLOR_BGR2RGB)
        
        plt.figure()
        plt.imshow(img1_warp_rgb)
        plt.title('Hopefully this looks like just the canvas')
        plt.show()

    def __del__(self):
        """Release the webcam when the object is destroyed"""
        if hasattr(self, 'webcam'):
            self.webcam.release()

    def test(self):
        """Test function to display both raw and canvas-corrected images"""
        import matplotlib.pyplot as plt
        
        # Get raw image
        raw_img = self.get_rgb_image()
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Get canvas view
        canvas = self.get_canvas()
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Display results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(raw_img_rgb)
        ax1.set_title('Raw Camera Image')
        ax1.axis('off')
        
        ax2.imshow(canvas_rgb)
        ax2.set_title('Canvas View')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test OpenCV WebCam implementation')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    args = parser.parse_args()
    
    try:
        # Initialize camera
        print(f"Initializing camera with ID {args.camera_id}")
        cam = WebCam(webcam_id=args.camera_id)
        
        # Test basic image capture
        print("Testing basic image capture...")
        img = cam.get_rgb_image()
        print(f"Successfully captured image with shape: {img.shape}")
        
        # Calibrate canvas and test canvas view
        print("\nStarting canvas calibration...")
        print("Please click on the four corners of the canvas (top-left, then clockwise)")
        cam.test()
        
        # Test tensor output
        print("\nTesting tensor output...")
        tensor = cam.get_canvas_tensor(h=224, w=224)
        print(f"Canvas tensor shape: {tensor.shape}")
        print(f"Tensor value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        print("\nTest completed")
