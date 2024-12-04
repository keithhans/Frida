import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
import torch
from brush_stroke import BrushStroke

def preprocess_sketch(image_path, threshold=127):
    """Preprocess sketch image to binary"""
    # Read and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Threshold to binary
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_contours(binary_img):
    """Find contours in binary image"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours

def simplify_contour(contour, epsilon_factor=0.01):
    """Simplify contour using Douglas-Peucker algorithm"""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def fit_bezier_curves(points, n_curves=1):
    """Fit Bezier curves to points"""
    # Convert points to numpy array
    points = np.array(points).reshape(-1, 2)

    print(f"Points shape: {points.shape}")
    
    # Check if we have enough points for spline fitting
    if points.shape[0] < 4:  # Need at least 4 points for cubic spline
        print(f"Not enough points ({points.shape[0]}) for spline fitting, using linear interpolation")
        # Use linear interpolation instead
        t = np.linspace(0, 1, n_curves + 1)
        control_points = []
        for i in range(n_curves):
            # Linear interpolation between start and end
            p0 = points[0] if i == 0 else points[-1] * t[i]
            p3 = points[-1] * t[i+1]
            
            # Simple control points at thirds
            p1 = p0 + (p3 - p0) / 3
            p2 = p0 + 2 * (p3 - p0) / 3
            
            control_points.append([p0, p1, p2, p3])
        return control_points
    
    try:
        # Try fitting B-spline
        tck, u = splprep([points[:,0], points[:,1]], k=min(3, points.shape[0]-1), s=0)
        
        # Sample points along the spline
        u_new = np.linspace(0, 1, n_curves + 1)
        x_new, y_new = splev(u_new, tck)
        
        # Convert to control points
        control_points = []
        for i in range(n_curves):
            p0 = np.array([x_new[i], y_new[i]])
            p3 = np.array([x_new[i+1], y_new[i+1]])
            
            # Estimate control points using tangent vectors
            tangent = np.array([
                x_new[i+1] - x_new[i],
                y_new[i+1] - y_new[i]
            ]) / 3.0
            
            p1 = p0 + tangent
            p2 = p3 - tangent
            
            control_points.append([p0, p1, p2, p3])
        
        return control_points
    
    except Exception as e:
        print(f"Spline fitting failed: {e}, falling back to linear interpolation")
        # Fall back to linear interpolation
        t = np.linspace(0, 1, n_curves + 1)
        control_points = []
        for i in range(n_curves):
            # Linear interpolation between start and end
            p0 = points[0] if i == 0 else points[-1] * t[i]
            p3 = points[-1] * t[i+1]
            
            # Simple control points at thirds
            p1 = p0 + (p3 - p0) / 3
            p2 = p0 + 2 * (p3 - p0) / 3
            
            control_points.append([p0, p1, p2, p3])
        return control_points

def bezier_to_brush_strokes(control_points, opt):
    """Convert Bezier curves to brush strokes"""
    brush_strokes = []
    
    for curve in control_points:
        # Extract parameters from Bezier curve
        p0, p1, p2, p3 = curve
        
        # Calculate stroke parameters
        length = np.linalg.norm(p3 - p0)
        angle = np.arctan2(p3[1] - p0[1], p3[0] - p0[0])
        
        # Calculate bend from control points
        mid_point = (p0 + p3) / 2
        control_mid = (p1 + p2) / 2
        bend = np.linalg.norm(control_mid - mid_point)
        if np.cross(p3 - p0, control_mid - mid_point) < 0:
            bend = -bend
        
        # Create brush stroke
        stroke = BrushStroke(
            opt,
            stroke_length=torch.tensor(length),
            stroke_bend=torch.tensor(bend),
            stroke_z=torch.tensor(0.5),  # Default z
            stroke_alpha=torch.tensor(0.0),  # Default alpha
            a=torch.tensor(angle),
            xt=torch.tensor(p0[0]),
            yt=torch.tensor(p0[1]),
            ink=True,
            device='cpu'
        )
        
        brush_strokes.append(stroke)
    
    return brush_strokes

def sketch_to_strokes(image_path, opt, max_strokes=100):
    """Convert sketch image to brush strokes"""
    # Preprocess image
    binary = preprocess_sketch(image_path)
    
    # Find contours
    contours = find_contours(binary)
    
    # Simplify contours
    simplified_contours = [simplify_contour(c) for c in contours]
    
    # Fit Bezier curves to each contour
    all_control_points = []
    for contour in simplified_contours:
        # Estimate number of curves based on contour length
        n_curves = max(1, int(cv2.arcLength(contour, True) / 50))
        control_points = fit_bezier_curves(contour, n_curves)
        all_control_points.extend(control_points)
    
    # Limit total number of curves
    if len(all_control_points) > max_strokes:
        # Sort by curve length and take the longest ones
        curve_lengths = [np.linalg.norm(c[3] - c[0]) for c in all_control_points]
        indices = np.argsort(curve_lengths)[-max_strokes:]
        all_control_points = [all_control_points[i] for i in indices]
    
    # Convert to brush strokes
    brush_strokes = bezier_to_brush_strokes(all_control_points, opt)
    
    return brush_strokes

def main():
    # Example usage
    from options import Options
    
    opt = Options()
    opt.gather_options()
    
    image_path = "/Users/keith/Downloads/lonely_expert.png"
    brush_strokes = sketch_to_strokes(image_path, opt)
    
    print(f"Converted sketch to {len(brush_strokes)} brush strokes")

if __name__ == "__main__":
    main() 