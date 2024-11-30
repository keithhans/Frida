import requests
import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64

def load_img(path, h=None, w=None):
    # return data format [n, c, h, w]
    im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im = np.array(im)
    im = cv2.resize(im, (w,h)) if h is not None and w is not None else im
    im = torch.from_numpy(im)
    im = im.permute(2,0,1)
    return im.unsqueeze(0).float()

def encode_tensor(tensor):
    # Serialize tensor to bytes and encode as base64
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def compute_style_loss(target_path, current_path):
    # Load images
    target_tensor = load_img(target_path, 256, 256) / 255.
    current_tensor = load_img(current_path, 256, 256) / 255.
    
    # Prepare data for sending
    data = {
        'target': encode_tensor(target_tensor),
        'current': encode_tensor(current_tensor)
    }
    
    # Send request to server
    response = requests.post('http://localhost:6789/compute_style_loss', json=data)
    return response.json()['loss']

def compute_clip_loss(target_path, current_path):
    # Load images
    target_tensor = load_img(target_path, 224, 224) / 255.
    current_tensor = load_img(current_path, 224, 224) / 255.
    
    # Prepare data for sending
    data = {
        'target': encode_tensor(target_tensor),
        'current': encode_tensor(current_tensor)
    }
    
    # Send request to server
    response = requests.post('http://localhost:6789/compute_clip_loss', json=data)
    return response.json()['loss']

def compute_clip_text_loss(image_path, text, num_augs=30):
    # Load image
    image_tensor = load_img(image_path, 224, 224) / 255.
    
    # Prepare data for sending
    data = {
        'image': encode_tensor(image_tensor),
        'text': text,
        'num_augs': num_augs
    }
    
    # Send request to server
    response = requests.post('http://localhost:6789/compute_clip_text_loss', json=data)
    return response.json()['loss']

def compute_clip_fc_loss(target_path, current_path, num_augs=30):
    # Load images
    target_tensor = load_img(target_path, 224, 224) / 255.
    current_tensor = load_img(current_path, 224, 224) / 255.
    
    # Prepare data for sending
    data = {
        'target': encode_tensor(target_tensor),
        'current': encode_tensor(current_tensor),
        'num_augs': num_augs
    }
    
    # Send request to server
    response = requests.post('http://localhost:6789/compute_clip_fc_loss', json=data)
    return response.json()['loss']

def compute_dino_loss(target_path, current_path):
    # Load images
    target_tensor = load_img(target_path, 224, 224) / 255.
    current_tensor = load_img(current_path, 224, 224) / 255.
    
    # Prepare data for sending
    data = {
        'target': encode_tensor(target_tensor),
        'current': encode_tensor(current_tensor)
    }
    
    # Send request to server
    response = requests.post('http://localhost:6789/compute_dino_loss', json=data)
    return response.json()['loss']

if __name__ == "__main__":
    # Example usage
    target_path = '/Users/keith/Desktop/t.png'
    current_path = '/Users/keith/Desktop/r.png'
    
    style_loss = compute_style_loss(target_path, current_path)
    print(f"Style loss between images: {style_loss}")
    
    clip_loss = compute_clip_loss(target_path, current_path)
    print(f"CLIP conv loss between images: {clip_loss}")
    
    clip_text_loss = compute_clip_text_loss(target_path, "a beautiful painting")
    print(f"CLIP text loss: {clip_text_loss}")
    
    clip_fc_loss = compute_clip_fc_loss(target_path, current_path)
    print(f"CLIP FC loss between images: {clip_fc_loss}")
    
    dino_loss = compute_dino_loss(target_path, current_path)
    print(f"DINO loss between images: {dino_loss}") 