# training/unimodal/unimodal_xai.py

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Import Model ---
# This allows us to load the saved model using the same class definition
from unimodal_model import DinoV2Classifier

def load_trained_model(model_path, config):
    """Loads a trained model from a .pth file."""
    model = DinoV2Classifier(
        n_classes=config['model']['n_classes'],
        model_name=config['model']['name'],
        freeze_backbone=False # Set to false as we are loading all weights
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path, image_size=224):
    """Preprocesses a single image for model input."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0) # Add batch dimension
    return tensor, np.array(image.resize((image_size, image_size))) / 255.0

def generate_grad_cam(model, input_tensor, original_image):
    """Generates a Grad-CAM heatmap."""
    # The target layer is the last block of the transformer backbone
    target_layer = [model.backbone.blocks[-1].norm1]
    
    # Construct the Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layer)
    
    # Specify the target category for explanation (e.g., class 1 for malignant)
    targets = [ClassifierOutputTarget(1)]
    
    # Generate the CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Get the first (and only) CAM
    
    # Overlay the CAM on the original image
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    return visualization

def generate_attention_map(model, input_tensor):
    """
    Generates and visualizes the attention map from the model's last layer.
    """
    # Get the raw attention map from our model
    attention_map = model.get_attention_maps(input_tensor).squeeze(0)
    
    # The attention map is for patches, so we need to resize it to image dimensions
    num_patches = int(np.sqrt(attention_map.shape[-1] - 1)) # Exclude CLS token
    attention_map = attention_map[1:, 1:].reshape(num_patches, num_patches) # Reshape and remove CLS token interactions
    
    # Resize to image size
    resized_attention_map = transforms.functional.to_pil_image(attention_map.unsqueeze(0))
    resized_attention_map = resized_attention_map.resize((224, 224), Image.BICUBIC)
    
    # Convert to a heatmap
    plt.imshow(resized_attention_map, cmap='viridis')
    plt.axis('off')
    # Save to a buffer to return as an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

