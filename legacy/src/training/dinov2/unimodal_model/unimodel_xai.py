# training/unimodal/unimodal_xai.py

import torch
from torchvision import transforms
import numpy as np
import yaml
import argparse
import os
import logging
import io
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Import Model ---
from src.training.unimodal.unimodal_model import DinoV2Classifier

def load_config(config_path):
    """Loads the model configuration from a YAML file."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info("Configuration loaded successfully.")
    return config

def load_trained_model(model_path, config):
    """Loads a trained model from a .pth file."""
    logging.info(f"Loading model from: {model_path}")
    model = DinoV2Classifier(
        n_classes=config['model']['n_classes'],
        model_name=config['model']['name'],
        dropout_rate=config['training'].get('dropout_rate', 0.5) 
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logging.info("Model loaded successfully.")
    return model

def preprocess_image(image_path, image_size=224):
    """Preprocesses a single image for model input."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logging.error(f"Image not found at {image_path}. Please check the path.")
        return None, None
    tensor = transform(image).unsqueeze(0)
    # Return the resized original image as a numpy array for visualization
    return tensor, np.array(image.resize((image_size, image_size))) / 255.0

def generate_grad_cam(model, input_tensor, original_image):
    """Generates a Grad-CAM heatmap for a Vision Transformer."""
    target_layer = [model.backbone.blocks[-1].norm1]
    
    def reshape_transform(tensor):
        patch_tokens = tensor[:, 1:, :]
        num_patches = patch_tokens.shape[1]
        height = width = int(num_patches**0.5)
        result = patch_tokens.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)
        return result

    cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform)
    
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)

# def generate_attention_map(model, input_tensor):
#     """
#     Generates and visualizes the raw attention map from the model's last layer.
#     """
#     attention_map = model.get_attention_maps(input_tensor).squeeze(0)
#     num_patches = int(np.sqrt(attention_map.shape[-1] - 1))
#     attention_map = attention_map[1:, 1:].reshape(num_patches, num_patches)
    
#     resized_attention_map = transforms.functional.to_pil_image(attention_map.unsqueeze(0)).resize((224, 224), Image.BICUBIC)
    
#     fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
#     ax.imshow(resized_attention_map, cmap='viridis')
#     ax.axis('off')
    
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     buf.seek(0)
#     img = Image.open(buf)
#     plt.close(fig)
#     return img

def main():
    parser = argparse.ArgumentParser(description="Generate XAI visualizations for unimodal models.")
    parser.add_argument('--config', type=str, default='/teamspace/studios/this_studio/MMIBC/src/training/unimodal/config.yaml', help='Path to the model configuration file.')
    parser.add_argument('--ultrasound_model', type=str, default="/teamspace/studios/this_studio/saved_models/best_model_final.pth", help='Path to the trained ultrasound model (.pth).')
    parser.add_argument('--mammo_model', type=str, default="/teamspace/studios/this_studio/saved_models/best_mammo_model_final.pth", help='Path to the trained mammography model (.pth).')
    parser.add_argument('--ultrasound_image', type=str, default="/teamspace/studios/this_studio/mmibc/paired/ultrasound/test/malignant/0029.png", help='Path to a sample ultrasound image.')
    parser.add_argument('--mammo_image', type=str, default="/teamspace/studios/this_studio/mmibc/paired/mammo/test/malignant/2017_BC019721_ MLO_R.png", help='Path to a sample mammography image.')
    parser.add_argument('--output', type=str, default='xai_evaluation.png', help='Path to save the combined output image.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = load_config(args.config)
    
    # --- Process Ultrasound Model ---
    us_model = load_trained_model(args.ultrasound_model, config)
    us_tensor, us_original_img = preprocess_image(args.ultrasound_image)
    if us_tensor is None: return
    
    logging.info("Generating XAI for Ultrasound model...")
    us_grad_cam = generate_grad_cam(us_model, us_tensor, us_original_img)
    
    # --- Process Mammography Model ---
    mammo_model = load_trained_model(args.mammo_model, config)
    mammo_tensor, mammo_original_img = preprocess_image(args.mammo_image)
    if mammo_tensor is None: return

    logging.info("Generating XAI for Mammography model...")
    mammo_grad_cam = generate_grad_cam(mammo_model, mammo_tensor, mammo_original_img)
    
    # --- Create Combined Visualization ---
    logging.info(f"Stitching visualizations and saving to {args.output}...")
    # Create a 2x2 grid for the plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Ultrasound Row
    axes[0, 0].imshow(us_original_img)
    axes[0, 0].set_title('Original Ultrasound Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(us_grad_cam)
    axes[0, 1].set_title('Ultrasound: Grad-CAM')
    axes[0, 1].axis('off')

    # Mammography Row
    axes[1, 0].imshow(mammo_original_img)
    axes[1, 0].set_title('Original Mammogram Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mammo_grad_cam)
    axes[1, 1].set_title('Mammography: Grad-CAM')
    axes[1, 1].axis('off')
    
    plt.tight_layout(pad=1.5)
    plt.savefig(args.output)
    logging.info("Done.")
    
    return args.output

if __name__ == '__main__':
    result_path = main()
    if result_path:
        print(f"XAI evaluation complete. Output image saved to: {result_path}")