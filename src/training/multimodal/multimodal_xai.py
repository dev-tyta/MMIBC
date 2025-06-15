# training/multimodal/multimodal_xai.py

import torch
from torchvision import transforms
import yaml
import argparse
import os
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import io

# --- Import Model Architectures ---
from src.training.multimodal.multimodal_architecture import MultimodalFusionModel

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class UnimodalModelWrapper(torch.nn.Module):
    """
    A simple wrapper to ensure the model's forward pass is called correctly
    by the Grad-CAM library.
    """
    def __init__(self, model):
        super(UnimodalModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # This standard forward pass is what the library expects.
        return self.model(x)

def preprocess_image(image_path, image_size=224):
    """Preprocesses a single image for model input."""
    # (Same preprocessing as unimodal evaluation)
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
    return tensor, np.array(image.resize((image_size, image_size))) / 255.0

def generate_multimodal_grad_cam(model, target_modality, input_mammo, input_us, original_image):
    """
    Generates a Grad-CAM heatmap for one of the model's unimodal branches.

    Args:
        model (nn.Module): The trained MultimodalFusionModel.
        target_modality (str): Either 'mammo' or 'us'. Specifies which branch to explain.
        input_mammo (torch.Tensor): The preprocessed mammogram tensor.
        input_us (torch.Tensor): The preprocessed ultrasound tensor.
        original_image (np.array): The resized original image for visualization.
    """

    wrapped_model = UnimodalModelWrapper(model)
    

    if target_modality == 'mammo':
        target_layers = [wrapped_model.mammo_encoder.backbone.blocks[-1].norm1]
    elif target_modality == 'us':
        target_layers = [wrapped_model.us_encoder.backbone.blocks[-1].norm1]
    else:
        raise ValueError("target_modality must be 'mammo' or 'us'")

    def reshape_transform(tensor):
        patch_tokens = tensor[:, 1:, :]
        num_patches = patch_tokens.shape[1]
        height = width = int(num_patches**0.5)
        result = patch_tokens.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)
        return result
    
    if target_modality == 'mammo':
        cam = GradCAM(model=wrapped_model.mammo_encoder, target_layers=target_layers, reshape_transform=reshape_transform)
        inputs = input_mammo.unsqueeze(0)  # Add batch dimension
    elif target_modality == 'us':
        cam = GradCAM(model=wrapped_model.us_encoder, target_layers=target_layers, reshape_transform=reshape_transform)
        inputs = input_us.unsqueeze(0)  # Add batch dimension
    
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=inputs, targets=targets)[0, :]
    
    # visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    if target_modality == 'mammo':
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    else:
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    return Image.fromarray(visualization)

def main():
    parser = argparse.ArgumentParser(description="Generate XAI visualizations for the multimodal model.")
    parser.add_argument('--config', type=str, default="/teamspace/studios/this_studio/MMIBC/src/training/multimodal/config.yaml", help='Path to the multimodal config file.')
    parser.add_argument('--model_path', type=str, default="/teamspace/studios/this_studio/saved_models/best_multimodal_model.pth", help='Path to the trained multimodal model (.pth).')
    parser.add_argument('--mammo_image', type=str, default="/teamspace/studios/this_studio/mmibc/mammo/test/malignant/1a1d13244aaafa6a12988c4e1d3efe5b.png", help='Path to a sample mammography image.')
    parser.add_argument('--us_image', type=str, default="/teamspace/studios/this_studio/mmibc/ultrasound/images/test/benign/benign (66).png", help='Path to a sample ultrasound image.')
    parser.add_argument('--output', type=str, default='multimodal_xai_evaluation.png', help='Path to save the combined output image.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    config = load_config(args.config)
    unimodal_config = load_config(config['models']['unimodal_config_path'])
    
    # --- Load the Multimodal Model and explicitly move it to the correct device ---
    model = MultimodalFusionModel(
        unimodal_config=unimodal_config,
        us_model_path=config['models']['us_model_path'],
        mammo_model_path=config['models']['mammo_model_path']
    ).to(device) # Move model to device
    
    # Load the state dictionary onto the same device
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # --- Preprocess Images ---
    mammo_tensor, mammo_original_img = preprocess_image(args.mammo_image)
    us_tensor, us_original_img = preprocess_image(args.us_image)
    if mammo_tensor is None or us_tensor is None: return
    
    # --- DEFINITIVE FIX: Move each input tensor to the same device as the model ---
    mammo_tensor = mammo_tensor.to(device)
    us_tensor = us_tensor.to(device)

    # --- Generate Explanations ---
    logging.info("Generating XAI for both modalities...")
    mammo_grad_cam = generate_multimodal_grad_cam(model, 'mammo', mammo_tensor, us_tensor, mammo_original_img)
    us_grad_cam = generate_multimodal_grad_cam(model, 'us', mammo_tensor, us_tensor, us_original_img)
    
    # --- Create Combined Visualization ---
    logging.info(f"Stitching visualizations and saving to {args.output}...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(mammo_grad_cam)
    axes[0].set_title('Mammogram Explanation (Grad-CAM)')
    axes[0].axis('off')
    
    axes[1].imshow(us_grad_cam)
    axes[1].set_title('Ultrasound Explanation (Grad-CAM)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output)
    logging.info("Done.")

if __name__ == '__main__':
    main()