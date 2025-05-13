import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def visualize_samples(images_dir, masks_dir, num_samples=3):
    """Plot image-mask pairs to verify alignment."""
    image_files = [f for f in os.listdir(images_dir) if not f.endswith('_mask.png')]
    selected = random.sample(image_files, num_samples)
    
    plt.figure(figsize=(15, 5*num_samples))
    for i, img_file in enumerate(selected):
        img = Image.open(os.path.join(images_dir, img_file))
        mask = Image.open(os.path.join(masks_dir, img_file.replace('.png', '_mask.png')))
        
        # Plot image
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(img)
        plt.title(f"Image: {img_file}")
        plt.axis('off')
        
        # Plot mask
        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask: {img_file.replace('.png', '_mask.png')}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_samples("data/resized/images", "data/resized/masks")