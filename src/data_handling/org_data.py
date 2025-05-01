import os
import shutil

def organize_images_and_masks(source_dir, images_dir, masks_dir):
    """Move images and masks to separate directories."""
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        if os.path.isfile(src_path):
            if filename.endswith('_mask.png'):
                shutil.move(src_path, os.path.join(masks_dir, filename))
            else:
                shutil.move(src_path, os.path.join(images_dir, filename))


if __name__ == "__main__":
    # Example usage
    source_dir = "./data/Dataset_BUSI_with_GT"
    images_dir = './data/ultrasound/images'
    masks_dir = './data/ultrasound/masks'
    
    organize_images_and_masks(source_dir, images_dir, masks_dir)