# prompt: code to move the content of ./data/Dataset_BUSI_with_GT/ into ./data/ultrasound/masks and ./data/ultrasound/images, but checking each directory within the main directory for the appropriate mask images to move into ./data/ultrasound/masks and normal images to move into ./data/ultrasound/images. Mask images contain "_mask"; make sure it's just moving the necessary files from the dataset Busi into ultrasound. make sure the sub-directory under the /data/Dataset_BUSI_with_GT/  is used as same subdirectory under the ultrasound/images and ultrasound/masks

import os
import shutil

def organize_images_and_masks(source_dir, images_dir, masks_dir):
    """Recursively move .png images and masks to separate directories."""
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.png'):
                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, source_dir)

                # Create corresponding subdirectories in images_dir and masks_dir
                image_subdir = os.path.join(images_dir, rel_path)
                mask_subdir = os.path.join(masks_dir, rel_path)
                os.makedirs(image_subdir, exist_ok=True)
                os.makedirs(mask_subdir, exist_ok=True)

                if "_mask" in filename:
                    dst_path = os.path.join(mask_subdir, filename)
                else:
                    dst_path = os.path.join(image_subdir, filename)

                shutil.move(src_path, dst_path)

if __name__ == "__main__":
    source_dir = "./data/Dataset_BUSI_with_GT"
    images_dir = './data/ultrasound/images'
    masks_dir = './data/ultrasound/masks'
    organize_images_and_masks(source_dir, images_dir, masks_dir)
