import os
from PIL import Image
import numpy as np
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create a file handler
handler = logging.FileHandler('ultrasound_handling.log')
handler.setLevel(logging.INFO)


def convert_tmp_to_img(input_dir, output_dir):
    """
    Convert .tmp files to .png images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Converting .bmp files in {input_dir} to .png format in {output_dir}")
    for class_name in ['benign', 'malignant']:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Directory not found: {class_dir}")
            continue
            
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        logger.info(f"Processing class: {class_name}")
        for tmp_file in os.listdir(class_dir):
            logger.info(f"Processing file: {tmp_file}")
            if tmp_file.endswith('.bmp'):
                try:
                    logger.info(f"Converting {tmp_file} to PNG format")
                    # Load .tmp file (adjust based on actual format)
                    img_path = os.path.join(class_dir, tmp_file)
                    img = Image.open(img_path)  # Validate .tmp format
                    
                    # Save as PNG
                    output_path = os.path.join(output_class_dir, tmp_file.replace('.bmp', '.png'))
                    img.save(output_path)
                    print(f"Converted: {tmp_file} → {output_path}")
                except Exception as e:
                    logger.error(f"Error processing {tmp_file}: {str(e)}")

def check_img_shape(input_dir):
    """
    Check the shape of images in the directory.
    """
    for class_name in ['benign', 'malignant']:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Directory not found: {class_dir}")
            continue
            
        logger.info(f"Checking images in class: {class_name}")
        for img_file in os.listdir(class_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    logger.info(f"Image {img_path} Image shape: {img_array.shape}")
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {str(e)}")




if __name__ == "__main__":
    # convert_tmp_to_img("data/originals", "data/ultrasound-2")
    check_img_shape("data/Dataset_BUSI_with_GT")
