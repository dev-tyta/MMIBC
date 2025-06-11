import os
import shutil

def list_files_with_formats(directory):
    """Lists all files in a directory, showing their formats."""
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                name, ext = os.path.splitext(filename)
                print(f"File: {filename}, Format: {ext}")
            elif os.path.isdir(filepath):
                print(f"Directory: {filename}")
                list_files_with_formats(filepath)  # Recursive call for subdirectories
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

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

                shutil.copy2(src_path, dst_path)  # Changed from move to copy

if __name__ == "__main__":
    source_dir = os.path.expanduser("~/data/Dataset_BUSI_with_GT")
    images_dir = os.path.expanduser('~/data/ultrasound/images')
    masks_dir = os.path.expanduser('~/data/ultrasound/masks')

    shutil.copytree(os.path.expanduser('~/data/images_png'), os.path.expanduser('~/data/mammo/images'))
    shutil.copy(os.path.expanduser('~/data/breast-level_annotations.csv'), os.path.expanduser('~/data/mammo/breast-level_annotations.csv'))
    shutil.copy(os.path.expanduser('~/data/finding_annotations.csv'), os.path.expanduser('~/data/mammo/finding_annotations.csv'))
    shutil.copy(os.path.expanduser('~/data/metadata.csv'), os.path.expanduser('~/data/mammo/metadata.csv'))
    shutil.copy(os.path.expanduser('~/data/SHA256SUMS.txt'), os.path.expanduser('~/data/mammo/SHA256SUMS.txt'))

    list_files_with_formats(os.path.expanduser("~/data/mammo/images"))
    organize_images_and_masks(source_dir, images_dir, masks_dir)