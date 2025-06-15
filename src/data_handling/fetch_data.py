import os
import requests
import zipfile
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import shutil
import sys

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Logger initialized")


def download_file(url, filename):
    logger.info(f"Downloading {filename} from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def unzip_file(zip_path, extract_to):
    logger.info(f"Unzipping {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Unzipped {zip_path} to {extract_to}")

def fetch_bus_images_dataset(output_path):
    # URL of the BUSI dataset
    url = 'https://scholar.cu.edu.eg/Dataset_BUSI.zip'
    filename = os.path.join(output_path, 'BUSI.zip')
    download_file(url, filename)
    unzip_file(filename, output_path)
    os.remove(filename)

def fetch_bus_dataset(output_path):
    url = 'https://data.mendeley.com/public-files/datasets/wmy84gzngw/files/b63daee9-78de-4122-8475-9b3aa22ffd64/file_downloaded'
    filename = os.path.join(output_path, 'BUS.zip')
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    unzip_file(filename, output_path)
    os.remove(filename)




def fetch_vindr_metadata(output_path):
    import kagglehub
    os.environ["KAGGLE_CONFIG"] = os.path.join(os.path.dirname(__file__), "./kaggle/kaggle.json")
    # Download latest version
    path = kagglehub.dataset_download("truthisneverlinear/vindr-mammo-annotations")
    logger.info(f"Downloaded dataset to {path}")
    move_dataset(path, output_path)

    
def move_dataset(path, output_path):
    # Move the dataset to the output path
    os.makedirs(output_path, exist_ok=True)
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(output_path, item)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
    os.rmdir(path)
    logger.info(f"Moved dataset to {output_path}")

def fetch_vindr_dataset(output_path):
    import kagglehub
    os.environ["KAGGLE_CONFIG"] = os.path.join(os.path.dirname(__file__), "./kaggle/kaggle.json")
    # Download latest version
    path = kagglehub.dataset_download("shantanughosh/vindr-mammogram-dataset-dicom-to-png")
    logger.info(f"Downloaded dataset to {path}")
    move_dataset(path, output_path)

def fetch_kau_bcmd_dataset(output_path):
    import kagglehub
    path = kagglehub.dataset_download("orvile/kau-bcmd-mamography-dataset")
    logger.info(f"Downloaded dataset to {path}")
    move_dataset(path, output_path)

def fetch_bus_uc_dataset(output_path):
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("orvile/bus-uc-breast-ultrasound")
    logger.info(f"Downloaded dataset to {path}")
    move_dataset(path, output_path)
    



def main():
    output_path = 'data'
    path = "/home/codespace/.cache/kagglehub/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png/versions/1"
    os.makedirs(output_path, exist_ok=True)

    # fetch_bus_dataset(output_path)
    # fetch_bus_images_dataset(output_path)
    # fetch_vindr_dataset(output_path)
    # fetch_vindr_metadata(output_path="data")
    fetch_kau_bcmd_dataset(output_path)
    fetch_bus_uc_dataset(output_path)

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script finished")
    logger.info("All datasets fetched successfully")
    
