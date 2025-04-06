import os
import requests
import zipfile
from tqdm import tqdm
import logging
from dotenv import load_dotenv

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


def main():
    output_path = 'data'
    os.makedirs(output_path, exist_ok=True)

    fetch_bus_dataset(output_path)
    fetch_bus_images_dataset(output_path)


if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script finished")
    logger.info("All datasets fetched successfully")
    