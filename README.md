# Explainable Multimodal Imaging for Breast Cancer Diagnosis (MMIBC)

This repository contains the official implementation for the research project **"Explainable Multimodal Imaging for Breast Cancer Diagnosis Using Mammography and Ultrasonography Datasets"**. The project develops and evaluates an explainable AI framework that fuses mammography and ultrasound images to improve breast cancer diagnosis, with a focus on application in resource-limited settings.

*Fig: Grad-CAM visualizations showing the model focusing on clinically relevant regions in both mammography and ultrasound images.*

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Datasets](#-datasets)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Citation](#-citation)

## 📝 Project Overview

High breast cancer mortality rates in resource-limited contexts are often driven by delayed diagnosis. While Artificial Intelligence (AI) has shown immense promise in medical imaging, its "black box" nature can hinder clinical adoption. This project addresses this gap by developing a multimodal AI framework that synergistically fuses data from mammography and ultrasound—the two most accessible imaging modalities. The framework is built to be transparent and trustworthy, leveraging Explainable AI (XAI) to provide insights into its diagnostic decisions.

## ✨ Key Features

- **Multimodal Fusion:** Combines mammography and ultrasound images to leverage the complementary strengths of each modality.
- **State-of-the-Art Backbone:** Utilizes the powerful `DINOv2` Vision Transformer (ViT) for robust feature extraction.
- **Explainable AI (XAI):** Integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to produce visual heatmaps, showing which image regions influenced the model's prediction.
- **Programmatic Data Pairing:** Implements a novel strategy to create a paired multimodal dataset from separate unimodal sources, addressing the common challenge of data scarcity.

## 🏗️ Model Architecture

The model consists of two parallel streams, each using a pre-trained `dinov2-base` ViT backbone to extract features from a mammogram and an ultrasound image, respectively. The resulting feature vectors are then concatenated and passed through a fusion head (a series of dense layers) to produce a final binary classification (benign or malignant).

> *[Placeholder for a detailed architectural diagram image]*

## 💾 Datasets

This project utilizes two primary public datasets:
1.  **VinDr-Mammo:** A large-scale benchmark dataset for mammography.
2.  **BUSI (Breast Ultrasound Images):** A well-curated dataset for breast ultrasound.

Due to the lack of natively paired public datasets, a synthetic patient cohort was created by programmatically pairing images with the same diagnostic label from each dataset. The scripts for data handling and pairing can be found in `src/data_handling/` and `src/training/multimodal/pairing_data.py`.

## 📊 Results

The final multimodal framework achieved the following performance on a held-out test set:
- **Overall Accuracy:** 84%
- **Macro Average F1-Score:** 76%
- **Weighted Average F1-Score:** 82%

The model showed high performance on the benign class but had a lower recall (52%) for the malignant class, a result attributed to significant class imbalance in the source datasets.

## ⚙️ Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/MMIBC.git](https://github.com/your-username/MMIBC.git)
    cd MMIBC
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

The project is structured into data handling and training modules.

### 1. Data Preparation
- Place the VinDr-Mammo and BUSI datasets in a `data/` directory (or update the paths in the config files).
- Run the necessary scripts in `src/data_handling/` to process and organize the data.

### 2. Training
- The training process is managed by scripts in the `src/training/` directory.
- Hyperparameters for unimodal and multimodal training can be configured in the respective `config.yaml` files.
- To train the final multimodal model, run:
  ```bash
  python src/training/multimodal/multimodal_train.py
  ```

### 3\. Evaluation

  - To evaluate a trained model on the test set, use the evaluation script:
    ```bash
    python src/training/multimodal/multimodal_evaluation.py --model_path /path/to/your/trained_model.pth
    ```

### 4\. Generating Explanations (XAI)

  - To generate Grad-CAM visualizations for a given image pair, use the XAI script:
    ```bash
    python src/training/multimodal/multimodal_xai.py --model_path /path/to/your/trained_model.pth --mammo_image /path/to/mammo.png --ultrasound_image /path/to/us.png
    ```

## 📜 Citation

If you use this work, please cite the following paper:

```bibtex
@inproceedings{adekoya2025mmibc,
  title={Explainable Multimodal Imaging for Breast Cancer Diagnosis Using Mammography and Ultrasonography Datasets},
  author={Adekoya, Testimony Oluwanifemi},
  booktitle={Conference or Journal Name},
  year={2025}
}
