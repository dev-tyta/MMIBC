# training/unimodal/unimodal_evaluation.py

import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Import from our project files ---
from unimodal_model import DinoV2Classifier 
from unimodal_dataset import ImageFolderDataset

def setup_logging(output_dir):
    """Sets up logging for the evaluation script."""
    log_filename = os.path.join(output_dir, "test_evaluation_log.txt")
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    logging.info(f"Evaluation log initialized. Log file: {log_filename}")

def load_config(config_path):
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file: config = yaml.safe_load(file)
    logging.info(f"Configuration loaded: {config}")
    return config

def evaluate_model(config, model_path, test_data_path, output_dir):
    """Loads a model and evaluates its performance on the test set."""
    setup_logging(output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    logging.info("Loading trained model...")
    model = DinoV2Classifier(
        n_classes=2,
        model_name="dinov2_vitl14",
        dropout_rate=0.3,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("Model loaded successfully.")

    # --- Load Test Data ---
    dataset_handler = ImageFolderDataset(
        data_path=test_data_path,
        image_size= 224
    )
    test_dataset = dataset_handler.get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    logging.info(f"Loaded {len(test_dataset)} test images.")

    # --- Run Inference ---
    logging.info("Running inference on the test set...")
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get probabilities for AUC score (for the positive class)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            
            # Get predicted classes
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- Calculate and Report Metrics ---
    logging.info("\n" + "="*30 + " PERFORMANCE REPORT " + "="*30)
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"Overall Test Accuracy: {accuracy:.4f}")

    # ROC AUC Score
    roc_auc = roc_auc_score(all_labels, all_probs)
    logging.info(f"Test ROC AUC Score: {roc_auc:.4f}")
    
    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logging.info("Classification Report:\n" + report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Test Set Confusion Matrix')
    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    logging.info("="*30 + " EVALUATION COMPLETE " + "="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained unimodal classifier on a test set.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('--config', type=str, default= "/teamspace/studios/this_studio/MMIBC/src/training/unimodal/config.yaml", help='Path to the config file used for training the model.')
    parser.add_argument('--test_data_dir', type=str, default="/teamspace/studios/this_studio/mmibc/paired/ultrasound", help='Path to the root of the test dataset (e.g., BUS_UC_split).')
    parser.add_argument('--output_dir', type=str, default='/teamspace/studios/this_studio/runs/evaluation_results', help='Directory to save evaluation results (logs, plots).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_model(args.config, args.model_path, args.test_data_dir, args.output_dir)
