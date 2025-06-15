import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# --- Import from our project files ---
from unimodal_model import DinoV2Classifier  # Assuming your model class is here
from unimodal_dataset import MammographyDataset, get_mammo_transforms # Assuming your dataset and transforms are here

def setup_logging(config, model_name_tag="best_mammo_model"):
    """Sets up logging to file and console for evaluation."""
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("logs", "evaluation", f"{config['run_name']}_mammo_eval_{model_name_tag}_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "evaluation_run.log")
    
    # Remove existing handlers to avoid duplicate logs if re-running in the same session
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized for mammography evaluation. Log file: {log_filename}")
    return log_dir

def load_config(config_path):
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded: {config}")
    return config

def get_test_loader(config):
    """Creates a test data loader for the mammography dataset."""
    image_size = config['training']['image_size'] # Use image size from training
    _, test_transforms = get_mammo_transforms(image_size) # Get test transforms

    test_dataset = MammographyDataset(
        csv_file=config['mammo_csv_path'], # Path to your CSV containing all splits
        root_dir=config['mammo_data_path'],
        split='test', # Crucial: specify the 'test' split
        transform=test_transforms
    )

    if len(test_dataset) == 0:
        logging.warning("Test dataset is empty. Please check your CSV and data paths for the 'test' split.")
        return None, []

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'], # Can use training batch_size or a specific eval batch_size
        shuffle=False, # No need to shuffle for evaluation
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Define class names (ensure this matches your training and dataset)
    class_names = ['benign', 'malignant'] 
    logging.info(f"Found {len(test_dataset)} mammography test images.")
    return test_loader, class_names

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Confusion matrix saved to {save_path}")
    plt.close(fig) # Close the figure to free memory
    return fig

def evaluate_model(config, model_path):
    """
    Evaluates the trained mammography classification model on the test set.
    """
    log_dir = setup_logging(config, model_name_tag=os.path.basename(model_path).replace(".pth",""))
    logging.info(f"Starting mammography model evaluation for model: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loader ---
    test_loader, class_names = get_test_loader(config)
    if test_loader is None:
        logging.error("Failed to create test loader. Exiting evaluation.")
        return

    # --- Model ---
    n_classes = config['model']['n_classes']
    model = DinoV2Classifier(n_classes=n_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Please check the path.")
        return
    except Exception as e:
        logging.error(f"Error loading model state_dict: {e}")
        return
        
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = [] # For AUC

    eval_progress = tqdm(test_loader, desc="Evaluating on Test Set")
    with torch.no_grad():
        for inputs, labels in eval_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    if not all_labels:
        logging.warning("No data processed from the test loader. Cannot compute metrics.")
        return

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # --- Calculate Metrics ---
    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    
    logging.info(f"\n--- Test Set Evaluation Results ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Macro F1-Score: {f1:.4f}")

    logging.info("\nClassification Report:")
    try:
        report = classification_report(all_labels_np, all_preds_np, target_names=class_names, zero_division=0)
        logging.info(f"\n{report}")
    except ValueError as e:
        logging.error(f"Error generating classification report: {e}")
        logging.info(f"Unique true labels: {np.unique(all_labels_np)}, Unique predicted labels: {np.unique(all_preds_np)}")


    # ROC AUC Score
    roc_auc = None
    if n_classes == 2: # Binary classification
        if len(np.unique(all_labels_np)) == 2: # Check if both classes are present in true labels
            try:
                # Probabilities for the positive class (assuming class 1 is positive)
                roc_auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
                logging.info(f"ROC AUC Score: {roc_auc:.4f}")
            except ValueError as e:
                logging.warning(f"Could not calculate ROC AUC: {e}. This might happen if only one class is present in true labels.")
        else:
            logging.warning("ROC AUC not calculated for binary case: not all classes present in true labels.")
    elif n_classes > 2: # Multiclass classification
        try:
            if len(np.unique(all_labels_np)) > 1: # Check if more than one class in labels
                roc_auc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro')
                logging.info(f"ROC AUC Score (OvR, macro): {roc_auc:.4f}")
            else:
                logging.warning("ROC AUC (OvR) not calculated: only one class present in true labels.")
        except ValueError as e:
            logging.warning(f"Could not calculate ROC AUC (OvR): {e}.")


    # Confusion Matrix
    cm = confusion_matrix(all_labels_np, all_preds_np)
    cm_save_path = os.path.join(log_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_save_path, title="Test Set Confusion Matrix")

    logging.info("--- Evaluation Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a Unimodal Mammography Classifier on the test set.")
    parser.add_argument(
        '--config', 
        type=str, 
        default='/teamspace/studios/this_studio/MMIBC/src/training/unimodal/config.yaml', 
        help='Path to the YAML configuration file used during training.'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="/teamspace/studios/this_studio/saved_models/best_mammo_model.pth", # Default to None, will try to use path from config
        help='Path to the trained model (.pth file). Overrides model_save_path in config.'
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_path_to_evaluate = args.model_path
    if model_path_to_evaluate is None:
        # Try to get the model path from the config (where it was saved during training)
        if 'output' in config and 'model_save_path' in config['output']:
            model_path_to_evaluate = os.path.join(config['output']['model_save_path'], "best_mammo_model.pth")
            logging.info(f"Using model path from config: {model_path_to_evaluate}")
        else:
            logging.error("Model path not provided via --model_path argument and not found in config. Please specify the model path.")
            exit(1)
    
    if not os.path.exists(model_path_to_evaluate):
        logging.error(f"Model file does not exist at the specified path: {model_path_to_evaluate}")
        exit(1)

    evaluate_model(config, model_path_to_evaluate)