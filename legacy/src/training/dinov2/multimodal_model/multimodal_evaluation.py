# training/multimodal/multimodal_evaluation.py

import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.training.multimodal.multimodal_dataset import MultimodalDataset
from src.training.multimodal.multimodal_architecture import MultimodalFusionModel

def evaluate_model(config, model_path, output_dir):
    """Loads and evaluates the final multimodal model on the test set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Test Data ---
    test_dataset = MultimodalDataset(csv_file=config['data']['csv_path'], split='test')
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    
    with open(config['models']['unimodal_config_path'], 'r') as f:
        unimodal_config = yaml.safe_load(f)

    # --- Load Model ---
    model = MultimodalFusionModel(
        unimodal_config=unimodal_config,
        us_model_path=config['models']['us_model_path'],
        mammo_model_path=config['models']['mammo_model_path']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Run Inference ---
    all_labels, all_preds = [], []
    with torch.no_grad():
        for mammo_batch, us_batch, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            mammo_batch, us_batch = mammo_batch.to(device), us_batch.to(device)
            outputs = model(mammo_batch, us_batch)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(preds.cpu().numpy())

    # --- Report Metrics ---
    logging.info("\n" + "="*30 + " MULTIMODAL PERFORMANCE REPORT " + "="*30)
    class_names = ['benign', 'malignant']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logging.info("Classification Report:\n" + report)
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Multimodal Test Set Confusion Matrix')
    cm_path = os.path.join(output_dir, "multimodal_test_confusion_matrix.png")
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix saved to {cm_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the multimodal model.")
    parser.add_argument('--config', type=str, default='/teamspace/studios/this_studio/MMIBC/src/training/multimodal/config.yaml', help='Path to the multimodal config file.')
    parser.add_argument('--model_path', type=str, default="/teamspace/studios/this_studio/saved_models/best_multimodal_model.pth", help='Path to the trained multimodal model.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = yaml.safe_load(open(args.config, 'r'))
    evaluate_model(config, args.model_path, args.output_dir)
