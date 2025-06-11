# training/unimodal/ultrasound_train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# --- NEW: Import Learning Rate Scheduler ---
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Import from our project files (Corrected duplicate import) ---
from unimodal_model import DinoV2Classifier 
from unimodal_dataset import UltrasoundDataset
from utils import EarlyStopping

def setup_logging(config):
    """Sets up logging to file and console."""
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("logs", f"{config['run_name']}_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "training_run.log")
    # Clear previous handlers to avoid duplicate logs in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    logging.info(f"Logging initialized. Log file: {log_filename}")

def load_config(config_path):
    """Loads and logs the training configuration."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Configuration loaded: {config}")
    return config

def get_data_loaders(config):
    """Creates training and validation data loaders."""
    dataset_handler = UltrasoundDataset(data_path=config['ultrasound_path'], image_size=config['training']['image_size'])
    train_dataset = dataset_handler.get_train_dataset()
    val_dataset = dataset_handler.get_validation_dataset()
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], pin_memory=True)
    logging.info(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    return train_loader, val_loader, train_dataset.classes

def plot_confusion_matrix(cm, class_names, save_path):
    """Plots and saves the confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names); ax.yaxis.set_ticklabels(class_names)
    plt.tight_layout(); plt.savefig(save_path)
    logging.info(f"Confusion matrix saved to {save_path}")
    return fig

def train_model(config):
    """Main function to orchestrate the model training process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_log_dir = os.path.join(config['output']['tensorboard_log_dir'], f"{config['run_name']}_{current_time}")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logging.info(f"TensorBoard logs will be saved to: {tb_log_dir}")

    train_loader, val_loader, class_names = get_data_loaders(config)
    
    # --- NEW: Calculate Class Weights for Weighted Loss ---
    # This automatically handles class imbalance by penalizing errors on the minority class more heavily.
    class_labels = np.array(train_loader.dataset.targets)
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    logging.info(f"Calculated class weights for loss function: {class_weights}")
    
    model = DinoV2Classifier(n_classes=config['model']['n_classes'], model_name=config['model']['name'], freeze_backbone=config['model']['freeze_backbone']).to(device)
    
    # --- Use the calculated class weights in the loss function ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])
    
    # --- NEW: Initialize Learning Rate Scheduler ---
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    model_save_path = os.path.join(config['output']['model_save_path'], "best_ultrasound_model.pth")
    os.makedirs(config['output']['model_save_path'], exist_ok=True)
    early_stopper = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        mode=config['training']['early_stopping']['mode'],
        save_path=model_save_path
    )
    
    logging.info("\n" + "="*20 + " Starting Training " + "="*20)
    for epoch in range(config['training']['epochs']):
        logging.info(f"--- Starting Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        # Training Phase
        model.train()
        # ... (training loop is the same)
        running_train_loss, train_labels, train_preds = 0.0, [], []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_labels.extend(labels.cpu().numpy()); train_preds.extend(preds.cpu().numpy())
            train_pbar.set_postfix(loss=loss.item())
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = accuracy_score(train_labels, train_preds)
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        
        # Validation Phase
        model.eval()
        # ... (validation loop is the same)
        running_val_loss, val_labels, val_preds, val_probs = 0.0, [], [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs); loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1); probs = torch.softmax(outputs, dim=1)[:, 1]
                val_labels.extend(labels.cpu().numpy()); val_preds.extend(preds.cpu().numpy()); val_probs.extend(probs.cpu().numpy())
                val_pbar.set_postfix(loss=loss.item())
        epoch_val_loss = running_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.0
        
        writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('F1-score/validation', val_f1, epoch)
        writer.add_scalar('AUC/validation', val_auc, epoch)
        
        logging.info(f"Epoch {epoch+1} Summary | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # --- NEW: Step the LR Scheduler and Early Stopper ---
        scheduler.step(epoch_val_loss)
        metric_to_monitor = epoch_val_loss if config['training']['early_stopping']['monitor'] == 'val_loss' else val_accuracy
        early_stopper(metric_to_monitor, model)
        if early_stopper.early_stop:
            break
    
    # --- NEW: Final Evaluation on the BEST saved model ---
    logging.info("Training complete. Performing final evaluation using the best model.")
    model.load_state_dict(torch.load(model_save_path)) # Load best weights
    
    final_val_labels, final_val_preds = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_val_labels.extend(labels.cpu().numpy())
            final_val_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(final_val_labels, final_val_preds)
    cm_save_path = os.path.join(os.path.dirname(tb_log_dir), "final_confusion_matrix.png")
    cm_fig = plot_confusion_matrix(cm, class_names, cm_save_path)
    writer.add_figure("Final Confusion Matrix", cm_fig, global_step=config['training']['epochs'])

    writer.close()
    logging.info("\n" + "="*20 + " Process Finished " + "="*20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Unimodal Ultrasound Classifier with DINOv2.")
    parser.add_argument('--config', type=str, default='/teamspace/studios/this_studio/MMIBC/src/training/unimodal/config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    setup_logging({'run_name': 'pre-config'}) 
    config = load_config(args.config)
    setup_logging(config)
    
    train_model(config)

