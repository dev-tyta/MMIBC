# training/multimodal/multimodal_train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datetime import datetime

from multimodal_dataset import MultimodalDataset
from multimodal_architecture import MultimodalFusionModel
from src.training.unimodal.utils import EarlyStopping # Re-use the same EarlyStopping class

def setup_logging(config):
    # (Same as unimodal scripts)
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("logs", f"{config['run_name']}_multimodal_{current_time}")
    os.makedirs(log_dir, exist_ok=True); log_filename = os.path.join(log_dir, "training_run.log")
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def load_config(config_path):
    with open(config_path, 'r') as file: return yaml.safe_load(file)

def train_model(config):
    setup_logging(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # --- Load Data ---
    train_dataset = MultimodalDataset(csv_file=config['data']['csv_path'], split='train')
    val_dataset = MultimodalDataset(csv_file=config['data']['csv_path'], split='validation')
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    
    # --- Load Unimodal Config for Model Init ---
    with open(config['models']['unimodal_config_path'], 'r') as f:
        unimodal_config = yaml.safe_load(f)

    # --- Initialize Model ---
    model = MultimodalFusionModel(
        unimodal_config=unimodal_config,
        us_model_path=config['models']['us_model_path'],
        mammo_model_path=config['models']['mammo_model_path']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # We only train the fusion classifier part
    optimizer = optim.AdamW(model.fusion_classifier.parameters(), lr=config['training']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # --- Setup Logging and Saving ---
    writer = SummaryWriter(log_dir=os.path.join(config['output']['tensorboard_log_dir'], f"{config['run_name']}_multimodal"))
    model_save_path = os.path.join(config['output']['model_save_path'], "best_multimodal_model.pth")
    os.makedirs(config['output']['model_save_path'], exist_ok=True)
    early_stopper = EarlyStopping(patience=config['training']['early_stopping']['patience'], mode='min', save_path=model_save_path)

    logging.info("Starting multimodal model training...")
    for epoch in range(config['training']['epochs']):
        model.train()
        # (Training loop is standard)
        for mammo_batch, us_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mammo_batch, us_batch, labels = mammo_batch.to(device), us_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mammo_batch, us_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        # (Validation loop is standard)
        all_labels, all_preds = [], []
        val_loss = 0
        with torch.no_grad():
            for mammo_batch, us_batch, labels in val_loader:
                mammo_batch, us_batch, labels = mammo_batch.to(device), us_batch.to(device), labels.to(device)
                outputs = model(mammo_batch, us_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy()); all_preds.extend(preds.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        logging.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        writer.add_scalar('Loss/validation', avg_val_loss, epoch); writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            logging.info("Early stopping triggered.")
            break
            
    writer.close()
    logging.info("Multimodal training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the multimodal fusion model.")
    parser.add_argument('--config', type=str, default='/teamspace/studios/this_studio/MMIBC/src/training/multimodal/config.yaml', help='Path to the multimodal config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    train_model(config)
