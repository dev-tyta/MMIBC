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

def setup_logging(config, stage):
    """Sets up logging to file and console for a specific training stage."""
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("logs", f"{config['run_name']}_{stage}_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "training_run.log")
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    logging.info(f"Logging initialized for {stage}. Log file: {log_filename}")

def load_config(config_path):
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as file: config = yaml.safe_load(file)
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

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plots and saves the confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); ax.set_title(title)
    ax.xaxis.set_ticklabels(class_names); ax.yaxis.set_ticklabels(class_names)
    plt.tight_layout(); plt.savefig(save_path)
    logging.info(f"Confusion matrix saved to {save_path}")
    return fig

def execute_stage(stage_name, model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopper, writer, start_epoch, end_epoch, device):
    """Executes a single stage of training."""
    logging.info(f"\n" + "="*20 + f" Starting Training Stage: {stage_name} " + "="*20)
    for epoch in range(start_epoch, end_epoch):
        logging.info(f"--- Starting Epoch {epoch+1}/{end_epoch} ({stage_name}) ---")
        model.train()
        train_loss_sum = 0.0
        all_train_labels = []
        all_train_preds = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            train_pbar.set_postfix(loss=loss.item())

            _, predicted = torch.max(outputs.data, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())
            train_pbar.set_postfix(loss=loss.item())
        
        epoch_train_loss = train_loss_sum / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(all_train_labels, all_train_preds)
        
        writer.add_scalar(f'Loss/train_{stage_name}', epoch_train_loss, epoch)
        writer.add_scalar(f'Accuracy/train_{stage_name}', epoch_train_acc, epoch)
        logging.info(f"Epoch {epoch+1} [{stage_name}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        
        model.eval()
        all_val_labels = []
        all_val_preds = []
        running_val_loss = 0.0
        # val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val - {stage_name}]")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs); loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                # val_pbar.set_postfix(loss=loss.item())
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(all_val_labels, all_val_preds)
        
        writer.add_scalar(f'Loss/validation_{stage_name}', epoch_val_loss, epoch)
        writer.add_scalar(f'Accuracy/validation_{stage_name}', epoch_val_acc, epoch)
        logging.info(f"Epoch {epoch+1} [{stage_name}] - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        
        scheduler.step(epoch_val_loss)
        early_stopper(epoch_val_loss, model)
        if early_stopper.early_stop:
            logging.warning(f"Early stopping triggered during {stage_name}.")
            break

def train_model(config):
    """Main function to orchestrate the progressive fine-tuning process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = get_data_loaders(config)
    
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_loader.dataset.targets), y=train_loader.dataset.targets), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    

    model = DinoV2Classifier(n_classes=config['model']['n_classes'], model_name=config['model']['name'], dropout_rate=config['training']['dropout_rate']).to(device)

    # --- STAGE 1: HEAD TRAINING ---
    setup_logging(config, stage="Stage1_Head")
    writer_s1 = SummaryWriter(log_dir=os.path.join(config['output']['tensorboard_log_dir'], f"{config['run_name']}_Stage1"))
    model.freeze_backbone()
    
    optimizer_stage1 = optim.AdamW(
        model.head.parameters(), 
        lr=config['training']['progressive_finetuning']['lr_stage1'],
        weight_decay=config['training']['weight_decay'] # Add weight decay here for consistency
    )
    
    scheduler_stage1 = ReduceLROnPlateau(optimizer_stage1, mode='min', factor=0.1, patience=5)
    model_save_path_s1 = os.path.join(config['output']['model_save_path'], "best_model_stage1.pth")
    os.makedirs(config['output']['model_save_path'], exist_ok=True)
    early_stopper_s1 = EarlyStopping(patience=config['training']['early_stopping']['patience'], mode='min', save_path=model_save_path_s1)
    
    execute_stage("Stage1_Head", model, train_loader, val_loader, criterion, optimizer_stage1, scheduler_stage1, early_stopper_s1, writer_s1, 0, config['training']['progressive_finetuning']['epochs_stage1'], device)
    writer_s1.close()

    # --- STAGE 2: FULL FINE-TUNING ---
    setup_logging(config, stage="Stage2_Finetune")
    writer_s2 = SummaryWriter(log_dir=os.path.join(config['output']['tensorboard_log_dir'], f"{config['run_name']}_Stage2"))
    
    logging.info("Loading best model from Stage 1 to begin Stage 2.")
    model.load_state_dict(torch.load(model_save_path_s1))
    model.unfreeze_last_n_layers(config['training']['progressive_finetuning']['unfreeze_layers_stage2'])

     # --- CORRECTED OPTIMIZER DEFINITION ---
    # The weight_decay parameter must be defined within each dictionary
    # when using multiple parameter groups.
    optimizer_stage2 = optim.AdamW([
        {
            'params': model.backbone.parameters(), 
            'lr': config['training']['progressive_finetuning']['lr_backbone_stage2'],
            'weight_decay': config['training']['weight_decay']
        },
        {
            'params': model.head.parameters(), 
            'lr': config['training']['progressive_finetuning']['lr_head_stage2'],
            'weight_decay': config['training']['weight_decay']
        }
    ])

    scheduler_stage2 = ReduceLROnPlateau(optimizer_stage2, mode='min', factor=0.1, patience=5)
    model_save_path_s2 = os.path.join(config['output']['model_save_path'], "best_model_final.pth")
    early_stopper_s2 = EarlyStopping(patience=config['training']['early_stopping']['patience'], mode='min', save_path=model_save_path_s2)
    
    start_epoch_s2 = config['training']['progressive_finetuning']['epochs_stage1']
    end_epoch_s2 = config['training']['epochs']
    execute_stage("Stage2_Finetune", model, train_loader, val_loader, criterion, optimizer_stage2, scheduler_stage2, early_stopper_s2, writer_s2, start_epoch_s2, end_epoch_s2, device)

    # --- FINAL EVALUATION ---
    logging.info("Progressive fine-tuning finished. Performing final evaluation using the best model from Stage 2.")
    model.load_state_dict(torch.load(model_save_path_s2))
    
    final_val_labels, final_val_preds = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); _, preds = torch.max(outputs, 1)
            final_val_labels.extend(labels.cpu().numpy()); final_val_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(final_val_labels, final_val_preds)
    cm_save_path = os.path.join(os.path.dirname(writer_s2.log_dir), "ultrasound_confusion_matrix_2.png")
    cm_fig = plot_confusion_matrix(cm, class_names, cm_save_path, title="Ultrasound Confusion Matrix (After Fine-tuning)")
    writer_s2.add_figure("Ultrasound Confusion Matrix", cm_fig, global_step=end_epoch_s2)
    
    final_acc = accuracy_score(final_val_labels, final_val_preds)
    final_f1 = f1_score(final_val_labels, final_val_preds, average='macro')
    logging.info(f"Final evaluation on best model: Accuracy={final_acc:.4f}, Macro-F1={final_f1:.4f}")

    writer_s2.close()
    logging.info("\n" + "="*20 + " Process Finished " + "="*20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Unimodal Ultrasound Classifier with Progressive Fine-Tuning.")
    parser.add_argument('--config', type=str, default='/teamspace/studios/this_studio/MMIBC/src/training/unimodal/config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    
    config = load_config(args.config)
    train_model(config)
