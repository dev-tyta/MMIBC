# training/unimodal/utils.py

import numpy as np
import torch
import logging

class EarlyStopping:
    """
    A utility class to handle early stopping of the training process.
    It monitors a validation metric and stops training if it doesn't improve
    after a given 'patience' number of epochs.
    """
    def __init__(self, patience=5, min_delta=0, mode='min', save_path='saved_models/best_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will
                        stop when the quantity monitored has stopped increasing.
                        Default: 'min'
            save_path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.counter = 0
        self.best_score = np.Inf if self.mode == 'min' else -np.Inf
        self.early_stop = False

    def __call__(self, val_metric, model):
        """
        Call method to update the early stopping state.
        
        Args:
            val_metric (float): The validation metric from the current epoch.
            model (torch.nn.Module): The model being trained.
        """
        score = -val_metric if self.mode == 'max' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score > self.best_score + self.min_delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logging.warning("--- Early stopping triggered! ---")
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """Saves model when validation metric improves."""
        logging.info(f'Validation metric improved ({self.best_score:.6f} --> {val_metric:.6f}). Saving model to {self.save_path}...')
        torch.save(model.state_dict(), self.save_path)

