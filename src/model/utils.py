import torch
import numpy as np
import logging
import os
import shutil
import time
from sklearn.metrics import accuracy_score, sensitivity_score, specificity_score, precision_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricLogger:
    """
    Utility class for logging metrics during training
    """
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        # Default behavior
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        # For multi-GPU training - Placeholder
        pass

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)

        start_time = time.time()
        end = time.time()
        for obj in iterable:
            data_time = time.time() - end
            yield obj
            batch_time = time.time() - end
            end = time.time()
            if i % print_freq == 0:
                eta_seconds = (len(iterable) - i) * (batch_time + data_time) / 2
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f"{header} [{i}/{len(iterable)}]\t"
                    f"eta: {eta_string}\t"
                    f"time: {batch_time:.4f}\t"
                    f"data: {data_time:.4f}\t"
                    f"{self}"
                )
            i += 1
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f}s / it)')


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a window
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.values = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.values.append(value)
        self.total += value
        self.count += 1
        if len(self.values) > self.window_size:
            self.total -= self.values.pop(0)

    @property
    def median(self):
        return np.median(self.values)

    @property
    def avg(self):
        return np.mean(self.values)

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return f"{self.global_avg:.4f} ({self.avg:.4f})"

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """
    Cosine scheduler with warmup for updating parameters like teacher momentum
    """
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)

    iters = np.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_checkpoint(state, is_best, filename='checkpoint.pth', save_dir='.'):
    """
    Saves model checkpoint.
    """
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(save_dir, 'model_best.pth')
        shutil.copyfile(filepath, best_filepath)
        logger.info(f"Best model saved to {best_filepath}")

def calculate_metrics(all_labels, all_preds, all_probs):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        all_labels (list): List of true labels.
        all_preds (list): List of predicted labels.
        all_probs (list): List of probabilities for the positive class.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    metrics = {}

    # Basic metrics using sklearn
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['sensitivity'] = sensitivity_score(all_labels, all_preds) # Recall
    metrics['specificity'] = specificity_score(all_labels, all_preds)
    metrics['precision'] = precision_score(all_labels, all_preds)
    metrics['f1'] = f1_score(all_labels, all_preds)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm
    tn, fp, fn, tp = cm.ravel()
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = tn, fp, fn, tp
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0 # Negative predictive value


    # ROC and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    metrics['fpr'], metrics['tpr'], metrics['roc_auc'] = fpr, tpr, roc_auc

    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)
    metrics['precision_curve'], metrics['recall_curve'], metrics['pr_auc'] = precision_curve, recall_curve, pr_auc

    return metrics

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """
    Plots and saves the ROC curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"ROC curve saved to {save_path}")

def plot_pr_curve(precision_curve, recall_curve, pr_auc, save_path):
    """
    Plots and saves the Precision-Recall curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Precision-Recall curve saved to {save_path}")

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    """
    Plots and saves the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")
