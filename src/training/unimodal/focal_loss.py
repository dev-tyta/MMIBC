# training/unimodal/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    A robust implementation of Focal Loss, designed to address extreme
    class imbalance by focusing training on hard-to-classify examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): The weighting factor for the minority class. Helps to
                           balance the importance of positive/negative examples.
            gamma (float): The focusing parameter. A higher gamma value makes the
                           model focus more on hard, misclassified examples.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean': the sum of the
                             output will be divided by the number of elements
                             in the output.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): The model's raw output (logits).
                                   Shape: (N, C) where C = number of classes.
            targets (torch.Tensor): The ground truth labels.
                                    Shape: (N).
        """
        # Calculate Cross-Entropy loss but keep it un-reduced
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probabilities of the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate the Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
