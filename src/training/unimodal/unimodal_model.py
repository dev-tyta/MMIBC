# training/unimodal/unimodal_model.py

import torch
import torch.nn as nn
import logging

class DinoV2Classifier(nn.Module):
    """
    A versatile classifier model using a DINOv2 backbone.
    This version includes an improved MLP head for better classification.
    """
    def __init__(self, n_classes, model_name='dinov2_vits14', freeze_backbone=True):
        """
        Args:
            n_classes (int): Number of output classes for the classifier.
            model_name (str): The specific DINOv2 model to use.
            freeze_backbone (bool): If True, freezes the DINOv2 backbone weights.
        """
        super().__init__()
        self.n_classes = n_classes
        logging.info(f"Initializing model with {n_classes} classes and backbone {model_name}.")
        
        # --- Load DINOv2 Backbone ---
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, force_reload=False)
        n_features = self.backbone.embed_dim 
        
        # --- NEW: Improved MLP Classification Head ---
        # A more robust head with a hidden layer, activation, and dropout.
        self.head = nn.Sequential(
            nn.Linear(n_features, n_features // 2), # Hidden layer
            nn.ReLU(),                              # Activation function
            nn.Dropout(0.5),                        # Dropout for regularization
            nn.Linear(n_features // 2, n_classes)   # Output layer
        )
        logging.info(f"Using MLP head with a hidden layer of size {n_features // 2}.")
        
        # --- Freeze Backbone ---
        if freeze_backbone:
            logging.info("Freezing backbone weights. Only the classification head will be trained.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Ensure the head is trainable
            for param in self.head.parameters():
                param.requires_grad = True
        
        logging.info("Model initialized successfully.")

    def forward(self, x):
        """Defines the forward pass of the model."""
        # Use forward_features to get the class token robustly
        if hasattr(self.backbone, 'forward_features'):
             features = self.backbone.forward_features(x)['x_norm_clstoken']
        else:
            features = self.backbone(x)
        
        output = self.head(features)
        return output

    def get_attention_maps(self, x):
        """
        A new method to extract attention maps from the last layer of the backbone.
        This is useful for XAI.
        """
        # We need to manually pass data through the blocks to get attentions
        # This is a simplified example; DINOv2's source might have a helper for this
        with torch.no_grad():
            # Get the attention weights from the last block
            # Note: This requires knowing the internal structure of the DINOv2 model
            # For `dinov2_vits14`, the last block is `blocks[-1]`.
            attentions = self.backbone.get_last_selfattention(x)
            # The attention map is the average over all heads
            # Shape is (batch_size, num_heads, patch_dim, patch_dim)
            # We average over the heads: (batch_size, patch_dim, patch_dim)
            return attentions.mean(dim=1)

