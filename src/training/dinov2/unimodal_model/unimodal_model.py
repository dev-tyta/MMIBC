# training/unimodal/unimodal_model.py

import torch
import torch.nn as nn
import logging
import types

class DinoV2Classifier(nn.Module):
    """
    A versatile classifier model using a DINOv2 backbone.
    This version includes a helper method for progressive fine-tuning.
    """
    def __init__(self, n_classes, model_name='dinov2_vits14', dropout_rate=0.5):
        super().__init__()
        self.n_classes = n_classes
        logging.info(f"Initializing model with {n_classes} classes and backbone {model_name}.")
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, force_reload=False)
        n_features = self.backbone.embed_dim 
        
        self.head = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_features // 2, n_classes)
        )
        
        logging.info("Model initialized successfully.")

    def forward(self, x):
        """Defines the forward pass of the model."""
        if hasattr(self.backbone, 'forward_features'):
             features = self.backbone.forward_features(x)['x_norm_clstoken']
        else:
            features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self):
        """Freezes all parameters in the backbone."""
        logging.info("Freezing backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreezes all parameters in the backbone."""
        logging.info("Unfreezing all backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n):
        """
        Unfreezes the last 'n' transformer blocks of the backbone.
        
        Args:
            n (int): The number of transformer blocks to unfreeze from the end.
        """
        if n <= 0:
            return
            
        # First, ensure the entire backbone is frozen
        self.freeze_backbone()
        
        # The transformer blocks are stored in a ModuleList
        num_blocks = len(self.backbone.blocks)
        n = min(n, num_blocks) # Ensure n is not larger than the number of blocks
        
        logging.info(f"Unfreezing the last {n} transformer blocks.")
        # Iterate backwards and unfreeze the last n blocks
        for i in range(num_blocks - n, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

    def get_attention_maps(self, x):
        """
        Extracts attention maps from the last attention layer of the backbone
        using a forward hook. This is the correct and stable method.
        """
        # A list to store the captured attention maps
        attention_maps = []
        
        # 1. Define a new forward method that captures the attention map
        def new_attn_forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            x = nn.functional.scaled_dot_product_attention(q, k, v)
            
            # --- Capture the attention weights ---
            # Re-compute attention weights to capture them, as the fused
            # scaled_dot_product_attention does not return them directly.
            attn_weights = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
            attention_maps.append(attn_weights)

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            return x

        # 2. Temporarily replace the original forward method with our new one
        attn_module = self.backbone.blocks[-1].attn
        original_forward = attn_module.forward
        attn_module.forward = types.MethodType(new_attn_forward, attn_module)

        # 3. Perform a forward pass on the backbone to trigger our custom method
        with torch.no_grad():
            self.backbone(x)

        # 4. Restore the original forward method to leave the model unchanged
        attn_module.forward = original_forward
        
        if not attention_maps:
            raise ValueError("Could not capture attention maps. The monkey-patching failed.")

        # The captured attention map has the shape: (batch, num_heads, num_patches+1, num_patches+1)
        # We return the first (and only) item from our list.
        return attention_maps[0]