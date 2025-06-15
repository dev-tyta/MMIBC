# training/multimodal/multimodal_architecture.py

import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
# We need to import the unimodal model definition to be able to load its weights
from src.training.unimodal.unimodal_model import DinoV2Classifier


class CrossAttention(nn.Module):
    """
    A simple Cross-Attention module.
    One modality provides the Query, the other provides Key and Value.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5

    def forward(self, query_features, context_features):
        # query_features: (batch_size, feature_dim) -> from modality A
        # context_features: (batch_size, feature_dim) -> from modality B
        
        # Reshape to (batch_size, 1, feature_dim) to mimic sequence processing
        query = self.q_proj(query_features).unsqueeze(1)
        key = self.k_proj(context_features).unsqueeze(1)
        value = self.v_proj(context_features).unsqueeze(1)
        
        # Calculate attention scores
        attention_scores = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to value
        attended_features = (attention_weights @ value).squeeze(1)
        
        return attended_features



# --- NEW: Gated Multimodal Unit for Advanced Fusion ---
class GatedMultimodalUnit(nn.Module):
    """
    A Gated Multimodal Unit that learns to dynamically weigh and fuse features
    from two different modalities.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc_mammo = nn.Linear(input_dim, input_dim)
        self.fc_us = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(2 * input_dim, input_dim)

    def forward(self, mammo_features, us_features):
        # Apply tanh activation to normalize features to [-1, 1]
        h_mammo = torch.tanh(self.fc_mammo(mammo_features))
        h_us = torch.tanh(self.fc_us(us_features))
        
        # Concatenate features to compute the gate
        x_gate = torch.cat((mammo_features, us_features), dim=-1)
        # The gate z determines the weighting of each modality
        z = torch.sigmoid(self.gate(x_gate))
        
        # Apply the gate to fuse the features
        fused_features = z * h_mammo + (1 - z) * h_us
        return fused_features

# --- NEW: Residual Block for a deeper, more stable classifier ---
class ResidualBlock(nn.Module):
    """A residual block for the classifier head."""
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Shortcut connection if dimensions are different
        self.shortcut = nn.Identity()
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out += residual # Add the residual connection
        return out


class MultimodalFusionModel(nn.Module):
    """
    A multimodal model that fuses features using Cross-Attention,
    aligning with the planned architecture.
    """
    def __init__(self, unimodal_config, us_model_path, mammo_model_path, fusion_output_size=512):
        super().__init__()
        
        # --- Load Pre-trained Unimodal Models ---
        logging.info("Loading pre-trained unimodal encoders...")
        self.us_encoder = DinoV2Classifier(
            n_classes=unimodal_config['model']['n_classes'],
            model_name=unimodal_config['model']['name'],
            dropout_rate=unimodal_config['training']['dropout_rate']
        )
        self.us_encoder.load_state_dict(torch.load(us_model_path, map_location='cpu'))
        
        self.mammo_encoder = DinoV2Classifier(
            n_classes=unimodal_config['model']['n_classes'],
            model_name=unimodal_config['model']['name'],
            dropout_rate=unimodal_config['training']['dropout_rate']
        )
        self.mammo_encoder.load_state_dict(torch.load(mammo_model_path, map_location='cpu'))
        
        logging.info("Freezing all parameters in the pre-trained encoders.")
        for param in self.us_encoder.parameters():
            param.requires_grad = True
        for param in self.mammo_encoder.parameters():
            param.requires_grad = True
            
        feature_dim = self.us_encoder.backbone.embed_dim
        
        # --- Cross-Attention Fusion Block ---
        # This aligns with the "Fusion Level" in your diagram
        logging.info("Initializing Cross-Attention Fusion block.")
        self.mammo_attends_to_us = CrossAttention(feature_dim)
        self.us_attends_to_mammo = CrossAttention(feature_dim)
        
        # --- IMPROVED: Gated Fusion Unit ---
        # This replaces simple concatenation
        self.gated_fusion = GatedMultimodalUnit(input_dim=feature_dim)
        
        # The output of the gated unit is our starting point for classification
        fused_dim = feature_dim

        # --- IMPROVED: Deeper Classifier Head with Residual Connections ---
        logging.info("Initializing deeper multimodal classification head with residual connections.")
        self.fusion_classifier = nn.Sequential(
            ResidualBlock(fused_dim, 1024),
            ResidualBlock(1024, 512),
            nn.Linear(512, 2) # Final output layer
        )
        logging.info("Multimodal model initialized successfully.")


    def forward(self, mammo_x, us_x):
        """Forward pass for the multimodal model."""
        # Extract features from the frozen backbones
        mammo_features = self.mammo_encoder.backbone(mammo_x)
        us_features = self.us_encoder.backbone(us_x)
        
        # 2. Apply Cross-Attention
        mammo_attended = self.mammo_attends_to_us(mammo_features, us_features)
        us_attended = self.us_attends_to_mammo(us_features, mammo_features)
        
        # 3. Fuse features using the Gated Unit
        gated_fused_features = self.gated_fusion(mammo_attended, us_attended)
        
        # 4. Pass through the improved classifier
        output = self.fusion_classifier(gated_fused_features)
        
        return output