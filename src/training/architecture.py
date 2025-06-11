import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat # Assuming einops is installed

# Helper for DropPath (Stochastic Depth) - often used in ViTs
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding with positional encoding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection using Conv2d
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Initialize parameters
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W)

        Returns:
            torch.Tensor: Patch embeddings with CLS token and positional encoding (B, num_patches + 1, embed_dim)
        """
        B, C, H, W = x.shape

        # Ensure input size matches expected
        # assert H == self.img_size and W == self.img_size, \
        #     f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})."

        # Project patches
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        # Flatten spatial dimensions and transpose to get (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        # Expand CLS token to match batch size
        cls_token = self.cls_token.expand(B, -1, -1)
        # Concatenate CLS token with patch embeddings
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-Attention module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # Scaling factor for attention scores

        # Linear projection for Query, Key, Value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) # Dropout for attention scores
        self.proj = nn.Linear(dim, dim) # Output projection
        self.proj_drop = nn.Dropout(proj_drop) # Dropout for output

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, C) where N is sequence length, C is dimension

        Returns:
            tuple: Output tensor (B, N, C) and attention weights (B, num_heads, N, N)
        """
        B, N, C = x.shape

        # Project and reshape Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Separate Q, K, V
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # Compute attention scores
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply softmax to get attention probabilities
        attn = attn.softmax(dim=-1)
        # Apply attention dropout
        attn = self.attn_drop(attn)

        # Apply attention to values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2) # Transpose heads and sequence length
        # Reshape back to original dimension
        x = x.reshape(B, N, C)

        # Apply output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class MLP(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer, etc.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Two linear layers with activation and dropout
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, in_features)

        Returns:
            torch.Tensor: Output tensor (B, N, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    """
    Layer scale from CaiT and for DINOv2
    """
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        # Learnable parameter gamma initialized to small values
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, dim)

        Returns:
            torch.Tensor: Scaled output tensor
        """
        return self.gamma * x

class ViTBlock(nn.Module):
    """
    Vision Transformer Block with LayerScale and DropPath
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5):
        super().__init__()
        # First normalization layer
        self.norm1 = norm_layer(dim)
        # Multi-Head Self-Attention module
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        # LayerScale after attention (optional)
        self.ls1 = LayerScale(dim, init_values=layer_scale_init_value) if layer_scale_init_value > 0 else nn.Identity()

        # Second normalization layer
        self.norm2 = norm_layer(dim)
        # MLP (Feed-Forward) module
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # LayerScale after MLP (optional)
        self.ls2 = LayerScale(dim, init_values=layer_scale_init_value) if layer_scale_init_value > 0 else nn.Identity()

        # Stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, dim)

        Returns:
            tuple: Output tensor (B, N, dim) and attention weights (B, num_heads, N, N) from the attention block
        """
        # Attention block with residual connection, normalization, LayerScale, and DropPath
        res = x
        x_norm = self.norm1(x)
        x_attn, attn = self.attn(x_norm)
        x_attn = self.ls1(x_attn)
        x = res + self.drop_path(x_attn)

        # MLP block with residual connection, normalization, LayerScale, and DropPath
        res = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x_mlp = self.ls2(x_mlp)
        x = res + self.drop_path(x_mlp)

        return x, attn

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder (Backbone)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5):
        super().__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Transformer blocks (encoder layers)
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, layer_scale_init_value=layer_scale_init_value
            )
            for i in range(depth)
        ])

        # Final normalization layer
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W)

        Returns:
            tuple: CLS token representation (B, embed_dim), all tokens (B, num_patches + 1, embed_dim),
                   list of attention maps from each block
        """
        # Apply patch embedding
        x = self.patch_embed(x)

        # Store attention maps from each block
        attn_maps = []

        # Apply transformer blocks sequentially
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)

        # Apply final normalization
        x = self.norm(x)

        # Extract the CLS token (the first token in the sequence)
        cls_token = x[:, 0]

        # Return CLS token, all tokens, and attention maps
        return cls_token, x, attn_maps

class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Fusion Module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # Scaling factor

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # Output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        """
        Args:
            query (torch.Tensor): Query tensor (B, N_q, C)
            key_value (torch.Tensor): Key/Value tensor (B, N_kv, C)

        Returns:
            tuple: Fused output tensor (B, N_q, C), cross-attention weights (B, num_heads, N_q, N_kv)
        """
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape

        # Project and reshape Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention scores
        # (B, num_heads, N_q, head_dim) @ (B, num_heads, head_dim, N_kv) -> (B, num_heads, N_q, N_kv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply softmax
        attn = attn.softmax(dim=-1)
        # Apply attention dropout
        attn = self.attn_drop(attn)

        # Apply attention to values
        # (B, num_heads, N_q, N_kv) @ (B, num_heads, N_kv, head_dim) -> (B, num_heads, N_q, head_dim)
        x = (attn @ v).transpose(1, 2) # Transpose heads and query sequence length
        # Reshape back to original dimension
        x = x.reshape(B, N_q, C)

        # Apply output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class MultimodalBreastCancerModel(nn.Module):
    """
    Complete Multimodal Breast Cancer Diagnosis Model
    Combines two ViT encoders and a Cross-Attention fusion module.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.1, fusion_dim=768, fusion_heads=8, num_classes=2):
        super().__init__()

        # Mammogram encoder (ViT backbone)
        self.mammogram_encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate
        )

        # Ultrasound encoder (ViT backbone)
        self.ultrasound_encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate
        )

        # Cross-attention fusion module
        # Fusion dimension should match encoder embedding dimension
        assert fusion_dim == embed_dim, "Fusion dimension must match encoder embedding dimension."
        self.fusion = CrossAttentionFusion(
            dim=fusion_dim, num_heads=fusion_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate
        )

        # Classification head
        # Takes the fused CLS token representation
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512), # First dense layer
            nn.ReLU(),                 # Activation
            nn.Dropout(0.1),           # Dropout for regularization
            nn.Linear(512, num_classes) # Output layer

    def forward(self, mammo, us):
        """
        Args:
            mammo (torch.Tensor): Mammogram image tensor (B, C, H, W)
            us (torch.Tensor): Ultrasound image tensor (B, C, H, W)

        Returns:
            tuple: Classification logits (B, num_classes), dictionary of intermediate outputs (including attention maps)
        """
        # Encode mammogram: get CLS token, all tokens, and attention maps
        mammo_cls, mammo_tokens, mammo_attns = self.mammogram_encoder(mammo)

        # Encode ultrasound: get CLS token, all tokens, and attention maps
        us_cls, us_tokens, us_attns = self.ultrasound_encoder(us)

        # Fuse features using cross-attention
        # Using mammogram tokens as Query, ultrasound tokens as Key/Value
        # The output shape will be (B, N_mammo + 1, fusion_dim)
        fused_feat, fusion_attn = self.fusion(mammo_tokens, us_tokens)

        # Extract the CLS token from the fused features
        # This corresponds to the CLS token position in the mammogram sequence
        fused_cls = fused_feat[:, 0]

        # Pass the fused CLS token through the classification head
        x = self.norm(fused_cls)
        logits = self.head(x)

        # Return logits and intermediate outputs for analysis/XAI
        return logits, {
            'mammo_cls': mammo_cls,
            'us_cls': us_cls,
            'fused_cls': fused_cls,
            'mammo_tokens': mammo_tokens, # Include tokens for potential XAI like Grad-CAM
            'us_tokens': us_tokens,
            'fused_tokens': fused_feat,
            'mammo_attns': mammo_attns,
            'us_attns': us_attns,
            'fusion_attn': fusion_attn
        }

    def get_attention_maps(self, mammo, us):
        """
        Get attention maps for explainability (requires model in eval mode)
        """
        # Ensure model is in eval mode before calling this for consistent behavior
        # self.eval()
        with torch.no_grad():
            _, outputs = self.forward(mammo, us)

        return {
            'mammo_self_attention': outputs['mammo_attns'],
            'us_self_attention': outputs['us_attns'],
            'cross_attention': outputs['fusion_attn']
        }

