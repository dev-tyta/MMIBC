import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding with positional encoding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Initialize parameters
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Project patches
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
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
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        
    def forward(self, x):
        return self.gamma * x

class ViTBlock(nn.Module):
    """
    Vision Transformer Block with LayerScale
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                    attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=layer_scale_init_value) if layer_scale_init_value > 0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=layer_scale_init_value) if layer_scale_init_value > 0 else nn.Identity()
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # Attention block
        res = x
        x = self.norm1(x)
        x, attn = self.attn(x)
        x = self.ls1(x)
        x = res + self.drop_path(x)
        
        # MLP block
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = res + self.drop_path(x)
        
        return x, attn

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder (DINOv2-like)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-5):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                norm_layer=norm_layer, layer_scale_init_value=layer_scale_init_value
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Store attention maps for visualization
        attn_maps = []
        
        # Apply transformer blocks
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)
        
        # Final norm
        x = self.norm(x)
        
        # Extract CLS token
        cls_token = x[:, 0]
        
        return cls_token, x, attn_maps

class CrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Fusion Module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key_value):
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape
        
        # Project query, key, value
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
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

class MultimodalBreastCancerModel(nn.Module):
    """
    Complete Multimodal Breast Cancer Diagnosis Model
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.1, fusion_dim=768, fusion_heads=8, num_classes=2):
        super().__init__()
        
        # Mammogram encoder
        self.mammogram_encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate
        )
        
        # Ultrasound encoder
        self.ultrasound_encoder = ViTEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate
        )
        
        # Fusion module
        self.fusion = CrossAttentionFusion(
            dim=fusion_dim, num_heads=fusion_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate
        )
        
        # Classification head
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, mammo, us):
        # Encode mammogram
        mammo_cls, mammo_tokens, mammo_attns = self.mammogram_encoder(mammo)
        
        # Encode ultrasound
        us_cls, us_tokens, us_attns = self.ultrasound_encoder(us)
        
        # Fuse with cross-attention (using mammogram as query, ultrasound as key/value)
        fused_feat, fusion_attn = self.fusion(mammo_tokens, us_tokens)
        
        # Extract CLS token from fused features
        fused_cls = fused_feat[:, 0]
        
        # Classification
        x = self.norm(fused_cls)
        x = self.head(x)
        
        return x, {
            'mammo_cls': mammo_cls,
            'us_cls': us_cls,
            'fused_cls': fused_cls,
            'mammo_attns': mammo_attns,
            'us_attns': us_attns,
            'fusion_attn': fusion_attn
        }
    
    def get_attention_maps(self, mammo, us):
        """
        Get attention maps for explainability
        """
        with torch.no_grad():
            _, outputs = self.forward(mammo, us)
            
        return {
            'mammo_attns': outputs['mammo_attns'],
            'us_attns': outputs['us_attns'],
            'fusion_attn': outputs['fusion_attn']
        }