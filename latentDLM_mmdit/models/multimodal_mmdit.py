# File: hdlm/models/multimodal_mmdit.py
import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")


try:
    from mmdit.mmdit_generalized_pytorch import MMDiT
    has_mmdit = True
    print("✓ MMDiT module imported successfully")
except ImportError as e:
    print(f"✗ Error importing MMDiT: {e}")
    print("Make sure mmdit_generalized_pytorch.py is in your Python path")
    has_mmdit = False
    raise ImportError("MMDiT package not found. Please install it.")


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        
        # Use float32 for calculations, then convert to target dtype
        device = t.device
        freqs = torch.exp(
            -math.log(max_period) 
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device) 
            / half
        )
        
        # Use t in float32 for calculation
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        # Convert to same dtype as t
        return embedding.to(dtype=t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TextEncoder(nn.Module):
    """Text token encoder for MMDiT."""
    
    def __init__(self, vocab_size, hidden_size, max_length=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Parameter(torch.randn(1, max_length, hidden_size) * 0.02)
        self.max_length = max_length
        
    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        token_emb = self.token_embed(token_ids)
        pos_emb = self.position_embed[:, :seq_len]
        return token_emb + pos_emb


class LatentEncoder(nn.Module):
    """Latent vector encoder for MMDiT."""
    
    def __init__(self, latent_dim, hidden_size, max_latent_len=1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.position_embed = nn.Parameter(torch.randn(1, max_latent_len, hidden_size) * 0.02)
        self.max_latent_len = max_latent_len
        
    def forward(self, latents):
        # latents: [batch, seq_len, latent_dim] or [batch, latent_dim]
        if latents.dim() == 2:
            latents = latents.unsqueeze(1)  # [batch, 1, latent_dim]
        
        batch_size, seq_len, _ = latents.shape
        latent_emb = self.projection(latents)
        pos_emb = self.position_embed[:, :seq_len]
        return latent_emb + pos_emb


# class MultimodalConditioning(nn.Module):
#     """Conditioning for both text and latent diffusion."""
    
#     def __init__(self, cond_dim):
#         super().__init__()
#         self.cond_dim = cond_dim
        
#         # Text timestep embedding
#         self.text_sigma_map = TimestepEmbedder(cond_dim)
        
#         # Latent timestep embedding
#         self.latent_sigma_map = TimestepEmbedder(cond_dim)
        
#         # Combine both conditioning
#         self.combine = nn.Sequential(
#             nn.Linear(cond_dim * 2, cond_dim * 4),
#             nn.SiLU(),
#             nn.Linear(cond_dim * 4, cond_dim)
#         )
    
#     def forward(self, text_timesteps, latent_timesteps):
#         # Ensure dtype match for Linear: mat1 (input) and mat2 (weight)
#         w_dtype = self.text_sigma_map.weight.dtype
#         text_timesteps = text_timesteps.to(dtype=w_dtype)
#         latent_timesteps = latent_timesteps.to(dtype=w_dtype)

#         # Embed both timesteps
#         text_cond = F.silu(self.text_sigma_map(text_timesteps))
#         latent_cond = F.silu(self.latent_sigma_map(latent_timesteps))
        
#         # Combine
#         combined = torch.cat([text_cond, latent_cond], dim=-1)
#         c = self.combine(combined)
#         return c


class MultimodalConditioning(nn.Module):
    """Conditioning for both text and latent diffusion."""
    
    def __init__(self, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim
        
        # Text timestep embedding
        self.text_sigma_map = TimestepEmbedder(cond_dim)
        
        # Latent timestep embedding
        self.latent_sigma_map = TimestepEmbedder(cond_dim)
        
        # Combine both conditioning
        self.combine = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim)
        )
    
    def forward(self, text_timesteps, latent_timesteps):
        # Get dtype from model parameters
        model_dtype = next(self.parameters()).dtype
        
        # Convert timesteps if needed
        if text_timesteps.dtype != model_dtype:
            text_timesteps = text_timesteps.to(dtype=model_dtype)
        if latent_timesteps.dtype != model_dtype:
            latent_timesteps = latent_timesteps.to(dtype=model_dtype)

        # Embed both timesteps
        text_cond = F.silu(self.text_sigma_map(text_timesteps))
        latent_cond = F.silu(self.latent_sigma_map(latent_timesteps))
        
        # Combine
        combined = torch.cat([text_cond, latent_cond], dim=-1)
        c = self.combine(combined)
        return c

class MultimodalMMDiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """Multimodal MMDiT for joint text-latent generation."""
    
    def __init__(self, config, vocab_size: int, latent_dim: int = 1024, cluster_size: int = 100):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        if not has_mmdit:
            raise ImportError("MMDiT package not available. Please install it.")
        
        self.conditional_mode = None  # None, "l2t", "t2l"

        self.config = config
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.cluster_size = cluster_size
        self.rounded_vocab_size = vocab_size + cluster_size + (128 - (vocab_size + cluster_size) % 128) % 128

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=self.rounded_vocab_size,
            hidden_size=config.hidden_size,
            max_length=config.max_seq_len
        )
        
        # Latent encoder
        self.latent_encoder = LatentEncoder(
            latent_dim=latent_dim,
            hidden_size=config.hidden_size,
            max_latent_len=1  # We use 1 latent token
        )
        
        # Conditioning
        self.conditioning = MultimodalConditioning(config.cond_dim)
        
        # MMDiT backbone
        self.mmdit = MMDiT(
            depth=config.n_blocks,
            dim_modalities=(config.hidden_size, config.hidden_size),  # Both text and latent use same hidden size
            dim_cond=config.cond_dim,
            qk_rmsnorm=config.get("qk_rmsnorm", True),
            num_residual_streams=config.get("num_residual_streams", 2),
        )
        
        # Text output head
        self.text_head = nn.Linear(config.hidden_size, self.rounded_vocab_size)
        
        # Latent output head (predicts noise for continuous diffusion)
        self.latent_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, latent_dim),
        )
        
        # Cluster head (if needed)
        if cluster_size > 0:
            self.text_head_clusters = nn.Linear(config.hidden_size, self.rounded_vocab_size)
        else:
            self.text_head_clusters = None
        
        # Initialize weights
        self._init_weights()
        
        print(f"Initialized MultimodalMMDiT with:")
        print(f"  Vocab size: {self.rounded_vocab_size}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  MMDiT depth: {config.n_blocks}")
        print(f"  Residual streams: {config.get('num_residual_streams', 2)}")
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.text_head.weight, std=0.02)
        nn.init.zeros_(self.text_head.bias)
        nn.init.normal_(self.latent_head[-1].weight, std=0.02)
        nn.init.zeros_(self.latent_head[-1].bias)
        
        if self.text_head_clusters is not None:
            nn.init.normal_(self.text_head_clusters.weight, std=0.02)
            nn.init.zeros_(self.text_head_clusters.bias)
    
    def forward(self, text_tokens, latents, text_timesteps, latent_timesteps, attention_mask=None):
        # Encode text
        text_emb = self.text_encoder(text_tokens)
        model_dtype = next(self.parameters()).dtype
        
        # Encode latents (if provided)
        if latents is not None:
            # Ensure latents have correct dtype
            latents = latents.to(dtype=text_emb.dtype)
            
            # Handle different latent shapes
            if latents.dim() == 2:  # [batch, latent_dim]
                latents = latents.unsqueeze(1)  # [batch, 1, latent_dim]
            elif latents.dim() == 3 and latents.shape[1] > 1:  # [batch, seq_len, latent_dim]
                # Take mean across sequence dimension
                latents = latents.mean(dim=1, keepdim=True)
            
            latent_emb = self.latent_encoder(latents)
        else:
            latent_emb = None
        
        # Get conditioning
        c = self.conditioning(text_timesteps, latent_timesteps)
        c = c.to(dtype=text_emb.dtype)
        
        # FIX: Ensure boolean masks
        if attention_mask is not None:
            # Convert to boolean if it's not already
            text_mask = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
        else:
            # Create boolean mask
            text_mask = torch.ones_like(text_tokens, dtype=torch.bool)
        
        # Create boolean latent mask
        if latents is not None:
            latent_mask = torch.ones(latent_emb.shape[:2], device=text_tokens.device, dtype=torch.bool)
        else:
            latent_mask = None
        
        # Pass through MMDiT
        if latent_emb is not None:
            text_out, latent_out = self.mmdit(
                modality_tokens=(text_emb, latent_emb),
                modality_masks=(text_mask, latent_mask),
                time_cond=c,
            )
        else:
            # Text-only mode
            text_out, _ = self.mmdit(
                modality_tokens=(text_emb,),
                modality_masks=(text_mask,),
                time_cond=c,
            )
            latent_out = None
        
        # Text predictions
        text_logits = self.text_head(text_out)
        
        # Latent predictions
        latent_pred = None
        if latent_out is not None:
            latent_pred = self.latent_head(latent_out)
            # If we have multiple latent tokens, average them
            if latent_pred.shape[1] > 1:
                latent_pred = latent_pred.mean(dim=1)
        
        # Handle cluster outputs if needed
        if self.cluster_size > 0 and self.text_head_clusters is not None:
            text_cluster_logits = self.text_head_clusters(text_out)
            return text_logits, latent_pred, text_cluster_logits
        
        return text_logits, latent_pred