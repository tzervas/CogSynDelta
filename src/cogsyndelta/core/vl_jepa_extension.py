"""
Vision-Language Joint Embedding Predictive Architecture (VL-JEPA) Extension

This module extends the PCN-VAE-GAN hybrid with VL-JEPA-inspired components:
- Vision encoder for processing visual input (frames from host video output)
- Silent semantic state retention (no token generation)
- Joint embedding space for multimodal representations
- Memory-augmented hierarchical predictive coding
- Temporal grounding for sequence learning
- mHC (moderated Hyper Connections) for controlled layer information flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from collections import deque


class VisionEncoder(nn.Module):
    """
    Vision encoder for processing visual input into semantic embeddings.
    Processes frames from host video output without token generation.
    """
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 512, num_layers: int = 6) -> None:
        """Initialize vision encoder with patch embeddings and transformer."""
        super(VisionEncoder, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding (convolutional projection)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to semantic embedding space
        self.semantic_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode visual input to semantic embeddings.
        
        Args:
            x: Visual input [batch, channels, height, width]
            
        Returns:
            Semantic embeddings [batch, embed_dim] - silent state, no tokens
        """
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch, embed_dim, num_patches_h, num_patches_w]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transform
        x = self.transformer(x)
        
        # Global pooling and semantic projection
        x = x.mean(dim=1)  # [batch, embed_dim]
        semantic_embedding = self.semantic_proj(x)
        
        return semantic_embedding


class TemporalMemoryBank(nn.Module):
    """
    External memory bank for hierarchical temporal sequence learning.
    Implements silent semantic state retention across time.
    """
    
    def __init__(self, memory_size: int = 1000, embed_dim: int = 512, 
                 num_read_heads: int = 4) -> None:
        """Initialize temporal memory bank with attention mechanisms."""
        super(TemporalMemoryBank, self).__init__()
        
        self.memory_size = memory_size
        self.embed_dim = embed_dim
        self.num_read_heads = num_read_heads
        
        # Memory slots - persistent semantic state storage
        self.register_buffer('memory', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.register_buffer('write_pointer', torch.tensor(0, dtype=torch.long))
        
        # Attention mechanism for memory read/write
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multi-head read attention
        self.read_attention = nn.MultiheadAttention(
            embed_dim, num_read_heads, batch_first=True
        )
        
        # Write gate
        self.write_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def write(self, content: torch.Tensor, importance: Optional[torch.Tensor] = None) -> None:
        """
        Write semantic state to memory without token generation.
        
        Args:
            content: Semantic embeddings to store [batch, embed_dim]
            importance: Optional importance scores for prioritization
        """
        batch_size = content.size(0)
        
        for i in range(batch_size):
            # Compute write location
            write_idx = self.write_pointer.item() % self.memory_size
            
            # Write to memory
            self.memory[write_idx] = content[i].detach()
            self.memory_age[write_idx] = 0  # Reset age
            
            # Update pointer
            self.write_pointer = (self.write_pointer + 1) % self.memory_size
        
        # Age existing memories
        self.memory_age += 1
        
    def read(self, query: torch.Tensor, num_reads: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read relevant semantic states from memory.
        
        Args:
            query: Query embedding [batch, embed_dim]
            num_reads: Number of memory slots to retrieve
            
        Returns:
            Retrieved memories and attention weights
        """
        batch_size = query.size(0)
        
        # Compute attention scores
        query_proj = self.query_proj(query).unsqueeze(1)  # [batch, 1, embed_dim]
        memory_keys = self.key_proj(self.memory).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Multi-head attention read
        retrieved, attn_weights = self.read_attention(
            query_proj, memory_keys, memory_keys
        )
        
        return retrieved.squeeze(1), attn_weights
    
    def temporal_context(self, window_size: int = 10) -> torch.Tensor:
        """
        Get recent temporal context window.
        
        Args:
            window_size: Size of temporal window
            
        Returns:
            Recent memory states [window_size, embed_dim]
        """
        # Get most recent entries
        ptr = self.write_pointer.item()
        if ptr < window_size:
            # Wrap around
            recent = torch.cat([
                self.memory[-(window_size - ptr):],
                self.memory[:ptr]
            ], dim=0)
        else:
            recent = self.memory[ptr - window_size:ptr]
        
        return recent


class ModeratedHyperConnection(nn.Module):
    """
    mHC: Moderated Hyper Connection for controlled information flow between layers.
    Uses gating mechanisms to modulate cross-layer connections dynamically.
    """
    
    def __init__(self, embed_dim: int) -> None:
        """Initialize mHC with gating and transform networks."""
        super(ModeratedHyperConnection, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Moderation gate (learns when to pass information)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Connection transform
        self.transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Residual scaling
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Apply moderated hyper connection from source to target layer.
        
        Args:
            source: Source layer representation [batch, embed_dim]
            target: Target layer representation [batch, embed_dim]
            
        Returns:
            Modulated target representation
        """
        # Compute moderation gate based on both layers
        gate_input = torch.cat([source, target], dim=-1)
        gate_values = self.gate(gate_input)
        
        # Transform source information
        transformed = self.transform(source)
        
        # Apply moderated connection with residual
        modulated = gate_values * transformed
        output = self.alpha * modulated + (1 - self.alpha) * target
        
        return output


class HierarchicalPredictiveCoding(nn.Module):
    """
    Hierarchical predictive coding with temporal grounding and mHC.
    Multi-level prediction without token generation.
    Uses moderated hyper connections for controlled cross-layer information flow.
    """
    
    def __init__(self, embed_dim: int = 512, num_levels: int = 3) -> None:
        """Initialize hierarchical predictive coding with levels and mHC."""
        super(HierarchicalPredictiveCoding, self).__init__()
        
        self.num_levels = num_levels
        self.embed_dim = embed_dim
        
        # Hierarchical levels (coarse to fine)
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_levels)
        ])
        
        # Prediction heads for each level
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_levels)
        ])
        
        # mHC: Moderated Hyper Connections for top-down flow
        self.mhc_top_down = nn.ModuleList([
            ModeratedHyperConnection(embed_dim)
            for _ in range(num_levels - 1)
        ])
        
        # mHC: Moderated Hyper Connections for bottom-up flow
        self.mhc_bottom_up = nn.ModuleList([
            ModeratedHyperConnection(embed_dim)
            for _ in range(num_levels - 1)
        ])
        
        # Skip connections with moderation
        self.mhc_skip = nn.ModuleList([
            ModeratedHyperConnection(embed_dim)
            for _ in range(num_levels)
        ])
        
    def forward(self, x: torch.Tensor, 
                temporal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Hierarchical predictive coding with temporal grounding and mHC.
        
        Args:
            x: Current input embedding [batch, embed_dim]
            temporal_context: Optional temporal context [window_size, embed_dim]
            
        Returns:
            Dictionary with predictions and prediction errors at each level
        """
        batch_size = x.size(0)
        
        # Bottom-up pass with mHC
        representations = []
        current = x
        
        for level_idx, level in enumerate(self.levels):
            # Apply level transformation
            level_output = level(current)
            
            # Apply mHC skip connection (moderated residual)
            if level_idx > 0:
                level_output = self.mhc_skip[level_idx](x, level_output)
            
            representations.append(level_output)
            
            # mHC bottom-up connection to next level
            if level_idx < self.num_levels - 1:
                current = self.mhc_bottom_up[level_idx](level_output, level_output)
            else:
                current = level_output
        
        # Top-down predictions with mHC
        predictions = []
        prediction_errors = []
        moderated_states = list(representations)  # Copy for modification
        
        for level_idx in range(self.num_levels - 1, -1, -1):
            # Predict current level
            prediction = self.predictors[level_idx](moderated_states[level_idx])
            predictions.append(prediction)
            
            # Compute prediction error (silent semantic difference)
            if level_idx < self.num_levels - 1:
                # mHC top-down influence (moderated)
                top_down_moderated = self.mhc_top_down[level_idx](
                    moderated_states[level_idx + 1],
                    moderated_states[level_idx]
                )
                prediction_error = moderated_states[level_idx] - top_down_moderated
                
                # Update with moderated top-down info
                moderated_states[level_idx] = top_down_moderated
            else:
                # Highest level compares with itself
                prediction_error = torch.zeros_like(prediction)
            
            prediction_errors.append(prediction_error)
        
        return {
            'representations': representations,
            'predictions': predictions[::-1],  # Reverse to match level order
            'prediction_errors': prediction_errors[::-1],
            'moderated_states': moderated_states,
            'final_state': moderated_states[-1]  # Silent semantic state with mHC
        }


class JointEmbeddingSpace(nn.Module):
    """
    Joint embedding space for vision-language-action alignment.
    Predicts semantic embeddings, not tokens.
    """
    
    def __init__(self, embed_dim: int = 512, latent_dim: int = 20) -> None:
        """Initialize joint embedding space with projections and predictor."""
        super(JointEmbeddingSpace, self).__init__()
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        # Embedding projections
        self.vision_proj = nn.Linear(embed_dim, embed_dim)
        self.state_proj = nn.Linear(latent_dim, embed_dim)
        
        # Joint space predictor
        self.joint_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Semantic similarity
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, vision_embed: torch.Tensor, 
                state_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Align vision and state in joint embedding space.
        
        Args:
            vision_embed: Vision embeddings [batch, embed_dim]
            state_embed: State embeddings [batch, latent_dim]
            
        Returns:
            Joint embeddings and alignment scores
        """
        # Project to joint space
        vision_proj = self.vision_proj(vision_embed)
        state_proj = self.state_proj(state_embed)
        
        # Concatenate and predict joint embedding
        combined = torch.cat([vision_proj, state_proj], dim=-1)
        joint_embed = self.joint_predictor(combined)
        
        # Compute semantic similarity (no token generation)
        vision_norm = F.normalize(vision_proj, dim=-1)
        state_norm = F.normalize(state_proj, dim=-1)
        similarity = torch.matmul(vision_norm, state_norm.T) / self.temperature
        
        return {
            'joint_embedding': joint_embed,
            'similarity': similarity,
            'vision_proj': vision_proj,
            'state_proj': state_proj
        }


class FrameBufferAdapter(nn.Module):
    """
    Adapter for processing host video output frames.
    Handles frame buffering, preprocessing, and temporal alignment.
    """
    
    def __init__(self, buffer_size: int = 16, target_size: int = 224) -> None:
        """Initialize frame buffer with preprocessing and temporal pooling."""
        super(FrameBufferAdapter, self).__init__()
        
        self.buffer_size = buffer_size
        self.target_size = target_size
        
        # Frame buffer (FIFO)
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Preprocessing
        self.normalize = nn.BatchNorm2d(3)
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
    def preprocess_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Preprocess single frame from host video output.
        
        Args:
            frame: Raw frame [channels, height, width]
            
        Returns:
            Preprocessed frame [channels, target_size, target_size]
        """
        # Resize if needed
        if frame.size(-1) != self.target_size or frame.size(-2) != self.target_size:
            frame = F.interpolate(
                frame.unsqueeze(0), 
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Normalize
        frame = self.normalize(frame.unsqueeze(0)).squeeze(0)
        
        return frame
    
    def add_frame(self, frame: torch.Tensor) -> None:
        """Add frame to buffer."""
        preprocessed = self.preprocess_frame(frame)
        self.frame_buffer.append(preprocessed)
    
    def get_current_frame(self) -> Optional[torch.Tensor]:
        """Get most recent frame."""
        if len(self.frame_buffer) > 0:
            return self.frame_buffer[-1]
        return None
    
    def get_temporal_window(self, window_size: int = 8) -> Optional[torch.Tensor]:
        """
        Get temporal window of frames.
        
        Args:
            window_size: Number of frames to retrieve
            
        Returns:
            Frame sequence [window_size, channels, height, width]
        """
        if len(self.frame_buffer) == 0:
            return None
        
        # Get recent frames
        frames = list(self.frame_buffer)[-window_size:]
        
        # Pad if needed
        while len(frames) < window_size:
            frames.insert(0, frames[0])  # Repeat first frame
        
        return torch.stack(frames)


# Integration helpers

def create_vision_language_extension(config: Dict) -> Dict[str, nn.Module]:
    """
    Create VL-JEPA extension modules for the PCN-VAE-GAN hybrid.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of extension modules
    """
    embed_dim = config.get('vision_language', {}).get('embed_dim', 512)
    latent_dim = config['exploratory']['latent_dim']
    
    modules = {
        'vision_encoder': VisionEncoder(embed_dim=embed_dim),
        'memory_bank': TemporalMemoryBank(embed_dim=embed_dim),
        'hierarchical_pcn': HierarchicalPredictiveCoding(embed_dim=embed_dim),
        'joint_space': JointEmbeddingSpace(embed_dim=embed_dim, latent_dim=latent_dim),
        'frame_adapter': FrameBufferAdapter()
    }
    
    return modules


if __name__ == '__main__':
    # Example usage
    print("VL-JEPA Extension - Silent Semantic State Retention")
    print("="*60)
    
    # Create modules
    vision_encoder = VisionEncoder()
    memory_bank = TemporalMemoryBank()
    hpc = HierarchicalPredictiveCoding()
    joint_space = JointEmbeddingSpace()
    
    # Simulate visual input
    frame = torch.randn(1, 3, 224, 224)
    
    # Encode to semantic embedding (no tokens)
    semantic_embed = vision_encoder(frame)
    print(f"✓ Vision encoding: {frame.shape} → {semantic_embed.shape}")
    print(f"  Silent semantic state (no token generation)")
    
    # Write to memory
    memory_bank.write(semantic_embed)
    print(f"✓ Semantic state stored in memory")
    
    # Hierarchical prediction with temporal grounding
    hpc_output = hpc(semantic_embed)
    print(f"✓ Hierarchical predictive coding: {len(hpc_output['representations'])} levels")
    
    # Joint embedding
    state = torch.randn(1, 20)
    joint_output = joint_space(semantic_embed, state)
    print(f"✓ Joint embedding space: {joint_output['joint_embedding'].shape}")
    
    print("\nAll modules operational ✓")
    print("Efficient visual processing without token generation")
