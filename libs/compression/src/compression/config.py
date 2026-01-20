"""Compression configuration."""

from dataclasses import dataclass, field


@dataclass
class CompressionConfig:
    """Configuration for compression pipeline.

    Attributes:
        # Matryoshka settings
        mrl_dimensions: Target dimensions for truncation (e.g., [2048, 1024, 512, 256])
        mrl_loss_weights: Weights for each dimension in training loss

        # QINCo2 settings
        qinco_stages: Number of residual quantization stages (default: 4)
        qinco_codebook_size: Size of implicit neural codebook (default: 256)

        # RVQ settings
        rvq_stages: Number of residual stages (default: 4)
        rvq_codebook_size: Codebook size per stage (default: 256)
        rvq_bits: Bits per code (default: 8)

        # BitNet settings
        bitnet_hidden_multiplier: Hidden size multiplier for ternary (default: 2.0)
        bitnet_quantize_weights: Whether to quantize weights (default: True)

        # Pipeline settings
        target_fidelity: Target cosine similarity (default: 0.95)
        target_compression_ratio: Target compression ratio (default: 10.0)
        device: Compute device (default: "cuda")
    """

    # Matryoshka
    mrl_dimensions: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128])
    mrl_loss_weights: list[float] | None = None  # If None, uniform weights

    # QINCo2
    qinco_stages: int = 4
    qinco_codebook_size: int = 256

    # RVQ
    rvq_stages: int = 4
    rvq_codebook_size: int = 256
    rvq_bits: int = 8

    # BitNet
    bitnet_hidden_multiplier: float = 2.0
    bitnet_quantize_weights: bool = True

    # Pipeline
    target_fidelity: float = 0.95
    target_compression_ratio: float = 10.0
    device: str = "cuda"
