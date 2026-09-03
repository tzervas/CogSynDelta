"""
Dynamic Model Sectioning and Intelligent Loading

Implements brain-inspired architecture with specialized sections:
1. Model sections as specialized brain regions
2. mHC as virtualized white matter interconnect
3. Dynamic loading of active sections for inference
4. Cross-section inference and communication
5. Automatic memory/timescale scaling with model parameters

Architecture:
- Sections: Specialized processing units (like cortical areas)
- mHC: Communication pathways (like white matter tracts)
- Dynamic loading: Load only needed sections (sparse activation)
- Scaling: Memory and timescales scale with total parameters

Benefits:
- Efficient inference (load only what's needed)
- Scalable to massive models
- Brain-inspired modularity
- Flexible specialization
"""

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn


class BrainRegionType(str, Enum):
    """Types of specialized brain regions/model sections."""

    VISUAL_CORTEX = "visual_cortex"  # Visual processing
    AUDITORY_CORTEX = "auditory_cortex"  # Audio processing
    LANGUAGE_CORTEX = "language_cortex"  # Language/text processing
    MOTOR_CORTEX = "motor_cortex"  # Action generation
    PREFRONTAL_CORTEX = "prefrontal_cortex"  # Planning and reasoning
    HIPPOCAMPUS = "hippocampus"  # Memory encoding/retrieval
    TEMPORAL_CORTEX = "temporal_cortex"  # Temporal processing
    PARIETAL_CORTEX = "parietal_cortex"  # Spatial reasoning
    CEREBELLUM = "cerebellum"  # Fine-tuning and coordination


@dataclass
class ModelSectionMetadata:
    """Metadata for a model section."""

    section_id: str
    region_type: BrainRegionType
    num_parameters: int
    input_dim: int
    output_dim: int
    specialization: str
    activation_frequency: int = 0
    last_loaded: float | None = None
    loaded: bool = False
    importance_score: float = 1.0


@dataclass
class mHCPathway:
    """mHC pathway connecting two brain regions (white matter tract)."""

    source_section: str
    target_section: str
    pathway_strength: float = 1.0
    transmission_latency: float = 0.0
    bandwidth: int = 512  # Embedding dimension
    active: bool = True


class ModelSection(nn.Module):
    """
    A specialized section of the model (brain region).

    Each section has:
    - Specialized processing layers
    - Input/output interfaces for mHC communication
    - Dynamic load/unload capability
    """

    def __init__(
        self,
        section_id: str,
        region_type: BrainRegionType,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ) -> None:
        """Initialize model section with specialized processing layers."""
        super(ModelSection, self).__init__()

        self.section_id = section_id
        self.region_type = region_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Specialized processing layers
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([nn.Linear(current_dim, next_dim), nn.LayerNorm(next_dim), nn.GELU()])
            current_dim = next_dim

        self.processing = nn.Sequential(*layers)

        # Input interface for receiving from mHC
        self.input_interface = nn.Linear(input_dim, input_dim)

        # Output interface for sending via mHC
        self.output_interface = nn.Linear(output_dim, output_dim)

        # Section state (for stateful processing)
        self.register_buffer("section_state", torch.zeros(1, output_dim))

        # Metadata
        self.metadata = ModelSectionMetadata(
            section_id=section_id,
            region_type=region_type,
            num_parameters=sum(p.numel() for p in self.parameters()),
            input_dim=input_dim,
            output_dim=output_dim,
            specialization=region_type.value,
        )

    def forward(self, x: torch.Tensor, external_input: torch.Tensor | None = None) -> torch.Tensor:
        """
        Process input through this brain region.

        Args:
            x: Primary input
            external_input: Optional input from other sections via mHC

        Returns:
            Processed output
        """
        # Receive input through interface
        interfaced = self.input_interface(x)

        # Integrate external input if provided (cross-section communication)
        if external_input is not None:
            interfaced = interfaced + 0.5 * external_input

        # Process through specialized layers
        output = self.processing(interfaced)

        # Update section state
        self.section_state = output[:1].detach()

        # Send through output interface
        final_output = self.output_interface(output)

        return final_output

    def get_state(self) -> torch.Tensor:
        """Get current section state."""
        return self.section_state

    def save_to_disk(self, path: str) -> None:
        """Save section to disk for dynamic loading."""
        torch.save({"state_dict": self.state_dict(), "metadata": self.metadata}, path)

    def load_from_disk(self, path: str) -> None:
        """Load section from disk.

        Uses weights_only=False since we serialize ModelSectionMetadata.
        This is safe for self-generated checkpoints from save_to_disk().
        """
        # PyTorch 2.6+ defaults to weights_only=True, but we need to load
        # custom dataclasses (ModelSectionMetadata). This is safe for
        # checkpoints we create ourselves.
        # weights_only=True is the safe default in torch>=2.6; it refuses to execute
        # arbitrary pickle during load. We still need our own metadata dataclass, so it
        # is allow-listed explicitly rather than disabling the protection wholesale.
        # This matters more than it looks: the moment a checkpoint can arrive from a Hub
        # or a peer instead of being self-generated, weights_only=False is remote code
        # execution on load.
        torch.serialization.add_safe_globals([ModelSectionMetadata, BrainRegionType])
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["state_dict"])
        self.metadata = checkpoint["metadata"]


class mHCInterconnect(nn.Module):
    """
    mHC (Moderated Hyper Connection) Interconnect.

    Virtualized white matter enabling communication between brain regions.
    Dynamically routes information between active sections.
    """

    def __init__(self, embed_dim: int = 512) -> None:
        """Initialize mHC interconnect with routing and modulation networks."""
        super(mHCInterconnect, self).__init__()

        self.embed_dim = embed_dim

        # Pathways (white matter tracts)
        self.pathways: dict[tuple[str, str], mHCPathway] = {}

        # Routing network (learns optimal paths)
        self.router = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # Modulation network (controls signal strength)
        self.modulator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def register_pathway(
        self, source: str, target: str, strength: float = 1.0, latency: float = 0.0
    ) -> Any:
        """Register a communication pathway between sections."""
        pathway = mHCPathway(
            source_section=source,
            target_section=target,
            pathway_strength=strength,
            transmission_latency=latency,
            bandwidth=self.embed_dim,
        )
        self.pathways[(source, target)] = pathway

    def transmit(
        self,
        source_id: str,
        target_id: str,
        signal: torch.Tensor,
        source_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transmit signal from source to target section via mHC pathway.

        Args:
            source_id: Source section ID
            target_id: Target section ID
            signal: Signal to transmit
            source_state: Current state of source section
            target_state: Current state of target section

        Returns:
            Modulated signal for target
        """
        pathway_key = (source_id, target_id)

        if pathway_key not in self.pathways:
            # No direct pathway, return zero signal
            return torch.zeros_like(signal)

        pathway = self.pathways[pathway_key]

        if not pathway.active:
            return torch.zeros_like(signal)

        # Compute routing weight (should this signal be transmitted?)
        combined_state = torch.cat([source_state, target_state], dim=-1)
        routing_weight = self.router(combined_state)

        # Modulate signal strength based on states
        modulation = self.modulator(combined_state)

        # Apply pathway strength and modulation
        transmitted = signal * pathway.pathway_strength * routing_weight * modulation

        return transmitted

    def get_active_pathways(self) -> list[tuple[str, str]]:
        """Get list of active pathways."""
        return [(k[0], k[1]) for k, v in self.pathways.items() if v.active]


class DynamicModelLoader:
    """
    Dynamic loader for model sections.

    Loads/unloads sections based on:
    - Current task requirements
    - Memory constraints
    - Activation patterns
    """

    def __init__(
        self,
        storage_path: str = "./model_sections",
        max_loaded_sections: int = 5,
        memory_limit_mb: float = 1024.0,
    ) -> None:
        """Initialize dynamic loader with storage and memory constraints."""
        self.storage_path = storage_path
        self.max_loaded_sections = max_loaded_sections
        self.memory_limit_mb = memory_limit_mb

        os.makedirs(storage_path, exist_ok=True)

        # Currently loaded sections
        self.loaded_sections: dict[str, ModelSection] = {}

        # Section metadata
        self.all_sections_metadata: dict[str, ModelSectionMetadata] = {}

        # Load history for intelligent prediction
        self.load_history: list[str] = []

    def register_section(self, section: ModelSection) -> None:
        """Register a section with the loader."""
        self.all_sections_metadata[section.section_id] = section.metadata

        # Save to disk
        section_path = os.path.join(self.storage_path, f"{section.section_id}.pt")
        section.save_to_disk(section_path)

    def load_section(self, section_id: str) -> ModelSection:
        """Load a section into memory."""
        if section_id in self.loaded_sections:
            # Already loaded
            self.loaded_sections[section_id].metadata.activation_frequency += 1
            return self.loaded_sections[section_id]

        # Check if we need to unload something
        if len(self.loaded_sections) >= self.max_loaded_sections:
            self._unload_least_important()

        # Load from disk
        section_path = os.path.join(self.storage_path, f"{section_id}.pt")
        metadata = self.all_sections_metadata[section_id]

        section = ModelSection(
            section_id=metadata.section_id,
            region_type=metadata.region_type,
            input_dim=metadata.input_dim,
            hidden_dim=512,  # Default
            output_dim=metadata.output_dim,
        )
        section.load_from_disk(section_path)

        self.loaded_sections[section_id] = section
        section.metadata.loaded = True
        section.metadata.activation_frequency += 1

        self.load_history.append(section_id)

        return section

    def unload_section(self, section_id: str) -> None:
        """Unload a section from memory."""
        if section_id in self.loaded_sections:
            section = self.loaded_sections[section_id]

            # Save current state to disk
            section_path = os.path.join(self.storage_path, f"{section_id}.pt")
            section.save_to_disk(section_path)

            # Remove from memory
            del self.loaded_sections[section_id]
            section.metadata.loaded = False

    def _unload_least_important(self) -> None:
        """Unload the least important section based on heuristics."""
        if not self.loaded_sections:
            return

        # Score sections by importance
        scores = {}
        for section_id, section in self.loaded_sections.items():
            # Lower score = less important
            score = (
                section.metadata.activation_frequency * 0.5  # Frequency
                + section.metadata.importance_score * 0.5  # Base importance
            )
            scores[section_id] = score

        # Unload lowest scoring section
        least_important = min(scores.keys(), key=lambda k: scores[k])
        self.unload_section(least_important)

    def predict_needed_sections(self, current_sections: list[str]) -> list[str]:
        """
        Predict which sections will be needed next.
        Based on recent history and common patterns.
        """
        # Simple prediction: look at what typically follows current sections
        predictions = set()

        # Look at recent history
        for i, section_id in enumerate(self.load_history[-20:]):
            if section_id in current_sections and i < len(self.load_history) - 1:
                # What came after this section?
                predictions.add(self.load_history[i + 1])

        return list(predictions)


class SectionedBrainModel(nn.Module):
    """
    Complete sectioned model with brain-inspired organization.

    Features:
    - Multiple specialized sections (brain regions)
    - mHC interconnect (white matter)
    - Dynamic loading based on task
    - Automatic scaling with parameters
    - Cross-section inference
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize sectioned brain model with mHC interconnect and loader."""
        super(SectionedBrainModel, self).__init__()

        self.config = config
        self.embed_dim = config.get("embed_dim", 512)

        # Calculate total parameters and scale memory/timescales
        self.total_params = config.get("total_params", 1e6)
        self.memory_scale = self._compute_memory_scale()
        self.timescale = self._compute_timescale()

        # Create mHC interconnect (white matter)
        self.mhc_interconnect = mHCInterconnect(embed_dim=self.embed_dim)

        # Dynamic loader
        max_loaded = config.get("max_loaded_sections", 5)
        self.loader = DynamicModelLoader(max_loaded_sections=max_loaded, memory_limit_mb=1024.0)

        # Section registry
        self.sections: dict[str, str] = {}  # section_id -> region_type

        # Currently active sections
        self.active_sections: set[str] = set()

    def _compute_memory_scale(self) -> int:
        """
        Scale memory capacity with total parameters.

        Larger models can maintain more memories.
        Following power law: memory ∝ params^0.5
        """
        base_memory = 1000
        scale_factor = math.sqrt(self.total_params / 1e6)
        return int(base_memory * scale_factor)

    def _compute_timescale(self) -> float:
        """
        Scale temporal processing with parameters.

        Larger models process over longer timescales.
        """
        base_timescale = 10.0  # frames/cycles
        scale_factor = math.log10(self.total_params / 1e6 + 1)
        return base_timescale * (1 + scale_factor)

    def add_section(
        self,
        region_type: BrainRegionType,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        section_id: str | None = None,
    ) -> str:
        """Add a specialized section to the model."""
        if section_id is None:
            section_id = f"{region_type.value}_{len(self.sections)}"

        # Create section
        section = ModelSection(
            section_id=section_id,
            region_type=region_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        # Register with loader
        self.loader.register_section(section)
        self.sections[section_id] = region_type.value

        return section_id

    def connect_sections(self, source_id: str, target_id: str, strength: float = 1.0) -> Any:
        """Create mHC pathway between sections."""
        self.mhc_interconnect.register_pathway(source_id, target_id, strength=strength)

    def forward(
        self, input_data: dict[str, torch.Tensor], required_sections: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through sectioned model.

        Args:
            input_data: Dict mapping section_ids to input tensors
            required_sections: Sections needed for this inference

        Returns:
            Dict mapping section_ids to output tensors
        """
        if required_sections is None:
            required_sections = list(input_data.keys())

        # Predict additional sections that might be needed
        predicted = self.loader.predict_needed_sections(required_sections)
        all_needed = set(required_sections + predicted)

        # Load required sections
        loaded_sections = {}
        for section_id in all_needed:
            if section_id in self.sections:
                loaded_sections[section_id] = self.loader.load_section(section_id)

        self.active_sections = set(loaded_sections.keys())

        # Process each section
        outputs = {}
        section_states = {}

        # First pass: process primary inputs
        for section_id, section in loaded_sections.items():
            if section_id in input_data:
                output = section(input_data[section_id])
                outputs[section_id] = output
                section_states[section_id] = section.get_state()

        # Second pass: cross-section communication via mHC
        for source_id, target_id in self.mhc_interconnect.get_active_pathways():
            if source_id in outputs and target_id in loaded_sections:
                # Transmit signal via mHC pathway
                source_state = section_states.get(source_id, outputs[source_id][:1])
                target_state = section_states.get(target_id, torch.zeros(1, self.embed_dim))

                transmitted_signal = self.mhc_interconnect.transmit(
                    source_id, target_id, outputs[source_id], source_state, target_state
                )

                # Process in target section with cross-section input
                target_section = loaded_sections[target_id]
                if target_id not in input_data:
                    # Use transmitted signal as input
                    cross_output = target_section(transmitted_signal, external_input=None)
                else:
                    # Integrate with existing input
                    cross_output = target_section(
                        input_data[target_id], external_input=transmitted_signal
                    )

                outputs[target_id] = cross_output
                section_states[target_id] = target_section.get_state()

        return outputs

    def get_scaling_info(self) -> dict[str, Any]:
        """Get information about automatic scaling."""
        return {
            "total_parameters": self.total_params,
            "memory_capacity": self.memory_scale,
            "timescale": self.timescale,
            "num_sections": len(self.sections),
            "active_sections": len(self.active_sections),
            "max_loaded_sections": self.loader.max_loaded_sections,
        }


if __name__ == "__main__":
    print("=" * 70)
    print("DYNAMIC MODEL SECTIONING WITH mHC INTERCONNECT")
    print("=" * 70)

    # Create sectioned brain model
    config = {
        "embed_dim": 512,
        "total_params": 10e6,  # 10M parameters
        "max_loaded_sections": 3,
    }

    model = SectionedBrainModel(config)

    # Add specialized sections (brain regions)
    print("\n[Step 1] Creating specialized sections:")
    visual_id = model.add_section(BrainRegionType.VISUAL_CORTEX, 512, 512, 512)
    language_id = model.add_section(BrainRegionType.LANGUAGE_CORTEX, 512, 512, 512)
    prefrontal_id = model.add_section(BrainRegionType.PREFRONTAL_CORTEX, 512, 512, 512)
    hippocampus_id = model.add_section(BrainRegionType.HIPPOCAMPUS, 512, 512, 512)

    print("  Created 4 sections: visual, language, prefrontal, hippocampus")

    # Connect sections via mHC pathways (white matter)
    print("\n[Step 2] Creating mHC pathways (white matter):")
    model.connect_sections(visual_id, prefrontal_id, strength=0.8)
    model.connect_sections(language_id, prefrontal_id, strength=0.9)
    model.connect_sections(prefrontal_id, hippocampus_id, strength=1.0)
    print("  Created 3 pathways with varying strengths")

    # Test dynamic inference
    print("\n[Step 3] Dynamic inference with selective loading:")
    input_data = {visual_id: torch.randn(2, 512), language_id: torch.randn(2, 512)}

    outputs = model.forward(input_data, required_sections=[visual_id, language_id])
    print(f"  Processed {len(outputs)} sections")
    print(f"  Active sections: {len(model.active_sections)}")

    # Show scaling
    print("\n[Step 4] Automatic scaling with parameters:")
    scaling_info = model.get_scaling_info()
    print(f"  Total parameters: {scaling_info['total_parameters'] / 1e6:.1f}M")
    print(f"  Memory capacity: {scaling_info['memory_capacity']}")
    print(f"  Timescale: {scaling_info['timescale']:.1f} cycles")
    print(f"  Sections: {scaling_info['num_sections']}")
    print(f"  Max loaded: {scaling_info['max_loaded_sections']}")

    print("\n" + "=" * 70)
    print("SECTIONED BRAIN MODEL READY")
    print("=" * 70)
