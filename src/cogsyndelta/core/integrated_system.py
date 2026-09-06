"""
Integrated Self-Improving AI System

Combines all components:
- PCN-VAE-GAN hybrid base architecture
- VL-JEPA vision-language joint embedding with mHC
- Self-improving agent framework for SWE/AIE/SWD/AID
- Silent semantic state retention (no token burning)
- Security hardening and quality assurance
- Persistent memory with compression and temporal continuity
- Safeguards against infinite loops and hazards

Status: SUPERSEDED (DEC-12, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md ~L1026), as part
of the dormant `core/` stack — imported by no training path. Its interconnect role is
harvested into the interconnect module specified in
docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md section 2 ("White matter — the interconnect
module specification", DEC-16), which is planned but not yet implemented. This module and its
tests are kept in place for the record; do not import it from new code.
"""

from typing import Any

import torch
from torch import nn

from cogsyndelta.agents.self_improving_agents import (
    SelfImprovingAgentFramework,
)
from cogsyndelta.core.pcn_vae_gan import PCNVAEGANHybrid, load_config
from cogsyndelta.core.vl_jepa_extension import (
    FrameBufferAdapter,
    HierarchicalPredictiveCoding,
    JointEmbeddingSpace,
    TemporalMemoryBank,
    VisionEncoder,
)
from cogsyndelta.memory.memory_persistence import (
    DEFAULT_GPU_TIMEOUT_S,
    DEFAULT_MAX_OUTPUT_BYTES,
    LAB_GPU_MAX_PROCESS_BYTES,
    LAB_GPU_MIN_PROCESS_BYTES,
    InfiniteLoopSafeguard,
    PersistentMemoryBank,
)


class IntegratedSelfImprovingSystem(nn.Module):
    """
    Complete integrated system combining:
    1. PCN-VAE-GAN for base self-improvement
    2. VL-JEPA for vision-language understanding without token generation
    3. mHC for moderated layer connections
    4. Self-improving agents for multi-language/framework exploration
    5. Security hardening and QA for production-ready outputs
    6. Persistent memory with compression and temporal continuity
    7. Safeguards against infinite loops and ethical hazards
    """

    # Type hints for instance attributes with multiple possible types
    memory_bank: PersistentMemoryBank | TemporalMemoryBank

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize integrated system with all components from config."""
        super(IntegratedSelfImprovingSystem, self).__init__()

        # Load configuration
        self.config = load_config(config_path)

        # Base PCN-VAE-GAN architecture
        self.base_model = PCNVAEGANHybrid(self.config)

        # VL-JEPA components for silent semantic processing
        vl_config = self.config.get("vision_language", {})
        embed_dim = vl_config.get("embed_dim", 512)

        self.vision_encoder = VisionEncoder(
            image_size=vl_config.get("image_size", 224),
            patch_size=vl_config.get("patch_size", 16),
            embed_dim=embed_dim,
            num_layers=vl_config.get("num_vision_layers", 6),
        )

        # Replace TemporalMemoryBank with PersistentMemoryBank
        persist_config = self.config.get("memory_persistence", {})
        if persist_config.get("enabled", True):
            self.memory_bank = PersistentMemoryBank(
                embed_dim=embed_dim,
                working_capacity=persist_config.get("working_memory", {}).get("capacity", 100),
                short_term_capacity=persist_config.get("short_term_memory", {}).get(
                    "capacity", 1000
                ),
                storage_path=persist_config.get("long_term_memory", {}).get(
                    "storage_path", "./memory_storage"
                ),
            )
        else:
            # Fallback to original memory bank
            self.memory_bank = TemporalMemoryBank(
                memory_size=vl_config.get("memory_size", 1000),
                embed_dim=embed_dim,
                num_read_heads=vl_config.get("num_read_heads", 4),
            )

        # Safeguards
        safeguard_config = self.config.get("safeguards", {})
        if safeguard_config.get("enabled", True):
            loop_config = safeguard_config.get("loop_detection", {})
            timeout_config = safeguard_config.get("timeouts", {})
            limits = safeguard_config.get("resource_limits", {})
            self.safeguard = InfiniteLoopSafeguard(
                max_iterations=loop_config.get("max_iterations", 1000),
                max_repetitions=loop_config.get("max_repetitions", 5),
                timeout_seconds=timeout_config.get("max_execution_time", DEFAULT_GPU_TIMEOUT_S),
                max_output_size=int(limits.get("max_output_size", DEFAULT_MAX_OUTPUT_BYTES)),
                max_memory_usage=int(limits.get("max_memory_usage", LAB_GPU_MIN_PROCESS_BYTES)),
                max_memory_usage_ceiling=int(
                    limits.get("max_memory_usage_ceiling", LAB_GPU_MAX_PROCESS_BYTES)
                ),
            )
        else:
            self.safeguard = None

        self.hierarchical_pcn = HierarchicalPredictiveCoding(
            embed_dim=embed_dim, num_levels=vl_config.get("num_hierarchical_levels", 3)
        )

        self.joint_space = JointEmbeddingSpace(
            embed_dim=embed_dim, latent_dim=self.config["exploratory"]["latent_dim"]
        )

        self.frame_adapter = FrameBufferAdapter(
            buffer_size=vl_config.get("frame_buffer_size", 16),
            target_size=vl_config.get("image_size", 224),
        )

        # Self-improving agent framework
        agent_config = self.config.get("agent_framework", {})
        if agent_config.get("enabled", True):
            self.agent_framework = SelfImprovingAgentFramework(
                {
                    "embed_dim": embed_dim,
                    "num_improvement_iterations": agent_config.get("num_improvement_iterations", 3),
                    "security_enabled": agent_config.get("security_hardening", {}).get(
                        "enabled", True
                    ),
                    "qa_enabled": agent_config.get("quality_assurance", {}).get("enabled", True),
                }
            )
        else:
            self.agent_framework = None

        # Cross-modal bridge (VAE latent to VL-JEPA embedding)
        self.latent_to_semantic = nn.Sequential(
            nn.Linear(self.config["exploratory"]["latent_dim"], embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Semantic to latent bridge
        self.semantic_to_latent = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.config["exploratory"]["latent_dim"]),
        )

    def process_visual_input(self, frames: torch.Tensor) -> dict[str, Any]:
        """
        Process visual input with silent semantic state retention.
        No token generation - pure embedding prediction.

        Includes safeguards against infinite loops.

        Args:
            frames: Video frames [batch, channels, height, width] or
                   [batch, sequence, channels, height, width]

        Returns:
            Dictionary with semantic states and predictions
        """
        # Safeguard check
        if self.safeguard:
            is_safe, message = self.safeguard.check_state(frames.flatten()[:512])
            if not is_safe:
                self.safeguard.trigger_circuit_breaker(message)
                raise RuntimeError(f"Safeguard triggered: {message}")

        # Handle single frame or sequence
        if frames.dim() == 4:
            # Single frame
            _batch_size = frames.size(0)  # kept for future batched processing

            # Encode to semantic embedding (silent state)
            semantic_embed = self.vision_encoder(frames)

            # Store in persistent memory bank
            if isinstance(self.memory_bank, PersistentMemoryBank):
                self.memory_bank.write(semantic_embed, importance=1.0, source="visual_input")
            else:
                self.memory_bank.write(semantic_embed)

            # Hierarchical predictive coding with mHC
            hpc_output = self.hierarchical_pcn(semantic_embed)

            return {
                "semantic_state": semantic_embed,
                "hierarchical_representations": hpc_output["representations"],
                "predictions": hpc_output["predictions"],
                "prediction_errors": hpc_output["prediction_errors"],
                "moderated_states": hpc_output["moderated_states"],
            }

        if frames.dim() == 5:
            # Sequence of frames - process with temporal grounding
            batch_size, seq_len, C, H, W = frames.shape

            semantic_states = []
            for t in range(seq_len):
                # Safeguard check per frame
                if self.safeguard:
                    is_safe, message = self.safeguard.check_state(frames[:, t].flatten()[:512])
                    if not is_safe:
                        self.safeguard.trigger_circuit_breaker(message)
                        break

                frame_t = frames[:, t]
                semantic_t = self.vision_encoder(frame_t)
                semantic_states.append(semantic_t)
                if isinstance(self.memory_bank, PersistentMemoryBank):
                    self.memory_bank.write(
                        semantic_t, importance=1.0, source=f"visual_sequence_{t}"
                    )
                else:
                    self.memory_bank.write(semantic_t)

            semantic_sequence = torch.stack(semantic_states, dim=1)

            # Get temporal context from persistent memory
            if isinstance(self.memory_bank, TemporalMemoryBank):
                temporal_context = self.memory_bank.temporal_context(
                    window_size=self.config["vision_language"].get("temporal_window", 10)
                )
            else:
                # Fallback for PersistentMemoryBank
                query = semantic_states[-1]
                temporal_context, _ = self.memory_bank.read(query, num_reads=10)

            # Process last frame with temporal context
            hpc_output = self.hierarchical_pcn(
                semantic_states[-1], temporal_context=temporal_context
            )

            return {
                "semantic_sequence": semantic_sequence,
                "semantic_state": semantic_states[-1],
                "temporal_context": temporal_context,
                "hierarchical_output": hpc_output,
            }

        raise ValueError(f"Expected frames with 4 or 5 dimensions, got {frames.dim()}")

    def self_improve_solution(
        self,
        code_tokens: torch.Tensor,
        language: str = "python",
        visual_context: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Self-improve code solution with visual context.

        Args:
            code_tokens: Code token IDs [batch, seq_len]
            language: Programming language
            visual_context: Optional visual context frames

        Returns:
            Improved solution with quality and security metrics
        """
        if self.agent_framework is None:
            raise ValueError("Agent framework not enabled in config")

        # Process visual context if provided
        if visual_context is not None:
            visual_output = self.process_visual_input(visual_context)
            semantic_context = visual_output["semantic_state"]
        else:
            semantic_context = None

        # Run self-improvement cycle
        result = self.agent_framework.improve_solution(
            code_tokens,
            language=language,
            num_iterations=self.config["agent_framework"].get("num_improvement_iterations", 3),
        )

        # Integrate with base VAE model for additional exploration
        if semantic_context is not None:
            # Convert semantic context to VAE latent space
            latent_context = self.semantic_to_latent(semantic_context)

            # Run exploratory phase with context
            exploratory_samples = self.base_model.exploratory_phase(
                latent_context.view(-1, self.config["exploratory"]["latent_dim"] * 28 * 28),
                k=self.config["exploratory"]["k"],
            )

            result["exploratory_samples"] = exploratory_samples

        return result

    def explore_multimodal_solution(
        self,
        problem_description: torch.Tensor,
        visual_examples: torch.Tensor | None = None,
        languages: list[str] = None,
    ) -> dict[str, Any]:
        """
        Explore solutions across languages with multimodal understanding.

        Args:
            problem_description: Problem description tokens
            visual_examples: Optional visual examples/diagrams
            languages: Target languages to explore

        Returns:
            Best solutions across languages with quality metrics
        """
        if languages is None:
            languages = self.config["agent_framework"].get("supported_languages", ["python"])

        # Process visual examples into semantic state
        if visual_examples is not None:
            visual_output = self.process_visual_input(visual_examples)
            semantic_visual = visual_output["semantic_state"]
        else:
            semantic_visual = None

        # Explore implementations
        multi_lang_result = self.agent_framework.explore_multi_language(
            problem_description, languages=languages
        )

        # Enhance with visual context if available
        if semantic_visual is not None:
            for lang, impl in multi_lang_result["all_results"].items():
                # Align code and visual semantics
                code_semantic = impl["implementation"]
                alignment = self.joint_space(semantic_visual, code_semantic.unsqueeze(-1))
                impl["visual_alignment"] = alignment["similarity"].mean().item()

        return multi_lang_result

    def continuous_learning_cycle(
        self,
        video_stream: torch.Tensor,
        code_stream: torch.Tensor | None = None,
        num_cycles: int = 10,
    ) -> dict[str, Any]:
        """
        Continuous learning from visual and code streams.
        Silent semantic state retention enables efficient long-term learning.

        Args:
            video_stream: Stream of video frames [num_frames, C, H, W]
            code_stream: Optional stream of code examples
            num_cycles: Number of learning cycles

        Returns:
            Learning statistics and final state
        """
        num_frames = video_stream.size(0)
        frames_per_cycle = max(1, num_frames // num_cycles)

        learning_history: dict[str, list[Any]] = {
            "quality_progression": [],
            "memory_utilization": [],
            "prediction_errors": [],
        }

        for cycle in range(num_cycles):
            start_idx = cycle * frames_per_cycle
            end_idx = min((cycle + 1) * frames_per_cycle, num_frames)

            # Process frame batch
            frames_batch = video_stream[start_idx:end_idx]

            # Add frames to buffer
            for frame in frames_batch:
                self.frame_adapter.add_frame(frame)

            # Get temporal window
            temporal_window = self.frame_adapter.get_temporal_window()
            if temporal_window is not None:
                # Process with temporal grounding
                visual_output = self.process_visual_input(temporal_window.unsqueeze(0))

                # Compute prediction errors (for self-improvement signal)
                pred_errors = visual_output["hierarchical_output"]["prediction_errors"]
                pred_errors_stacked = (
                    torch.stack(pred_errors) if isinstance(pred_errors, list) else pred_errors
                )
                mean_error = pred_errors_stacked.abs().mean()

                learning_history["prediction_errors"].append(mean_error.item())

            # Track memory utilization
            if hasattr(self.memory_bank, "memory_age") and isinstance(
                self.memory_bank, TemporalMemoryBank
            ):
                memory_age = self.memory_bank.memory_age
                learning_history["memory_utilization"].append(
                    (memory_age < 100).float().mean().item()
                )

        # Get final semantic state
        final_state: Any = None
        if hasattr(self.memory_bank, "temporal_context") and isinstance(
            self.memory_bank, TemporalMemoryBank
        ):
            final_state = self.memory_bank.temporal_context()

        return {
            "learning_history": learning_history,
            "final_semantic_state": final_state,
            "total_cycles": num_cycles,
        }


def create_integrated_system(config_path: str = "config.yaml") -> IntegratedSelfImprovingSystem:
    """
    Create fully integrated self-improving AI system.

    Args:
        config_path: Path to configuration file

    Returns:
        Integrated system ready for deployment
    """
    return IntegratedSelfImprovingSystem(config_path)


if __name__ == "__main__":
    print("=" * 70)
    print("INTEGRATED SELF-IMPROVING AI SYSTEM")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✓ PCN-VAE-GAN base architecture")
    print("  ✓ VL-JEPA vision-language joint embedding")
    print("  ✓ mHC moderated hyper connections")
    print("  ✓ Silent semantic state retention (no token burning)")
    print("  ✓ Temporal grounding with memory bank")
    print("  ✓ Self-improving agents (SWE/AIE/SWD/AID)")
    print("  ✓ Multi-language/framework exploration")
    print("  ✓ Security hardening & QA")
    print()
    print("=" * 70)

    # Create system
    system = create_integrated_system()
    print("✓ System initialized")

    # Demo 1: Visual processing with silent semantic states
    print("\n[Demo 1] Visual processing - silent semantic state retention:")
    frames = torch.randn(2, 3, 224, 224)
    visual_result = system.process_visual_input(frames)
    print(f"  Input: {frames.shape}")
    print(f"  Semantic state: {visual_result['semantic_state'].shape}")
    print(f"  Hierarchical levels: {len(visual_result['hierarchical_representations'])}")
    print("  No tokens generated - pure embedding prediction ✓")

    # Demo 2: Self-improving code solution
    print("\n[Demo 2] Self-improving code solution:")
    code_tokens = torch.randint(0, 1000, (1, 100))
    improvement_result = system.self_improve_solution(code_tokens, language="python")
    print(f"  Iterations: {len(improvement_result['history']['iterations'])}")
    print(
        f"  Quality progression: {[f'{q:.3f}' for q in improvement_result['history']['quality_scores']]}"
    )
    print(
        f"  Security scores: {[f'{s:.3f}' for s in improvement_result['history']['security_scores']]}"
    )

    # Demo 3: Multi-language exploration
    print("\n[Demo 3] Multi-language exploration:")
    problem = torch.randint(0, 1000, (1, 50))
    multi_result = system.explore_multimodal_solution(problem, languages=["python", "rust", "go"])
    print(f"  Best language: {multi_result['best_language']}")
    print(f"  Explored: {list(multi_result['all_results'].keys())}")

    print("\n" + "=" * 70)
    print("SYSTEM READY FOR DEPLOYMENT")
    print("=" * 70)
