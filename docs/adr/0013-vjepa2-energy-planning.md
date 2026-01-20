# ADR-0013: V-JEPA 2 Energy-Based Planning Integration

## Status

**PROPOSED** - 2026-01-19

## Context

The research document "Latent-Space-Native AI Systems: A Technical Deep Dive" highlights
V-JEPA 2 (Meta AI, June 2025) as the state-of-the-art for latent-space reasoning and planning.
CogSynDelta already has VL-JEPA extension but lacks energy-based planning capabilities.

### Research Findings

**V-JEPA 2 Key Features:**
- Vision Transformer encoder with 3D Rotary Position Embeddings
- Trained on 1M+ hours of unlabeled video
- **Zero-shot robot planning** via L1 distance between predicted and goal embeddings
- 77.3% top-1 accuracy on Something-Something v2
- V-JEPA 2-AC: Action-conditioned variant for embodied AI

**VL-JEPA Features:**
- Predicts continuous text embeddings (not tokens)
- 2.85x inference speedup via selective decoding
- 50% fewer trainable parameters than equivalent VLMs

### Current State

CogSynDelta's `vl_jepa_extension.py` provides:
- VisionEncoder with patch embeddings and transformer
- TemporalMemoryBank for sequence context
- JointEmbeddingSpace for vision-language alignment
- HierarchicalPredictiveCoding for multi-scale processing

**Missing:**
- Energy-based planning objective
- Goal-conditioned prediction
- Action conditioning for embodied scenarios

## Decision

Extend VL-JEPA with V-JEPA 2-style energy-based planning:

### Phase 1: Energy-Based Planning Module

```python
# src/cogsyndelta/core/energy_planning.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyBasedPlanner(nn.Module):
    """Energy-based planning using L1 distance in latent space.
    
    Following V-JEPA 2-AC: The energy function is the L1 distance
    between current state embedding and goal embedding. Planning
    involves finding action sequences that minimize this energy.
    
    Why L1 over L2: V-JEPA 2 found L1 more robust to outliers
    in high-dimensional embedding spaces.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 64,
        planning_horizon: int = 10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.planning_horizon = planning_horizon
        
        # Action encoder (maps action tokens to embeddings)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # Predictor: given (state, action) -> next_state
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def compute_energy(
        self,
        state_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute planning energy as L1 distance to goal.
        
        Args:
            state_embedding: Current state [batch, embed_dim]
            goal_embedding: Goal state [batch, embed_dim]
            
        Returns:
            Energy scalar [batch] - lower is closer to goal
        """
        return F.l1_loss(state_embedding, goal_embedding, reduction='none').sum(dim=-1)
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state given current state and action.
        
        Args:
            state: Current state embedding [batch, embed_dim]
            action: Action embedding [batch, action_dim]
            
        Returns:
            Predicted next state [batch, embed_dim]
        """
        action_embed = self.action_encoder(action)
        combined = torch.cat([state, action_embed], dim=-1)
        return self.predictor(combined)
    
    def plan(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        action_candidates: torch.Tensor,
        num_rollouts: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan action sequence to reach goal via energy minimization.
        
        Uses CEM (Cross-Entropy Method) for action optimization.
        
        Args:
            current_state: Starting state [embed_dim]
            goal_state: Target state [embed_dim]
            action_candidates: Candidate actions [num_actions, action_dim]
            num_rollouts: Number of random rollouts to evaluate
            
        Returns:
            (best_action_sequence, trajectory_energies)
        """
        # Initialize random action sequences
        batch_states = current_state.unsqueeze(0).expand(num_rollouts, -1)
        
        best_energy = float('inf')
        best_actions = None
        
        # Simple planning: greedy single-step for now
        # TODO: Implement full CEM planning
        for action in action_candidates:
            action_batch = action.unsqueeze(0).expand(num_rollouts, -1)
            next_states = self.predict_next_state(batch_states, action_batch)
            energies = self.compute_energy(next_states, goal_state.unsqueeze(0))
            mean_energy = energies.mean().item()
            
            if mean_energy < best_energy:
                best_energy = mean_energy
                best_actions = action
        
        return best_actions, torch.tensor([best_energy])
```

### Phase 2: Integrate with VL-JEPA Extension

```python
# Update vl_jepa_extension.py

class VLJEPAWithPlanning(nn.Module):
    """VL-JEPA extended with energy-based planning.
    
    Combines:
    - Vision encoding (existing)
    - Language embedding (existing)
    - Joint embedding space (existing)
    - Energy-based goal-conditioned planning (new)
    """
    
    def __init__(self, base_model: VLJEPAExtension, action_dim: int = 64):
        super().__init__()
        self.base = base_model
        self.planner = EnergyBasedPlanner(
            embed_dim=base_model.embed_dim,
            action_dim=action_dim,
        )
    
    def plan_to_goal(
        self,
        current_image: torch.Tensor,
        goal_description: str,
        action_space: torch.Tensor,
    ) -> torch.Tensor:
        """Plan actions to achieve goal described in language.
        
        This is the key V-JEPA 2 capability: zero-shot planning
        using the joint embedding space as the planning manifold.
        """
        # Encode current state
        current_embed = self.base.vision_encoder(current_image)
        
        # Encode goal (from language)
        goal_embed = self.base.encode_text_goal(goal_description)
        
        # Plan in embedding space
        best_action, _ = self.planner.plan(
            current_embed, goal_embed, action_space
        )
        
        return best_action
```

## Consequences

### Positive

- **Zero-shot planning**: Plan actions from language descriptions without training
- **Latent-space reasoning**: All planning happens in learned embedding space
- **Embodied AI ready**: Foundation for robotics/agent applications
- **Aligned with SOTA**: Following V-JEPA 2's proven approach

### Negative

- **Requires action space definition**: Need to define action vocabulary
- **Planning compute cost**: CEM requires many forward passes
- **No temporal video**: Current VL-JEPA lacks video training

### Neutral

- Trade-off between planning horizon and compute cost
- May require fine-tuning predictor on domain-specific transitions

## Implementation Plan

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Energy Planner | Week 1-2 | energy_planning.py, tests |
| 2. VL-JEPA Integration | Week 3-4 | VLJEPAWithPlanning class |
| 3. Action Vocabulary | Week 5 | Domain-specific action encodings |

## References

- V-JEPA 2: https://github.com/facebookresearch/vjepa2 (MIT license)
- Assran et al. "V-JEPA 2: Self-Supervised Video Models" (2025)
- Research doc: docs/Latent-Space-Native AI Systems_A Technical Deep Dive for CogSynDelta.md

---

*Authored: 2026-01-19 | Related: ADR-0002, vl_jepa_extension.py*
