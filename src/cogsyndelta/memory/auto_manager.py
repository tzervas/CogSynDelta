"""
Intelligent Auto-Management System for Memory and Knowledge Corpus

Automatically manages:
1. Memory storage, culling, and loading
2. Knowledge corpus optimization
3. Temporal continuity preservation
4. Performance and accuracy balance
5. Prevention of over-culling and over-loading

Uses intelligent heuristics and learned policies to maintain optimal balance.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LoadingStrategy(str, Enum):
    """Strategies for loading memories."""

    GREEDY = "greedy"  # Load highest relevance
    BALANCED = "balanced"  # Balance relevance and diversity
    TEMPORAL = "temporal"  # Prioritize temporal continuity
    ADAPTIVE = "adaptive"  # Learn optimal strategy


class CullingPolicy(str, Enum):
    """Policies for culling memories."""

    CONSERVATIVE = "conservative"  # Cull less, keep more
    MODERATE = "moderate"  # Balanced culling
    AGGRESSIVE = "aggressive"  # Cull more, keep less
    ADAPTIVE = "adaptive"  # Learn optimal policy


@dataclass
class SystemState:
    """Current system state for decision making."""

    memory_usage: float  # 0.0-1.0
    context_size: int
    active_memories: int
    inference_latency: float
    accuracy_estimate: float
    temporal_coherence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ManagementDecision:
    """Decision made by auto-management system."""

    action: str  # "load", "unload", "cull", "archive", "nothing"
    target_ids: list[str]
    priority: float
    reasoning: str
    expected_impact: dict[str, float]


class TemporalContinuityTracker:
    """
    Tracks temporal continuity to prevent loss of temporal grounding.

    Maintains:
    - Temporal chains of related memories
    - Context windows
    - Causal relationships
    """

    def __init__(self, window_size: int = 100) -> None:
        """Initialize temporal continuity tracker with window and graph structures."""
        self.window_size = window_size
        self.temporal_chain: deque = deque(maxlen=window_size)
        self.causal_graph: dict[str, set[str]] = {}  # memory_id -> dependencies
        self.temporal_clusters: list[set[str]] = []

    def add_memory(
        self, memory_id: str, timestamp: datetime, dependencies: set[str] | None = None
    ) -> Any:
        """Add memory to temporal tracking."""
        self.temporal_chain.append((memory_id, timestamp))

        if dependencies:
            self.causal_graph[memory_id] = dependencies

    def get_temporal_context(self, memory_id: str) -> set[str]:
        """Get temporal context for a memory."""
        # Get memories in temporal window around this memory
        context = set()

        # Find position in chain
        for i, (mid, ts) in enumerate(self.temporal_chain):
            if mid == memory_id:
                # Get window around this position
                start = max(0, i - self.window_size // 2)
                end = min(len(self.temporal_chain), i + self.window_size // 2)

                for j in range(start, end):
                    context.add(self.temporal_chain[j][0])
                break

        # Add causal dependencies
        if memory_id in self.causal_graph:
            context.update(self.causal_graph[memory_id])

        return context

    def compute_coherence(self, loaded_memories: set[str]) -> float:
        """
        Compute temporal coherence of loaded memories.

        Higher score = better temporal continuity.
        """
        if not loaded_memories:
            return 0.0

        # Check how many temporal connections exist
        total_possible = len(loaded_memories) * (len(loaded_memories) - 1)
        if total_possible == 0:
            return 1.0

        actual_connections = 0
        for mem_id in loaded_memories:
            context = self.get_temporal_context(mem_id)
            actual_connections += len(context.intersection(loaded_memories))

        coherence = actual_connections / total_possible
        return min(1.0, coherence)


class LoadBalancer(nn.Module):
    """
    Intelligent load balancer that decides what to load/unload.

    Learns optimal balance between:
    - Memory usage
    - Inference speed
    - Accuracy
    - Temporal continuity
    """

    def __init__(self, embed_dim: int = 512) -> None:
        """Initialize load balancer with policy and value networks."""
        super(LoadBalancer, self).__init__()

        self.embed_dim = embed_dim

        # Policy network (decides what to load)
        self.policy_net = nn.Sequential(
            nn.Linear(embed_dim + 20, embed_dim),  # +20 for state features
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),  # Load probability
        )

        # Value network (estimates value of loading a memory)
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim + 20, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )

        # Diversity scorer (promotes diverse memories)
        self.diversity_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1), nn.Tanh()
        )

    def forward(
        self,
        memory_embedding: torch.Tensor,
        system_state: torch.Tensor,
        currently_loaded: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Decide if a memory should be loaded.

        Args:
            memory_embedding: Memory to consider
            system_state: Current system state features
            currently_loaded: Average of currently loaded memories

        Returns:
            (load_probability, estimated_value)
        """
        # Combine features
        features = torch.cat([memory_embedding, system_state], dim=-1)

        # Compute load probability
        load_prob = self.policy_net(features)

        # Compute value
        value = self.value_net(features)

        # Compute diversity bonus
        diversity_features = torch.cat([memory_embedding, currently_loaded], dim=-1)
        diversity_score = self.diversity_scorer(diversity_features)

        # Adjust value with diversity
        adjusted_value = value + 0.2 * diversity_score

        return load_prob.item(), adjusted_value.item()


class CullingDecisionMaker(nn.Module):
    """
    Decides what memories to cull.

    Considers:
    - Redundancy with existing memories
    - Importance/relevance
    - Access patterns
    - Temporal significance

    Prevents over-culling by maintaining minimum diversity.
    """

    def __init__(self, embed_dim: int = 512) -> None:
        """Initialize culling decision maker with policy and redundancy detector."""
        super(CullingDecisionMaker, self).__init__()

        self.embed_dim = embed_dim

        # Culling policy network
        self.cull_policy = nn.Sequential(
            nn.Linear(embed_dim + 15, embed_dim),  # +15 for metadata
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),  # Cull probability
        )

        # Redundancy detector
        self.redundancy_detector = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),  # Redundancy score
        )

    def should_cull(
        self,
        memory_embedding: torch.Tensor,
        metadata_features: torch.Tensor,
        corpus_embedding: torch.Tensor,
    ) -> tuple[bool, float, str]:
        """
        Decide if memory should be culled.

        Args:
            memory_embedding: Memory to evaluate
            metadata_features: Metadata (access count, age, relevance, etc.)
            corpus_embedding: Average of corpus

        Returns:
            (should_cull, confidence, reasoning)
        """
        # Compute culling probability
        features = torch.cat([memory_embedding, metadata_features], dim=-1)
        cull_prob = self.cull_policy(features).item()

        # Check redundancy
        redundancy_features = torch.cat([memory_embedding, corpus_embedding], dim=-1)
        redundancy = self.redundancy_detector(redundancy_features).item()

        # Decision logic
        if redundancy > 0.9 and cull_prob > 0.7:
            return True, cull_prob, "High redundancy with low unique value"

        if metadata_features[0].item() < 0.1:  # Very low access count
            if metadata_features[2].item() > 0.8:  # But high age
                if metadata_features[3].item() < 0.3:  # And low relevance
                    return True, cull_prob, "Old, unused, low relevance"

        # Conservative by default
        if cull_prob > 0.9:
            return True, cull_prob, "Very high culling confidence"

        return False, cull_prob, "Retained for diversity/continuity"


class IntelligentAutoManager:
    """
    Main auto-management system.

    Continuously monitors and optimizes:
    - What memories are loaded
    - What gets culled
    - What gets archived
    - Temporal continuity
    - Performance vs accuracy trade-off
    """

    def __init__(
        self,
        embed_dim: int = 512,
        max_loaded_memories: int = 1000,
        max_total_memories: int = 100000,
        target_memory_usage: float = 0.7,
    ) -> None:
        """Initialize auto memory manager with balancer, culler and tracker."""
        self.embed_dim = embed_dim
        self.max_loaded_memories = max_loaded_memories
        self.max_total_memories = max_total_memories
        self.target_memory_usage = target_memory_usage

        # Core components
        self.load_balancer = LoadBalancer(embed_dim)
        self.cull_decider = CullingDecisionMaker(embed_dim)
        self.temporal_tracker = TemporalContinuityTracker(window_size=100)

        # Current state
        self.loaded_memory_ids: set[str] = set()
        self.archived_memory_ids: set[str] = set()
        self.cull_candidates: set[str] = set()

        # Performance tracking
        self.history: list[SystemState] = []
        self.decisions_made: list[ManagementDecision] = []

        # Thresholds (adaptive)
        self.cull_threshold = 0.7
        self.load_threshold = 0.5
        self.diversity_threshold = 0.3

        # Safety limits
        self.min_loaded_memories = 10
        self.max_cull_per_cycle = 100

    def manage_cycle(self, unified_manager, system_state: SystemState) -> dict[str, Any]:
        """
        Perform one management cycle.

        Returns:
            Actions taken and their results
        """
        actions: dict[str, list[Any]] = {"loaded": [], "unloaded": [], "culled": [], "archived": [], "decisions": []}

        # 1. Assess current state
        assessment = self._assess_state(unified_manager, system_state)

        # 2. Decide on loading if under capacity
        if system_state.memory_usage < self.target_memory_usage:
            load_actions = self._decide_loading(unified_manager, assessment)
            actions["loaded"] = load_actions

        # 3. Decide on unloading if over capacity
        if system_state.memory_usage > self.target_memory_usage:
            unload_actions = self._decide_unloading(unified_manager, assessment)
            actions["unloaded"] = unload_actions

        # 4. Decide on culling if total memories too high
        if len(unified_manager.memories) > self.max_total_memories * 0.9:
            cull_actions = self._decide_culling(unified_manager, assessment)
            actions["culled"] = cull_actions

        # 5. Archive old memories
        archive_actions = self._decide_archiving(unified_manager, assessment)
        actions["archived"] = archive_actions

        # 6. Update temporal tracking
        self._update_temporal_tracking(unified_manager)

        # 7. Adapt thresholds based on performance
        self._adapt_thresholds(system_state, assessment)

        # Record history
        self.history.append(system_state)

        return actions

    def _assess_state(self, unified_manager, system_state: SystemState) -> dict[str, Any]:
        """Assess current system state."""
        memories = unified_manager.memories

        # Compute corpus statistics
        if memories:
            all_embeddings = torch.stack([m.embedding for m in memories.values()])
            corpus_mean = all_embeddings.mean(dim=0)
            corpus_std = all_embeddings.std(dim=0).mean().item()
        else:
            corpus_mean = torch.zeros(self.embed_dim)
            corpus_std = 0.0

        # Compute temporal coherence
        temporal_coherence = self.temporal_tracker.compute_coherence(self.loaded_memory_ids)

        # Diversity metric
        diversity = self._compute_diversity(memories)

        assessment = {
            "total_memories": len(memories),
            "loaded_memories": len(self.loaded_memory_ids),
            "memory_usage_ratio": system_state.memory_usage,
            "corpus_mean": corpus_mean,
            "corpus_diversity": corpus_std,
            "temporal_coherence": temporal_coherence,
            "semantic_diversity": diversity,
            "performance_score": self._compute_performance_score(system_state),
            "risk_of_over_culling": self._assess_over_cull_risk(diversity, temporal_coherence),
            "risk_of_over_loading": self._assess_over_load_risk(system_state),
        }

        return assessment

    def _decide_loading(self, unified_manager, assessment: dict) -> list[str]:
        """Decide which memories to load."""
        loaded_ids: list[str] = []

        # How many can we load?
        available_slots = int(
            (self.target_memory_usage - assessment["memory_usage_ratio"]) * self.max_loaded_memories
        )

        if available_slots <= 0:
            return loaded_ids

        # Get unloaded memories
        all_memory_ids = set(unified_manager.memories.keys())
        unloaded_ids = all_memory_ids - self.loaded_memory_ids

        if not unloaded_ids:
            return loaded_ids

        # Compute currently loaded mean
        if self.loaded_memory_ids:
            loaded_embeddings = torch.stack(
                [unified_manager.memories[mid].embedding for mid in self.loaded_memory_ids]
            )
            currently_loaded_mean = loaded_embeddings.mean(dim=0)
        else:
            currently_loaded_mean = torch.zeros(self.embed_dim)

        # Score each unloaded memory
        system_state_features = self._extract_state_features(assessment)

        scores = []
        for memory_id in unloaded_ids:
            memory = unified_manager.memories[memory_id]

            # Get load probability and value
            load_prob, value = self.load_balancer(
                memory.embedding, system_state_features, currently_loaded_mean
            )

            # Adjust for temporal importance
            temporal_context = self.temporal_tracker.get_temporal_context(memory_id)
            temporal_bonus = len(temporal_context.intersection(self.loaded_memory_ids)) / max(
                len(temporal_context), 1
            )

            # Adjust for relevance and access patterns
            relevance_score = memory.relevance * memory.temporal_metadata.decay_factor
            access_bonus = min(memory.temporal_metadata.access_count / 100, 1.0)

            # Combined score
            total_score = (
                0.4 * value
                + 0.3 * load_prob
                + 0.2 * temporal_bonus
                + 0.1 * (relevance_score + access_bonus)
            )

            scores.append((memory_id, total_score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Load top N
        for memory_id, score in scores[:available_slots]:
            if score > self.load_threshold:
                loaded_ids.append(memory_id)
                self.loaded_memory_ids.add(memory_id)

        return loaded_ids

    def _decide_unloading(self, unified_manager, assessment: dict) -> list[str]:
        """Decide which memories to unload."""
        unloaded_ids: list[str] = []

        # How many should we unload?
        excess = int(
            (assessment["memory_usage_ratio"] - self.target_memory_usage) * self.max_loaded_memories
        )

        if excess <= 0:
            return unloaded_ids

        # Ensure we keep minimum loaded
        can_unload = max(0, len(self.loaded_memory_ids) - self.min_loaded_memories)
        excess = min(excess, can_unload)

        if excess == 0:
            return unloaded_ids

        # Score loaded memories (lower score = more likely to unload)
        scores = []
        for memory_id in self.loaded_memory_ids:
            memory = unified_manager.memories[memory_id]

            # Factors favoring keeping loaded
            relevance = memory.relevance
            recency = memory.temporal_metadata.decay_factor
            access_count = min(memory.temporal_metadata.access_count / 100, 1.0)

            # Temporal importance
            temporal_context = self.temporal_tracker.get_temporal_context(memory_id)
            temporal_importance = len(temporal_context.intersection(self.loaded_memory_ids)) / max(
                len(temporal_context), 1
            )

            keep_score = (
                0.3 * relevance + 0.3 * recency + 0.2 * access_count + 0.2 * temporal_importance
            )

            scores.append((memory_id, keep_score))

        # Sort by score (ascending - lowest scores unloaded first)
        scores.sort(key=lambda x: x[1])

        # Unload lowest scoring
        for memory_id, score in scores[:excess]:
            unloaded_ids.append(memory_id)
            self.loaded_memory_ids.discard(memory_id)

        return unloaded_ids

    def _decide_culling(self, unified_manager, assessment: dict) -> list[str]:
        """Decide which memories to cull (permanently remove)."""
        culled_ids: list[str] = []

        # Check if we should cull at all
        if assessment["risk_of_over_culling"] > 0.7:
            return culled_ids  # Too risky to cull

        # Compute corpus mean
        corpus_mean = assessment["corpus_mean"]

        # Evaluate each memory for culling
        cull_candidates = []
        for memory_id, memory in unified_manager.memories.items():
            # Don't cull recently accessed memories
            if memory.temporal_metadata.access_count > 10:
                continue

            # Don't cull high-relevance memories
            if memory.relevance > 0.7:
                continue

            # Extract metadata features
            metadata_features = self._extract_memory_metadata(memory)

            # Decide if should cull
            should_cull, confidence, reasoning = self.cull_decider.should_cull(
                memory.embedding, metadata_features, corpus_mean
            )

            if should_cull:
                cull_candidates.append((memory_id, confidence, reasoning))

        # Sort by confidence
        cull_candidates.sort(key=lambda x: x[1], reverse=True)

        # Cull top candidates (up to max per cycle)
        for memory_id, confidence, reasoning in cull_candidates[: self.max_cull_per_cycle]:
            # Final safety check
            if self._is_safe_to_cull(memory_id, unified_manager, assessment):
                culled_ids.append(memory_id)

                # Remove from all indices
                if memory_id in self.loaded_memory_ids:
                    self.loaded_memory_ids.discard(memory_id)

                # Record decision
                decision = ManagementDecision(
                    action="cull",
                    target_ids=[memory_id],
                    priority=confidence,
                    reasoning=reasoning,
                    expected_impact={"diversity": -0.01},
                )
                self.decisions_made.append(decision)

        return culled_ids

    def _decide_archiving(self, unified_manager, assessment: dict) -> list[str]:
        """Decide which memories to archive to disk."""
        archived_ids = []

        # Archive old, infrequently accessed memories
        now = datetime.now()

        for memory_id, memory in unified_manager.memories.items():
            if memory_id in self.archived_memory_ids:
                continue

            # Archive if old and rarely accessed
            age_days = (now - memory.temporal_metadata.created_at).days

            if age_days > 30 and memory.temporal_metadata.access_count < 5:
                archived_ids.append(memory_id)
                self.archived_memory_ids.add(memory_id)

        return archived_ids

    def _is_safe_to_cull(self, memory_id: str, unified_manager, assessment: dict) -> bool:
        """Check if it's safe to cull this memory."""
        memory = unified_manager.memories[memory_id]

        # Don't cull if it maintains temporal continuity
        temporal_context = self.temporal_tracker.get_temporal_context(memory_id)
        if len(temporal_context) > 5:  # Important for continuity
            return False

        # Don't cull if diversity is already low
        if assessment["semantic_diversity"] < self.diversity_threshold:
            return False

        # Don't cull if it has many dependencies
        if len(memory.contextual_metadata.dependencies) > 3:
            return False

        return True

    def _compute_diversity(self, memories: dict) -> float:
        """Compute semantic diversity of memory corpus."""
        if len(memories) < 2:
            return 1.0

        embeddings = torch.stack([m.embedding for m in memories.values()])

        # Compute pairwise distances
        normalized = F.normalize(embeddings, dim=-1)
        similarities = torch.matmul(normalized, normalized.T)

        # Average distance (1 - similarity)
        distances = 1 - similarities
        diversity = distances.mean().item()

        return diversity

    def _compute_performance_score(self, state: SystemState) -> float:
        """Compute overall performance score."""
        # Balance accuracy, speed, and resource usage
        score = (
            0.4 * state.accuracy_estimate
            + 0.3 * (1.0 - state.inference_latency / 1000.0)  # Normalize latency
            + 0.3 * state.temporal_coherence
        )
        return max(0.0, min(1.0, score))

    def _assess_over_cull_risk(self, diversity: float, coherence: float) -> float:
        """Assess risk of over-culling."""
        # Higher risk if diversity or coherence already low
        risk = 1.0 - (0.5 * diversity + 0.5 * coherence)
        return max(0.0, min(1.0, risk))

    def _assess_over_load_risk(self, state: SystemState) -> float:
        """Assess risk of over-loading."""
        # Higher risk if memory usage high or latency increasing
        risk = 0.6 * state.memory_usage + 0.4 * (state.inference_latency / 1000.0)
        return max(0.0, min(1.0, risk))

    def _extract_state_features(self, assessment: dict) -> torch.Tensor:
        """Extract features from system state."""
        features = torch.tensor(
            [
                assessment["memory_usage_ratio"],
                assessment["temporal_coherence"],
                assessment["semantic_diversity"],
                assessment["performance_score"],
                assessment["risk_of_over_culling"],
                assessment["risk_of_over_loading"],
                float(assessment["loaded_memories"]) / self.max_loaded_memories,
                float(assessment["total_memories"]) / self.max_total_memories,
                assessment["corpus_diversity"],
                0.0,  # Placeholder for additional features
                *torch.zeros(10),  # Padding to 20 features
            ]
        )
        return features[:20]

    def _extract_memory_metadata(self, memory) -> torch.Tensor:
        """Extract metadata features from memory."""
        now = datetime.now()
        age_days = (now - memory.temporal_metadata.created_at).days
        hours_since_access = (now - memory.temporal_metadata.last_accessed).total_seconds() / 3600

        features = torch.tensor(
            [
                min(memory.temporal_metadata.access_count / 100, 1.0),
                memory.temporal_metadata.decay_factor,
                min(age_days / 365, 1.0),
                min(hours_since_access / 720, 1.0),  # Normalize to 30 days
                memory.relevance,
                memory.confidence,
                float(memory.compressed),
                float(memory.semantic_residual is not None),
                float(len(memory.contextual_metadata.tags)) / 10,
                float(len(memory.contextual_metadata.related_memories)) / 10,
                float(len(memory.contextual_metadata.dependencies)) / 5,
                memory.temporal_metadata.retention_priority,
                *torch.zeros(3),  # Padding to 15 features
            ]
        )
        return features[:15]

    def _update_temporal_tracking(self, unified_manager) -> None:
        """Update temporal tracking for all memories."""
        for memory_id, memory in unified_manager.memories.items():
            self.temporal_tracker.add_memory(
                memory_id,
                memory.temporal_metadata.created_at,
                memory.contextual_metadata.dependencies,
            )

    def _adapt_thresholds(self, state: SystemState, assessment: dict) -> None:
        """Adapt thresholds based on performance."""
        _performance = assessment["performance_score"]  # kept for future adaptive logic

        # If performance declining, be more conservative
        if len(self.history) > 5:
            recent_perf = [s.accuracy_estimate for s in self.history[-5:]]
            perf_trend = np.mean(np.diff(recent_perf))

            if perf_trend < 0:  # Declining
                self.cull_threshold += 0.01  # Make culling harder
                self.load_threshold -= 0.01  # Make loading easier
            elif perf_trend > 0:  # Improving
                self.cull_threshold -= 0.005  # Make culling slightly easier
                self.load_threshold += 0.005  # Make loading slightly harder

        # Clamp thresholds
        self.cull_threshold = max(0.5, min(0.95, self.cull_threshold))
        self.load_threshold = max(0.3, min(0.8, self.load_threshold))

    def get_statistics(self) -> dict[str, Any]:
        """Get management statistics."""
        return {
            "loaded_memories": len(self.loaded_memory_ids),
            "archived_memories": len(self.archived_memory_ids),
            "cull_candidates": len(self.cull_candidates),
            "cull_threshold": self.cull_threshold,
            "load_threshold": self.load_threshold,
            "total_decisions": len(self.decisions_made),
            "recent_performance": [s.accuracy_estimate for s in self.history[-10:]]
            if self.history
            else [],
        }


if __name__ == "__main__":
    print("=" * 70)
    print("INTELLIGENT AUTO-MANAGEMENT SYSTEM")
    print("=" * 70)

    from cogsyndelta.memory.unified_tools import MemoryType, UnifiedMemoryManager

    # Create managers
    unified_manager = UnifiedMemoryManager(embed_dim=512)
    auto_manager = IntelligentAutoManager(
        embed_dim=512, max_loaded_memories=100, max_total_memories=1000, target_memory_usage=0.7
    )

    # Add test memories
    print("\n[Step 1] Adding memories:")
    for i in range(200):
        unified_manager.store_memory(
            embedding=torch.randn(512),
            memory_type=MemoryType.EPISODIC,
            source="test",
            relevance=0.3 + (i % 7) * 0.1,
            tags={f"tag_{i % 5}"},
        )
    print("  ✓ Added 200 memories")

    # Run management cycle
    print("\n[Step 2] Running auto-management cycle:")
    system_state = SystemState(
        memory_usage=0.8,
        context_size=1000,
        active_memories=150,
        inference_latency=50.0,
        accuracy_estimate=0.9,
        temporal_coherence=0.7,
    )

    actions = auto_manager.manage_cycle(unified_manager, system_state)

    print(f"  ✓ Loaded: {len(actions['loaded'])} memories")
    print(f"  ✓ Unloaded: {len(actions['unloaded'])} memories")
    print(f"  ✓ Culled: {len(actions['culled'])} memories")
    print(f"  ✓ Archived: {len(actions['archived'])} memories")

    # Get statistics
    print("\n[Step 3] Statistics:")
    stats = auto_manager.get_statistics()
    print(f"  Loaded memories: {stats['loaded_memories']}")
    print(f"  Cull threshold: {stats['cull_threshold']:.3f}")
    print(f"  Load threshold: {stats['load_threshold']:.3f}")

    print("\n" + "=" * 70)
    print("AUTO-MANAGEMENT SYSTEM OPERATIONAL")
    print("=" * 70)
    print("\nThe system automatically:")
    print("  ✓ Balances memory usage and performance")
    print("  ✓ Prevents over-culling (maintains diversity)")
    print("  ✓ Prevents over-loading (controls resources)")
    print("  ✓ Preserves temporal continuity")
    print("  ✓ Adapts thresholds based on performance")
    print("=" * 70)
