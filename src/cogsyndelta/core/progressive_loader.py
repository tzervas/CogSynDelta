"""
Progressive Dynamic Selective Loading for CogSynDelta

Extends the Interconnect Manager with memory-efficient submodel loading/unloading
for consumer GPUs (RTX 5080, 16GB VRAM). Designed for pseudo-generalized highly
specialized models with selective and progressive loading.

Key Features:
- Progressive loading: Load submodels on-demand based on interconnect routing
- Dynamic unloading: Evict unused submodels to free VRAM
- Selective activation: Only keep active submodels in GPU memory
- Context-aware prefetching: Predict and preload likely-needed submodels
- Specialized model composition: Combine domain-specific submodels dynamically

Design Philosophy:
    This system targets a practical configuration of 10-20 specialized submodels
    with 4-8 active simultaneously. Rather than scaling to massive monolithic
    models, the approach favors highly specialized, smaller models that can be
    composed dynamically based on task requirements.

Memory Budget Strategy (RTX 5080 16GB):
    ├── Core framework: ~2GB (always loaded)
    ├── Active submodels: ~8GB (dynamic, 4-8 submodels)
    ├── KV Cache: ~2GB
    ├── Activations: ~2GB (checkpointed)
    └── Overhead: ~2GB
    TOTAL: ~16GB ✅

Typical Configuration:
    - Total submodels: 10-20 specialized modules
    - Simultaneously active: 4-8 submodels (~8GB)
    - Load/unload latency: ~100ms per submodel
    - Prefetch accuracy: >80% (reduces effective latency)

Architecture:
    ┌──────────────────────────────────────┐
    │ Progressive Loader Manager           │
    │ - Memory budget tracking             │
    │ - Load/unload scheduling             │
    │ - Prefetch orchestration             │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────┴───────────────────────┐
    │ Interconnect Manager (existing)      │
    │ - mHC routing decisions              │
    │ - Context-aware communication        │
    │ - Usage pattern tracking             │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────┴───────────────────────┐
    │ Submodel Pool                        │
    │ Active (GPU): Immediately available  │
    │ Staged (CPU): Fast load (~100ms)     │
    │ Archive (Disk): Lazy load (~1s)      │
    └──────────────────────────────────────┘
"""

import logging
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cogsyndelta.core.interconnect_manager import IntelligentInterconnectManager

logger = logging.getLogger(__name__)


class LoadingState(Enum):
    """Lifecycle state of a submodel.

    State machine:
        DISK -> LOADING -> LOADED
        LOADED -> STAGED (moved to CPU staging)
        STAGED -> LOADING -> LOADED
        LOADED -> DISK (evicted)
    """

    DISK = "disk"  # Weights on disk only
    LOADING = "loading"  # Currently loading (disk->GPU or CPU->GPU)
    LOADED = "loaded"  # On GPU, ready for inference
    STAGED = "staged"  # Weights in CPU RAM (fast reload)
    EVICTING = "evicting"  # Currently being evicted from GPU


@dataclass
class SubmodelInfo:
    """Information about a registered submodel."""

    name: str
    module: nn.Module
    parameter_count: int
    size_mb: float  # Memory size in MB (aliased from memory_mb for test compat)
    state: LoadingState = LoadingState.DISK
    submodel_type: str = "default"  # Type for categorization
    pinned: bool = False  # Never unload (alias for pin_memory)

    # Usage tracking
    access_count: int = 0
    last_accessed: float = 0.0
    avg_compute_time: float = 0.0

    # Context affinity (from interconnect routing)
    context_tags: set[str] = field(default_factory=set)
    co_activation_freq: dict[str, int] = field(default_factory=dict)

    # Configuration
    load_priority: int = 5  # 0-10, higher = load first
    allow_cpu_fallback: bool = True
    checkpoint_path: Path | None = None

    @property
    def memory_mb(self) -> float:
        """Alias for size_mb for backward compatibility."""
        return self.size_mb

    @property
    def pin_memory(self) -> bool:
        """Alias for pinned for backward compatibility."""
        return self.pinned


@dataclass
class ProgressiveLoadingConfig:
    """Configuration for progressive loading system."""

    # Memory budgets (MB)
    gpu_budget_mb: float = 8000.0  # 8GB for submodels (conservative for 16GB GPU)
    cpu_staging_size_mb: float = 16000.0  # 16GB CPU staging
    disk_cache_size_mb: float = 50000.0  # 50GB disk cache

    # Active limit - designed for 10-20 total submodels
    max_active_submodels: int = 8  # Maximum submodels on GPU simultaneously

    # Loading thresholds
    eager_load_threshold: float = 0.7  # Load if >70% routing probability
    lazy_unload_threshold: float = 0.2  # Unload if <20% probability
    prefetch_depth: int = 2  # Prefetch co-activated (depth 2 for smaller pools)

    # Performance
    async_loading: bool = True  # Async GPU transfers
    background_prefetch: bool = True  # Prefetch in background
    max_concurrent_loads: int = 2  # Parallel loads

    # Checkpointing
    enable_activation_checkpointing: bool = True
    checkpoint_segments: int = 4

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.eager_load_threshold <= self.lazy_unload_threshold:
            raise AssertionError(
                f"eager_load_threshold ({self.eager_load_threshold}) must be greater than "
                f"lazy_unload_threshold ({self.lazy_unload_threshold}) to prevent thrashing"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, profile: str = "base") -> ProgressiveLoadingConfig:
        """Load configuration from YAML file.

        Args:
            cls: The class itself.
            yaml_path: Path to YAML configuration file
            profile: Configuration profile name (base, focused, balanced, expansive)

        Returns:
            ProgressiveLoadingConfig instance
        """
        import yaml

        with open(yaml_path) as f:
            config_data = yaml.safe_load(f)

        profile_data = config_data.get("profiles", {}).get(profile, {})

        return cls(
            gpu_budget_mb=profile_data.get("gpu_budget_mb", 8000.0),
            cpu_staging_size_mb=profile_data.get("cpu_staging_size_mb", 16000.0),
            disk_cache_size_mb=profile_data.get("disk_cache_size_mb", 50000.0),
            max_active_submodels=profile_data.get("max_active_submodels", 8),
            eager_load_threshold=profile_data.get("eager_load_threshold", 0.7),
            lazy_unload_threshold=profile_data.get("lazy_unload_threshold", 0.2),
            prefetch_depth=profile_data.get("prefetch_depth", 2),
            async_loading=profile_data.get("async_loading", True),
            background_prefetch=profile_data.get("background_prefetch", True),
            max_concurrent_loads=profile_data.get("max_concurrent_loads", 2),
        )

    # Backward compatibility aliases
    @property
    def cpu_staging_mb(self) -> float:
        """Alias for cpu_staging_size_mb."""
        return self.cpu_staging_size_mb

    @property
    def disk_cache_mb(self) -> float:
        """Alias for disk_cache_size_mb."""
        return self.disk_cache_size_mb


# Alias for backward compatibility
LoadingConfig = ProgressiveLoadingConfig


class ProgressiveLoaderManager:
    """Manages progressive loading/unloading of submodels for memory efficiency.

    Integrates with IntelligentInterconnectManager to use routing decisions
    for predictive prefetching.
    """

    def __init__(
        self,
        config: ProgressiveLoadingConfig | None = None,
        interconnect: IntelligentInterconnectManager | None = None,
        device: str = "cuda",
        max_active_submodels: int | None = None,
    ) -> None:
        """Initialize the ProgressiveLoaderManager.

        Args:
            config: Optional configuration for progressive loading.
            interconnect: Optional IntelligentInterconnectManager.
            device: Target device for loading (e.g., 'cuda' or 'cpu').
            max_active_submodels: Optional override for max active submodels.
        """
        self.config = config or ProgressiveLoadingConfig()
        self.device = torch.device(device)
        self.interconnect = interconnect

        # Override max_active if provided
        if max_active_submodels is not None:
            self.config.max_active_submodels = max_active_submodels

        # Submodel registry
        self.submodels: dict[str, SubmodelInfo] = {}
        self.active_names: set[str] = set()

        # Memory tracking
        self.gpu_memory_used = 0.0
        self.cpu_memory_used = 0.0

        # State dicts (CPU storage)
        self.state_dicts: dict[str, dict[str, torch.Tensor]] = {}

        # Loading queues - use queue.Queue for thread-safety (not asyncio.Queue)
        import queue

        self._load_queue: queue.Queue[str] = queue.Queue()
        self._prefetch_queue: queue.Queue[str] = queue.Queue()

        logger.info(
            f"ProgressiveLoaderManager initialized: "
            f"GPU budget={self.config.gpu_budget_mb:.1f}MB, "
            f"max_active={self.config.max_active_submodels}, "
            f"CPU staging={self.config.cpu_staging_mb:.1f}MB"
        )

    @property
    def load_queue(self) -> queue.Queue[str]:
        """Get load queue (property for backward compatibility)."""
        return self._load_queue

    @property
    def prefetch_queue(self) -> queue.Queue[str]:
        """Get prefetch queue (property for backward compatibility)."""
        return self._prefetch_queue

    def register_submodel(
        self,
        name: str,
        module: nn.Module,
        context_tags: set[str] | None = None,
        load_priority: int = 5,
        pin_memory: bool = False,
        checkpoint_path: Path | None = None,
        size_mb: float | None = None,
        submodel_type: str = "default",
    ) -> None:
        """Register a submodel for progressive loading.

        Args:
            name: Unique submodel identifier
            module: The nn.Module instance
            context_tags: Context tags for routing affinity
            load_priority: Loading priority (0-10, higher first)
            pin_memory: Keep on GPU permanently (for core modules)
            checkpoint_path: Path to saved weights (if separate from module)
            size_mb: Override calculated memory size (MB)
            submodel_type: Type of submodel for categorization
        """
        param_count = sum(p.numel() for p in module.parameters())
        calc_memory_mb = param_count * 4 / (1024**2)  # float32 assumption
        actual_size_mb = size_mb if size_mb is not None else calc_memory_mb

        info = SubmodelInfo(
            name=name,
            module=module,
            parameter_count=param_count,
            size_mb=actual_size_mb,
            state=LoadingState.STAGED,
            submodel_type=submodel_type,
            pinned=pin_memory,
            context_tags=context_tags or set(),
            load_priority=load_priority,
            checkpoint_path=checkpoint_path,
        )

        self.submodels[name] = info

        # Save state dict to CPU
        self.state_dicts[name] = {k: v.cpu() for k, v in module.state_dict().items()}
        self.cpu_memory_used += actual_size_mb

        # Move module to CPU initially (unless pinned)
        if not pin_memory:
            module.to("cpu")
        else:
            # Pin memory modules loaded immediately
            self._load_to_gpu(name)

        logger.info(
            f"Registered submodel '{name}': {param_count:,} params, "
            f"{actual_size_mb:.1f}MB, priority={load_priority}, pinned={pin_memory}, "
            f"type={submodel_type}, tags={context_tags}"
        )

    def is_loaded(self, name: str) -> bool:
        """Check if a submodel is currently loaded on GPU.

        Args:
            name: Submodel name

        Returns:
            True if loaded on GPU
        """
        if name not in self.submodels:
            return False
        return self.submodels[name].state == LoadingState.LOADED

    def is_loading(self, name: str) -> bool:
        """Check if a submodel is currently being loaded.

        Args:
            name: Submodel name

        Returns:
            True if currently loading
        """
        if name not in self.submodels:
            return False
        return self.submodels[name].state == LoadingState.LOADING

    def get_loaded(self, name: str) -> nn.Module | None:
        """Get a loaded submodel's module.

        Args:
            name: Submodel name

        Returns:
            The nn.Module if loaded, None otherwise
        """
        if name not in self.submodels:
            return None
        info = self.submodels[name]
        if info.state == LoadingState.LOADED:
            return info.module
        return None

    def pin_submodel(self, name: str) -> bool:
        """Pin a submodel to GPU (prevent eviction).

        Args:
            name: Submodel name

        Returns:
            True if successfully pinned
        """
        if name not in self.submodels:
            return False

        info = self.submodels[name]
        info.pinned = True

        # Load to GPU if not already loaded
        if info.state != LoadingState.LOADED:
            self.activate(name)

        return True

    def clear(self) -> None:
        """Deactivate all submodels and clear state."""
        for name in list(self.active_names):
            self.deactivate(name)
        self.gpu_memory_used = 0.0

    def activate(self, name: str, blocking: bool = True) -> bool:
        """Activate a submodel (load to GPU if needed).

        Args:
            name: Submodel to activate
            blocking: Wait for load to complete

        Returns:
            True if successful or already active
        """
        info = self.submodels[name]

        # Already loaded
        if info.state == LoadingState.LOADED:
            return True

        # Check max_active_submodels limit
        if len(self.active_names) >= self.config.max_active_submodels:
            # Need to evict an existing submodel
            if not self._evict_least_used():
                logger.warning(
                    f"Cannot activate '{name}': max_active_submodels limit reached "
                    f"({self.config.max_active_submodels})"
                )
                return False

        # Check memory budget
        if not self._can_load(info.size_mb):
            # Try to make room
            if not self._evict_for_space(info.size_mb):
                logger.warning(
                    f"Cannot activate '{name}': insufficient memory "
                    f"({info.size_mb:.1f}MB needed, {self._available_gpu_mb():.1f}MB available)"
                )
                return False

        # Load to GPU
        if blocking:
            success = self._load_to_gpu(name)
        else:
            # Async load
            self._load_queue.put_nowait(name)
            success = True

        return success

    def _evict_least_used(self) -> bool:
        """Evict the least recently used non-pinned submodel.

        Returns:
            True if a submodel was evicted
        """
        # Find least recently used non-pinned active submodel
        candidates = [
            (name, self.submodels[name])
            for name in self.active_names
            if not self.submodels[name].pinned
        ]

        if not candidates:
            return False

        # Sort by last_accessed (oldest first)
        candidates.sort(key=lambda x: x[1].last_accessed)

        # Evict oldest
        name, _ = candidates[0]
        return self._unload_from_gpu(name)

    def deactivate(self, name: str) -> bool:
        """Deactivate a submodel (unload from GPU).

        Args:
            name: Submodel to deactivate

        Returns:
            True if successful
        """
        info = self.submodels[name]

        if info.state != LoadingState.LOADED:
            return False

        if info.pinned:
            logger.warning(f"Cannot deactivate pinned submodel '{name}'")
            return False

        return self._unload_from_gpu(name)

    def update_from_routing(
        self, routing_decisions: dict[tuple[str, str], float], section_to_submodel: dict[str, str]
    ) -> None:
        """Update loading based on interconnect routing decisions.

        Args:
            routing_decisions: {(source, target): importance} from interconnect
            section_to_submodel: Mapping from section ID to submodel name
        """
        # Track which submodels are being routed to
        routed_submodels: dict[str, float] = {}

        for (source, target), importance in routing_decisions.items():
            # Get submodels for source and target
            source_sub = section_to_submodel.get(source)
            target_sub = section_to_submodel.get(target)

            if source_sub:
                routed_submodels[source_sub] = max(
                    routed_submodels.get(source_sub, 0.0), importance
                )
            if target_sub:
                routed_submodels[target_sub] = max(
                    routed_submodels.get(target_sub, 0.0), importance
                )

        # Activate high-importance submodels
        for name, importance in routed_submodels.items():
            if name in self.submodels:
                info = self.submodels[name]

                # Eager loading for high-importance routes
                if importance >= self.config.eager_load_threshold:
                    if info.state != LoadingState.LOADED:
                        logger.info(f"Eager loading '{name}' (importance={importance:.2f})")
                        self.activate(name, blocking=False)

                # Update access stats
                info.last_accessed = time.time()
                info.access_count += 1

        # Deactivate low-importance submodels
        for name in list(self.active_names):
            importance = routed_submodels.get(name, 0.0)
            info = self.submodels[name]

            if importance < self.config.lazy_unload_threshold and not info.pin_memory:
                time_since_access = time.time() - info.last_accessed

                # Unload if not accessed recently and low importance
                if time_since_access > 10.0:  # 10 seconds idle
                    logger.info(
                        f"Lazy unloading '{name}' (importance={importance:.2f}, "
                        f"idle={time_since_access:.1f}s)"
                    )
                    self.deactivate(name)

        # Prefetch co-activated submodels
        if self.config.background_prefetch:
            self._prefetch_co_activated(list(routed_submodels.keys()))

    def _load_to_gpu(self, name: str) -> bool:
        """Load submodel weights to GPU.

        Args:
            name: Submodel name

        Returns:
            True if successful
        """
        info = self.submodels[name]

        if info.state == LoadingState.LOADED:
            return True

        logger.info(f"Loading '{name}' to GPU ({info.memory_mb:.1f}MB)...")

        info.state = LoadingState.LOADING
        start_time = time.time()

        try:
            # Load state dict
            state_dict = self.state_dicts[name]
            info.module.load_state_dict(state_dict)

            # Move to GPU
            info.module.to(self.device)

            # Update tracking
            self.gpu_memory_used += info.memory_mb
            info.state = LoadingState.LOADED
            self.active_names.add(name)

            load_time = time.time() - start_time
            logger.info(
                f"Loaded '{name}' in {load_time * 1000:.1f}ms. "
                f"GPU: {self.gpu_memory_used:.1f}/{self.config.gpu_budget_mb:.1f}MB "
                f"({self.gpu_memory_used / self.config.gpu_budget_mb * 100:.1f}%)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load '{name}': {e}")
            info.state = LoadingState.STAGED
            return False

    def _unload_from_gpu(self, name: str) -> bool:
        """Unload submodel from GPU to CPU.

        Args:
            name: Submodel name

        Returns:
            True if successful
        """
        info = self.submodels[name]

        if info.state != LoadingState.LOADED:
            return False

        logger.info(f"Unloading '{name}' from GPU ({info.memory_mb:.1f}MB freed)...")

        info.state = LoadingState.EVICTING

        try:
            # Save state dict to CPU
            self.state_dicts[name] = {k: v.cpu() for k, v in info.module.state_dict().items()}

            # Move module to CPU
            info.module.to("cpu")

            # Update tracking
            self.gpu_memory_used -= info.memory_mb
            info.state = LoadingState.STAGED
            self.active_names.discard(name)

            logger.info(
                f"Unloaded '{name}'. "
                f"GPU: {self.gpu_memory_used:.1f}/{self.config.gpu_budget_mb:.1f}MB"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to unload '{name}': {e}")
            info.state = LoadingState.LOADED
            return False

    def _can_load(self, memory_mb: float) -> bool:
        """Check if there's enough GPU memory for loading."""
        available = self.config.gpu_budget_mb - self.gpu_memory_used
        return memory_mb <= available

    def _available_gpu_mb(self) -> float:
        """Get available GPU memory in MB."""
        return self.config.gpu_budget_mb - self.gpu_memory_used

    def _evict_for_space(self, required_mb: float) -> bool:
        """Evict submodels to make room for new loading.

        Args:
            required_mb: Memory needed

        Returns:
            True if enough space freed
        """
        available = self._available_gpu_mb()

        if required_mb <= available:
            return True

        needed = required_mb - available

        # Find eviction candidates (not pinned, least recently used)
        candidates = []
        current_time = time.time()

        for name, info in self.submodels.items():
            if info.state == LoadingState.LOADED and not info.pin_memory:
                # Score: higher = better candidate for eviction
                recency = current_time - info.last_accessed
                access_freq = 1.0 / (1 + info.access_count)
                score = recency * access_freq

                candidates.append((score, info.memory_mb, name))

        candidates.sort(reverse=True)

        freed = 0.0
        evicted = []

        for score, memory, name in candidates:
            if freed >= needed:
                break

            if self._unload_from_gpu(name):
                freed += memory
                evicted.append(name)

        if evicted:
            logger.info(f"Evicted {len(evicted)} submodels ({freed:.1f}MB freed): {evicted}")

        return freed >= needed

    def _prefetch_co_activated(self, active_names: list[str]) -> None:
        """Prefetch frequently co-activated submodels.

        Args:
            active_names: Currently active submodel names
        """
        # Collect co-activation frequencies
        prefetch_scores: dict[str, float] = {}

        for name in active_names:
            info = self.submodels[name]

            for other_name, freq in info.co_activation_freq.items():
                if other_name not in self.active_names:
                    # Score by frequency and availability
                    other_info = self.submodels[other_name]
                    if other_info.state == LoadingState.STAGED:
                        prefetch_scores[other_name] = freq

        # Sort by score and prefetch top N
        top_prefetch = sorted(prefetch_scores.items(), key=lambda x: x[1], reverse=True)[
            : self.config.prefetch_depth
        ]

        for name, score in top_prefetch:
            logger.debug(f"Prefetching '{name}' (co-activation score={score})")
            self._prefetch_queue.put_nowait(name)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive loading statistics."""
        active_modules = list(self.active_names)
        staged_modules = [n for n, i in self.submodels.items() if i.state == LoadingState.STAGED]
        loaded_count = len(active_modules)

        return {
            "gpu_memory_mb": self.gpu_memory_used,
            "gpu_budget_mb": self.config.gpu_budget_mb,
            "gpu_utilization": self.gpu_memory_used / self.config.gpu_budget_mb
            if self.config.gpu_budget_mb > 0
            else 0.0,
            "cpu_memory_mb": self.cpu_memory_used,
            "total_submodels": len(self.submodels),
            # Multiple names for compatibility
            "active_submodels": loaded_count,
            "num_loaded": loaded_count,  # Test-expected key
            "max_active": self.config.max_active_submodels,  # Test-expected key
            "staged_submodels": len(staged_modules),
            "active_names": active_modules,
            "pinned_modules": [n for n, i in self.submodels.items() if i.pinned],
        }


# Example usage and configuration for different submodel pool sizes
class ModelScaleConfig:
    """Predefined configurations for different submodel compositions.

    These configs represent different strategies for composing specialized submodels:
    - focused: 4-6 active from ~10 total (task-specific composition)
    - balanced: 6-8 active from ~15 total (general-purpose)
    - expansive: 8-10 active from ~20 total (maximum specialization)
    """

    @staticmethod
    def focused() -> ProgressiveLoadingConfig:
        """Configuration for focused task composition (4-6 active from ~10 submodels).

        Best for: Single-domain tasks with clear specialization requirements.
        Target: RTX 3090/4090 (24GB), RTX 5080 (16GB)
        """
        return ProgressiveLoadingConfig(
            gpu_budget_mb=6000.0,
            cpu_staging_size_mb=12000.0,
            max_active_submodels=6,
            eager_load_threshold=0.7,
            lazy_unload_threshold=0.2,
            prefetch_depth=2,
        )

    @staticmethod
    def balanced() -> ProgressiveLoadingConfig:
        """Configuration for balanced composition (6-8 active from ~15 submodels).

        Best for: Multi-task workloads with moderate specialization.
        Target: RTX 5080 (16GB) primary, RTX 4090 (24GB)
        """
        return ProgressiveLoadingConfig(
            gpu_budget_mb=8000.0,
            cpu_staging_size_mb=16000.0,
            max_active_submodels=8,
            eager_load_threshold=0.65,
            lazy_unload_threshold=0.2,
            prefetch_depth=2,
            max_concurrent_loads=2,
        )

    @staticmethod
    def expansive() -> ProgressiveLoadingConfig:
        """Configuration for expansive composition (8-10 active from ~20 submodels).

        Best for: Complex tasks requiring broad domain coverage.
        Target: High-VRAM setups (24GB+), multi-GPU
        """
        return ProgressiveLoadingConfig(
            gpu_budget_mb=10000.0,
            cpu_staging_size_mb=24000.0,
            max_active_submodels=10,
            eager_load_threshold=0.6,
            lazy_unload_threshold=0.25,
            prefetch_depth=3,
            max_concurrent_loads=3,
            background_prefetch=True,
        )

    # Legacy aliases for backward compatibility (deprecated)
    @staticmethod
    def small_10b() -> ProgressiveLoadingConfig:
        """DEPRECATED: Use focused() instead."""
        return ModelScaleConfig.focused()

    @staticmethod
    def medium_25b() -> ProgressiveLoadingConfig:
        """DEPRECATED: Use balanced() instead."""
        return ModelScaleConfig.balanced()

    @staticmethod
    def large_50b() -> ProgressiveLoadingConfig:
        """DEPRECATED: Use expansive() instead."""
        return ModelScaleConfig.expansive()

    @staticmethod
    def xlarge_100b() -> ProgressiveLoadingConfig:
        """DEPRECATED: Use expansive() instead.

        Note: CogSynDelta focuses on composing specialized submodels,
        not scaling to massive monolithic models.
        """
        return ModelScaleConfig.expansive()
