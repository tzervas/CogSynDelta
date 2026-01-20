"""Process-specific resource tracking for benchmarks.

This module provides fine-grained resource tracking that captures CPU, GPU,
VRAM, RAM, and disk usage *specifically* for benchmark processes, not system-wide.

Classes:
    ProcessResourceTracker: Track resources for a specific process/benchmark
    CPUTopology: CPU core topology (P-cores, E-cores, core complexes)
    GPUTopology: GPU compute unit topology (SMs, tensor cores)
    ResourceSnapshot: Point-in-time resource usage snapshot
    ResourceTimeSeries: Time series of resource snapshots during benchmark

Why process-specific:
    System-wide metrics include unrelated processes. For accurate benchmark
    comparison, we need resources consumed by the benchmark itself.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from benchmarks.metrics.base import METRIC_REGISTRY, MetricMixin


@dataclass
class CPUCoreInfo:
    """Information about a single CPU core.

    Attributes:
        core_id: Physical core ID.
        logical_ids: Logical processor IDs (threads) on this core.
        core_type: "performance", "efficiency", or "unknown".
        base_freq_mhz: Base frequency in MHz.
        max_freq_mhz: Maximum turbo frequency in MHz.
        current_freq_mhz: Current frequency at capture time.
        utilization_pct: Current utilization percentage.
    """

    core_id: int
    logical_ids: list[int] = field(default_factory=list)
    core_type: str = "unknown"  # "performance", "efficiency", "unknown"
    base_freq_mhz: float | None = None
    max_freq_mhz: float | None = None
    current_freq_mhz: float | None = None
    utilization_pct: float | None = None


@dataclass
class CPUTopology(MetricMixin):
    """CPU core topology with P-core/E-core distinction.

    Modern Intel CPUs (12th gen+) and ARM big.LITTLE have heterogeneous
    cores. This tracks utilization by core type for better analysis.

    Attributes:
        total_physical_cores: Total physical cores.
        total_logical_cores: Total logical processors (with HT/SMT).
        performance_cores: Number of performance (P) cores.
        efficiency_cores: Number of efficiency (E) cores.
        p_core_utilization_pct: Average P-core utilization.
        e_core_utilization_pct: Average E-core utilization.
        cores: Detailed per-core information.
        architecture: CPU architecture string.
    """

    total_physical_cores: int = 0
    total_logical_cores: int = 0
    performance_cores: int = 0
    efficiency_cores: int = 0
    p_core_utilization_pct: float | None = None
    e_core_utilization_pct: float | None = None
    overall_utilization_pct: float | None = None
    cores: list[CPUCoreInfo] = field(default_factory=list)
    architecture: str = "unknown"
    capture_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def detect(cls) -> CPUTopology:
        """Detect CPU topology from system.

        Attempts to detect P-core/E-core configuration on Intel hybrid
        and ARM big.LITTLE architectures.

        Returns:
            CPUTopology with detected configuration.
        """
        if not HAS_PSUTIL:
            return cls()

        try:
            physical = psutil.cpu_count(logical=False) or 0
            logical = psutil.cpu_count(logical=True) or 0

            # Try to detect architecture
            arch = "unknown"
            with contextlib.suppress(Exception):
                import platform

                arch = platform.processor() or platform.machine()

            # Get per-CPU frequencies and utilization
            cores: list[CPUCoreInfo] = []
            p_cores = 0
            e_cores = 0

            try:
                freqs = psutil.cpu_freq(percpu=True) or []
                utils = psutil.cpu_percent(interval=0.1, percpu=True)

                # Heuristic: on hybrid CPUs, higher max freq = P-core
                # This is imperfect but better than nothing
                if freqs:
                    max_freqs = [f.max for f in freqs if f]
                    if max_freqs:
                        avg_max = sum(max_freqs) / len(max_freqs)

                        for i, (freq, util) in enumerate(zip(freqs, utils)):
                            if freq:
                                core_type = "unknown"
                                if freq.max > avg_max * 1.1:
                                    core_type = "performance"
                                    p_cores += 1
                                elif freq.max < avg_max * 0.9:
                                    core_type = "efficiency"
                                    e_cores += 1

                                cores.append(
                                    CPUCoreInfo(
                                        core_id=i,
                                        logical_ids=[i],
                                        core_type=core_type,
                                        base_freq_mhz=freq.min,
                                        max_freq_mhz=freq.max,
                                        current_freq_mhz=freq.current,
                                        utilization_pct=util,
                                    )
                                )
            except (OSError, AttributeError):
                pass

            # Calculate utilization by core type
            p_utils = [c.utilization_pct for c in cores if c.core_type == "performance" and c.utilization_pct]
            e_utils = [c.utilization_pct for c in cores if c.core_type == "efficiency" and c.utilization_pct]
            all_utils = [c.utilization_pct for c in cores if c.utilization_pct]

            return cls(
                total_physical_cores=physical,
                total_logical_cores=logical,
                performance_cores=p_cores,
                efficiency_cores=e_cores,
                p_core_utilization_pct=sum(p_utils) / len(p_utils) if p_utils else None,
                e_core_utilization_pct=sum(e_utils) / len(e_utils) if e_utils else None,
                overall_utilization_pct=sum(all_utils) / len(all_utils) if all_utils else None,
                cores=cores,
                architecture=arch,
            )
        except Exception:
            return cls()

    def summary(self) -> dict[str, Any]:
        """Get summary without per-core details."""
        return {
            "physical_cores": self.total_physical_cores,
            "logical_cores": self.total_logical_cores,
            "p_cores": self.performance_cores,
            "e_cores": self.efficiency_cores,
            "p_core_util_pct": self.p_core_utilization_pct,
            "e_core_util_pct": self.e_core_utilization_pct,
            "overall_util_pct": self.overall_utilization_pct,
            "architecture": self.architecture,
        }


@dataclass
class GPUComputeUnit:
    """Information about GPU compute units (SMs/CUs).

    Attributes:
        unit_id: Compute unit ID.
        unit_type: "sm" (NVIDIA), "cu" (AMD), "xe_core" (Intel).
        utilization_pct: Current utilization percentage.
        active_warps: Number of active warps/wavefronts.
        max_warps: Maximum warps capacity.
    """

    unit_id: int
    unit_type: str = "sm"  # "sm", "cu", "xe_core"
    utilization_pct: float | None = None
    active_warps: int | None = None
    max_warps: int | None = None


@dataclass
class GPUTopology(MetricMixin):
    """GPU compute topology with SM/CU details.

    Tracks streaming multiprocessors (NVIDIA), compute units (AMD),
    or Xe cores (Intel) utilization and occupancy.

    Attributes:
        device_name: GPU device name.
        total_sms: Total streaming multiprocessors / compute units.
        active_sms: Currently active SMs.
        sm_occupancy_pct: Average SM occupancy percentage.
        tensor_core_utilization_pct: Tensor core utilization (if available).
        memory_controller_util_pct: Memory controller utilization.
        pcie_bandwidth_util_pct: PCIe bandwidth utilization.
        nvlink_bandwidth_util_pct: NVLink bandwidth utilization (multi-GPU).
        compute_units: Detailed per-unit information.
    """

    device_name: str = "unknown"
    compute_capability: tuple[int, int] | None = None
    total_sms: int = 0
    active_sms: int | None = None
    sm_occupancy_pct: float | None = None
    tensor_core_utilization_pct: float | None = None
    memory_controller_util_pct: float | None = None
    pcie_bandwidth_util_pct: float | None = None
    nvlink_bandwidth_util_pct: float | None = None
    compute_units: list[GPUComputeUnit] = field(default_factory=list)
    capture_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def detect(cls, gpu_index: int = 0) -> GPUTopology:
        """Detect GPU topology from system.

        Args:
            gpu_index: GPU device index.

        Returns:
            GPUTopology with detected configuration.
        """
        if not HAS_PYNVML:
            return cls()

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            device_name = "unknown"
            with contextlib.suppress(pynvml.NVMLError):
                raw_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(raw_name, bytes):
                    device_name = raw_name.decode("utf-8")
                elif isinstance(raw_name, str):
                    device_name = raw_name

            compute_cap = None
            with contextlib.suppress(pynvml.NVMLError):
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_cap = (major, minor)

            # SM count - try NVML first, fall back to lookup table
            total_sms = 0
            with contextlib.suppress(pynvml.NVMLError, AttributeError):
                # nvmlDeviceGetNumGpuCores is NVML 11+ and may not be available
                cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
                if cores > 0:
                    # 128 CUDA cores per SM on most architectures
                    total_sms = cores // 128

            # Fallback: lookup table for known GPUs by compute capability
            if total_sms == 0 and compute_cap:
                sm_lookup = {
                    # Blackwell
                    (10, 0): 128,  # GB200 approx
                    # Ada Lovelace
                    (8, 9): 128,  # RTX 4090
                    (8, 6): 84,   # RTX 4080/5080 approx
                    # Ampere
                    (8, 6): 84,   # RTX 3090
                    (8, 0): 108,  # A100
                    # Turing
                    (7, 5): 72,   # RTX 2080 Ti
                    # Volta
                    (7, 0): 80,   # V100
                }
                total_sms = sm_lookup.get(compute_cap, 0)

            # Utilization rates
            sm_util = None
            mem_ctrl_util = None
            with contextlib.suppress(pynvml.NVMLError):
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                sm_util = rates.gpu
                mem_ctrl_util = rates.memory

            # PCIe throughput
            pcie_util = None
            with contextlib.suppress(pynvml.NVMLError):
                tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                # Estimate utilization (PCIe 4.0 x16 = ~32GB/s each direction)
                max_bandwidth = 32 * 1024 * 1024 * 1024  # 32 GB/s in bytes
                pcie_util = ((tx + rx) / 2 / max_bandwidth) * 100

            pynvml.nvmlShutdown()

            return cls(
                device_name=device_name,
                compute_capability=compute_cap,
                total_sms=total_sms,
                sm_occupancy_pct=sm_util,
                memory_controller_util_pct=mem_ctrl_util,
                pcie_bandwidth_util_pct=pcie_util,
            )
        except Exception:
            return cls()

    def summary(self) -> dict[str, Any]:
        """Get summary without per-unit details."""
        return {
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "total_sms": self.total_sms,
            "sm_occupancy_pct": self.sm_occupancy_pct,
            "tensor_core_util_pct": self.tensor_core_utilization_pct,
            "mem_controller_util_pct": self.memory_controller_util_pct,
            "pcie_util_pct": self.pcie_bandwidth_util_pct,
        }


@dataclass
class ResourceSnapshot(MetricMixin):
    """Point-in-time resource usage snapshot for a process.

    Captures all resource usage at a specific moment, specifically
    for the benchmark process (not system-wide).

    Attributes:
        timestamp: When snapshot was taken.
        process_cpu_pct: CPU usage by the process (0-100 per core).
        process_memory_mb: Process memory (RSS) in MB.
        process_memory_pct: Process memory as % of total RAM.
        process_threads: Number of threads in process.
        process_io_read_mb: Cumulative IO reads in MB.
        process_io_write_mb: Cumulative IO writes in MB.
        gpu_memory_used_mb: GPU memory used by process (estimated).
        gpu_utilization_pct: GPU utilization (may include other processes).
        disk_read_rate_mbs: Disk read rate in MB/s.
        disk_write_rate_mbs: Disk write rate in MB/s.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    process_cpu_pct: float | None = None
    process_memory_mb: float | None = None
    process_memory_pct: float | None = None
    process_threads: int | None = None
    process_io_read_mb: float | None = None
    process_io_write_mb: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_utilization_pct: float | None = None
    disk_read_rate_mbs: float | None = None
    disk_write_rate_mbs: float | None = None

    # Detailed topology snapshots
    cpu_topology: CPUTopology | None = None
    gpu_topology: GPUTopology | None = None


@dataclass
class ResourceTimeSeries(MetricMixin):
    """Time series of resource snapshots during a benchmark run.

    Attributes:
        snapshots: List of resource snapshots.
        sample_interval_sec: Time between samples.
        start_time: When tracking started.
        end_time: When tracking ended.
    """

    snapshots: list[ResourceSnapshot] = field(default_factory=list)
    sample_interval_sec: float = 0.5
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Aggregated statistics
    peak_cpu_pct: float | None = None
    avg_cpu_pct: float | None = None
    peak_memory_mb: float | None = None
    avg_memory_mb: float | None = None
    peak_gpu_memory_mb: float | None = None
    avg_gpu_memory_mb: float | None = None
    total_io_read_mb: float | None = None
    total_io_write_mb: float | None = None

    def compute_statistics(self) -> None:
        """Compute aggregate statistics from snapshots."""
        if not self.snapshots:
            return

        cpu_vals = [s.process_cpu_pct for s in self.snapshots if s.process_cpu_pct is not None]
        mem_vals = [s.process_memory_mb for s in self.snapshots if s.process_memory_mb is not None]
        gpu_mem_vals = [s.gpu_memory_used_mb for s in self.snapshots if s.gpu_memory_used_mb is not None]

        if cpu_vals:
            self.peak_cpu_pct = max(cpu_vals)
            self.avg_cpu_pct = sum(cpu_vals) / len(cpu_vals)

        if mem_vals:
            self.peak_memory_mb = max(mem_vals)
            self.avg_memory_mb = sum(mem_vals) / len(mem_vals)

        if gpu_mem_vals:
            self.peak_gpu_memory_mb = max(gpu_mem_vals)
            self.avg_gpu_memory_mb = sum(gpu_mem_vals) / len(gpu_mem_vals)

        # IO totals from last snapshot (cumulative)
        if self.snapshots:
            last = self.snapshots[-1]
            self.total_io_read_mb = last.process_io_read_mb
            self.total_io_write_mb = last.process_io_write_mb

    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        self.compute_statistics()
        return {
            "duration_sec": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time
                else None
            ),
            "num_samples": len(self.snapshots),
            "sample_interval_sec": self.sample_interval_sec,
            "peak_cpu_pct": self.peak_cpu_pct,
            "avg_cpu_pct": self.avg_cpu_pct,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "avg_gpu_memory_mb": self.avg_gpu_memory_mb,
            "total_io_read_mb": self.total_io_read_mb,
            "total_io_write_mb": self.total_io_write_mb,
        }


class ProcessResourceTracker:
    """Track resources used by a specific process during benchmark execution.

    This tracker monitors CPU, memory, GPU, and disk usage specifically
    for the benchmark process, capturing snapshots at regular intervals.

    Example:
        >>> tracker = ProcessResourceTracker(sample_interval=0.5)
        >>> tracker.start()
        >>> # ... run benchmark ...
        >>> tracker.stop()
        >>> print(tracker.get_time_series().summary())
    """

    def __init__(
        self,
        pid: int | None = None,
        sample_interval: float = 0.5,
        include_children: bool = True,
        track_cpu_topology: bool = True,
        track_gpu_topology: bool = True,
        gpu_index: int = 0,
    ) -> None:
        """Initialize resource tracker.

        Args:
            pid: Process ID to track. If None, tracks current process.
            sample_interval: Seconds between samples.
            include_children: Include child process resources.
            track_cpu_topology: Capture detailed CPU core info.
            track_gpu_topology: Capture detailed GPU SM info.
            gpu_index: GPU device index to track.
        """
        self.pid = pid or os.getpid()
        self.sample_interval = sample_interval
        self.include_children = include_children
        self.track_cpu_topology = track_cpu_topology
        self.track_gpu_topology = track_gpu_topology
        self.gpu_index = gpu_index

        self._time_series = ResourceTimeSeries(sample_interval_sec=sample_interval)
        self._running = False
        self._thread: threading.Thread | None = None
        self._process: Any = None  # psutil.Process

    def start(self) -> None:
        """Start background resource tracking."""
        if self._running:
            return

        self._running = True
        self._time_series = ResourceTimeSeries(sample_interval_sec=self.sample_interval)
        self._time_series.start_time = datetime.now(UTC)

        if HAS_PSUTIL:
            self._process = psutil.Process(self.pid)
            # Prime CPU measurement (first call always returns 0)
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                self._process.cpu_percent()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceTimeSeries:
        """Stop tracking and return time series.

        Returns:
            ResourceTimeSeries with all collected snapshots.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        self._time_series.end_time = datetime.now(UTC)
        self._time_series.compute_statistics()
        return self._time_series

    def get_time_series(self) -> ResourceTimeSeries:
        """Get current time series (can be called while running)."""
        return self._time_series

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        while self._running:
            # Intentionally silent on errors - don't crash benchmark due to sampling
            with contextlib.suppress(Exception):
                snapshot = self._capture_snapshot()
                self._time_series.snapshots.append(snapshot)

            time.sleep(self.sample_interval)

    def _capture_snapshot(self) -> ResourceSnapshot:
        """Capture a single resource snapshot."""
        snapshot = ResourceSnapshot(timestamp=datetime.now(UTC))

        if HAS_PSUTIL and self._process:
            try:
                # CPU (returns % across all cores, so 400% = 4 cores fully used)
                snapshot.process_cpu_pct = self._process.cpu_percent()

                # Memory
                mem_info = self._process.memory_info()
                snapshot.process_memory_mb = mem_info.rss / (1024 * 1024)
                snapshot.process_memory_pct = self._process.memory_percent()

                # Threads
                snapshot.process_threads = self._process.num_threads()

                # IO
                try:
                    io_counters = self._process.io_counters()
                    snapshot.process_io_read_mb = io_counters.read_bytes / (1024 * 1024)
                    snapshot.process_io_write_mb = io_counters.write_bytes / (1024 * 1024)
                except (psutil.AccessDenied, AttributeError):
                    pass

                # Include children
                if self.include_children:
                    try:
                        for child in self._process.children(recursive=True):
                            try:
                                child_mem = child.memory_info()
                                snapshot.process_memory_mb = (snapshot.process_memory_mb or 0) + child_mem.rss / (1024 * 1024)
                                snapshot.process_cpu_pct = (snapshot.process_cpu_pct or 0) + child.cpu_percent()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # GPU metrics
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

                # GPU memory (total, not just this process)
                # Note: Per-process GPU memory requires CUDA context
                with contextlib.suppress(pynvml.NVMLError):
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    snapshot.gpu_memory_used_mb = mem_info.used / (1024 * 1024)

                # GPU utilization
                with contextlib.suppress(pynvml.NVMLError):
                    rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot.gpu_utilization_pct = rates.gpu

                pynvml.nvmlShutdown()
            except Exception:  # noqa: S110 - Intentional: GPU tracking should not crash benchmark
                pass

        # Detailed topology (less frequent - expensive)
        if self.track_cpu_topology and len(self._time_series.snapshots) % 10 == 0:
            snapshot.cpu_topology = CPUTopology.detect()

        if self.track_gpu_topology and len(self._time_series.snapshots) % 10 == 0:
            snapshot.gpu_topology = GPUTopology.detect(self.gpu_index)

        return snapshot


@dataclass
class BenchmarkResourceUsage(MetricMixin):
    """Complete resource usage summary for a benchmark run.

    Aggregates all resource tracking into a single summary suitable
    for storage and comparison.

    Attributes:
        benchmark_name: Name of the benchmark.
        model_name: Name of model being benchmarked.
        time_series: Full time series data.
        cpu_topology: CPU topology at start of benchmark.
        gpu_topology: GPU topology at start of benchmark.
    """

    benchmark_name: str = ""
    model_name: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_sec: float | None = None

    # Process-specific resource usage
    peak_process_cpu_pct: float | None = None
    avg_process_cpu_pct: float | None = None
    peak_process_memory_mb: float | None = None
    avg_process_memory_mb: float | None = None

    # GPU resources
    peak_gpu_memory_mb: float | None = None
    avg_gpu_memory_mb: float | None = None
    peak_gpu_utilization_pct: float | None = None
    avg_gpu_utilization_pct: float | None = None

    # IO
    total_disk_read_mb: float | None = None
    total_disk_write_mb: float | None = None

    # Topologies
    cpu_topology_summary: dict[str, Any] = field(default_factory=dict)
    gpu_topology_summary: dict[str, Any] = field(default_factory=dict)

    # Full time series (optional, for detailed analysis)
    time_series: ResourceTimeSeries | None = None

    @classmethod
    def from_tracker(
        cls,
        tracker: ProcessResourceTracker,
        benchmark_name: str = "",
        model_name: str = "",
    ) -> BenchmarkResourceUsage:
        """Create usage summary from tracker.

        Args:
            tracker: ProcessResourceTracker that has been stopped.
            benchmark_name: Name of the benchmark.
            model_name: Model being benchmarked.

        Returns:
            BenchmarkResourceUsage with aggregated statistics.
        """
        ts = tracker.get_time_series()
        ts.compute_statistics()

        # Get GPU utilization stats
        gpu_utils = [s.gpu_utilization_pct for s in ts.snapshots if s.gpu_utilization_pct is not None]

        # Get topologies from first snapshot that has them
        cpu_topo = next((s.cpu_topology for s in ts.snapshots if s.cpu_topology), None)
        gpu_topo = next((s.gpu_topology for s in ts.snapshots if s.gpu_topology), None)

        duration = None
        if ts.start_time and ts.end_time:
            duration = (ts.end_time - ts.start_time).total_seconds()

        return cls(
            benchmark_name=benchmark_name,
            model_name=model_name,
            start_time=ts.start_time or datetime.now(UTC),
            end_time=ts.end_time,
            duration_sec=duration,
            peak_process_cpu_pct=ts.peak_cpu_pct,
            avg_process_cpu_pct=ts.avg_cpu_pct,
            peak_process_memory_mb=ts.peak_memory_mb,
            avg_process_memory_mb=ts.avg_memory_mb,
            peak_gpu_memory_mb=ts.peak_gpu_memory_mb,
            avg_gpu_memory_mb=ts.avg_gpu_memory_mb,
            peak_gpu_utilization_pct=max(gpu_utils) if gpu_utils else None,
            avg_gpu_utilization_pct=sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
            total_disk_read_mb=ts.total_io_read_mb,
            total_disk_write_mb=ts.total_io_write_mb,
            cpu_topology_summary=cpu_topo.summary() if cpu_topo else {},
            gpu_topology_summary=gpu_topo.summary() if gpu_topo else {},
            time_series=ts,
        )


# Register with global registry
METRIC_REGISTRY.register("cpu_topology", CPUTopology)
METRIC_REGISTRY.register("gpu_topology", GPUTopology)
METRIC_REGISTRY.register("resource_snapshot", ResourceSnapshot)
METRIC_REGISTRY.register("resource_time_series", ResourceTimeSeries)
METRIC_REGISTRY.register("benchmark_resource_usage", BenchmarkResourceUsage)
