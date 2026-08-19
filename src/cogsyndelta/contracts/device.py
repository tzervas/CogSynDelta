"""Device resolution for CPU-default / optional CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DevicePrefer = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True)
class DeviceContext:
    """Resolved compute device. Construct only via resolve()."""

    device: torch.device
    prefer: DevicePrefer

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def name(self) -> str:
        if self.device.type == "cpu":
            return "cpu"
        index = self.device.index if self.device.index is not None else 0
        return f"cuda:{index}"

    @classmethod
    def resolve(cls, prefer: DevicePrefer = "auto") -> DeviceContext:
        """Resolve prefer into a concrete torch.device.

        - cpu: always CPU
        - cuda: require CUDA or raise
        - auto: CUDA if available else CPU
        """
        if prefer == "cpu":
            return cls(device=torch.device("cpu"), prefer=prefer)
        if prefer == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
            return cls(device=torch.device("cuda:0"), prefer=prefer)
        # auto
        if torch.cuda.is_available():
            return cls(device=torch.device("cuda:0"), prefer=prefer)
        return cls(device=torch.device("cpu"), prefer=prefer)

    def tensor(self, data: torch.Tensor) -> torch.Tensor:
        """Move tensor to this device."""
        return data.to(self.device)

    def module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Move module to this device."""
        return module.to(self.device)
