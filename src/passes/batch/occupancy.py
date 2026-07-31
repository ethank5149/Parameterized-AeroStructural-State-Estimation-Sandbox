"""CUDA occupancy model for the batch layer (Paper I, §5.2 / V8).

Paper I's V8 asks for *achieved* occupancy alongside throughput. Achieved
occupancy is a hardware counter and needs a profiler (Nsight Compute),
which is not available in this environment. What *is* computable, exactly
and without a profiler, is **theoretical occupancy**: the standard CUDA
occupancy model, evaluated from the compiled kernel's register and
shared-memory footprint together with the device's per-SM limits.

That is a real number, not an estimate — it is the same arithmetic the
CUDA Occupancy Calculator performs — and it bounds achieved occupancy
from above. Reporting it, together with the throughput-saturation curve
that is achieved occupancy's externally observable consequence, closes
most of what V8 asks for; the residual gap is stated rather than papered
over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["OccupancyReport", "device_limits", "theoretical_occupancy"]


@dataclass(frozen=True)
class OccupancyReport:
    """Theoretical occupancy of one kernel at one block size."""

    threads_per_block: int
    registers_per_thread: int
    shared_bytes_per_block: int
    active_blocks_per_sm: int
    active_warps_per_sm: int
    max_warps_per_sm: int
    limiter: str
    """Which resource binds: ``"registers"``, ``"shared_memory"``,
    ``"warps"`` or ``"blocks"``."""

    @property
    def occupancy(self) -> float:
        """Active warps per SM divided by the architectural maximum."""
        return self.active_warps_per_sm / self.max_warps_per_sm


def device_limits(device: int = 0) -> dict[str, int]:
    """Per-SM architectural limits of a CUDA device."""
    import cupy.cuda.runtime as runtime

    props = runtime.getDeviceProperties(device)
    return {
        "multiprocessors": int(props["multiProcessorCount"]),
        "max_threads_per_sm": int(props["maxThreadsPerMultiProcessor"]),
        "registers_per_sm": int(props["regsPerMultiprocessor"]),
        "shared_bytes_per_sm": int(props["sharedMemPerMultiprocessor"]),
        "warp_size": int(props["warpSize"]),
        "max_threads_per_block": int(props["maxThreadsPerBlock"]),
        "compute_capability_major": int(props["major"]),
        "compute_capability_minor": int(props["minor"]),
    }


def theoretical_occupancy(
    kernel: Any,
    threads_per_block: int,
    device: int = 0,
    max_blocks_per_sm: int = 16,
    register_allocation_unit: int = 256,
) -> OccupancyReport:
    """Theoretical occupancy from the compiled kernel's resource use.

    Parameters
    ----------
    kernel:
        A compiled ``cupy.RawKernel`` (its ``attributes`` supply the
        register and shared-memory footprint).
    threads_per_block:
        Launch block size.
    max_blocks_per_sm:
        Architectural resident-block limit per SM (16 on Ampere).
    register_allocation_unit:
        Register file allocation granularity, in registers per warp.
    """
    if threads_per_block < 1:
        raise ValueError(f"threads_per_block must be >= 1, got {threads_per_block}")
    limits = device_limits(device)
    warp = limits["warp_size"]
    if threads_per_block > limits["max_threads_per_block"]:
        raise ValueError(
            f"threads_per_block {threads_per_block} exceeds the device limit "
            f"{limits['max_threads_per_block']}"
        )

    attributes = kernel.attributes
    regs = int(attributes["num_regs"])
    shared = int(attributes["shared_size_bytes"])

    warps_per_block = -(-threads_per_block // warp)  # ceil
    max_warps = limits["max_threads_per_sm"] // warp

    # registers are allocated per warp, rounded to the allocation unit
    regs_per_warp = regs * warp
    rounded = -(-regs_per_warp // register_allocation_unit) * register_allocation_unit
    by_registers = (
        limits["registers_per_sm"] // (rounded * warps_per_block)
        if rounded > 0
        else max_blocks_per_sm
    )
    by_shared = (
        limits["shared_bytes_per_sm"] // shared if shared > 0 else max_blocks_per_sm
    )
    by_warps = max_warps // warps_per_block
    candidates = {
        "registers": by_registers,
        "shared_memory": by_shared,
        "warps": by_warps,
        "blocks": max_blocks_per_sm,
    }
    active_blocks = min(candidates.values())
    limiter = min(candidates, key=lambda k: candidates[k])
    return OccupancyReport(
        threads_per_block=threads_per_block,
        registers_per_thread=regs,
        shared_bytes_per_block=shared,
        active_blocks_per_sm=int(active_blocks),
        active_warps_per_sm=int(active_blocks * warps_per_block),
        max_warps_per_sm=int(max_warps),
        limiter=limiter,
    )
