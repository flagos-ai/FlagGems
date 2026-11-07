"""Explicit producer/consumer GEMM with ordinary shared storage."""

from dataclasses import dataclass
from functools import lru_cache

import torch
import triton

from flag_gems.runtime import torch_device_fn


@lru_cache(maxsize=1)
def _supported():
    try:
        from triton._C.libtriton import hcu
        from triton.backends.hcu.compiler import HIPOptions
        from triton.experimental.gluon.language.hcu import (
            cta_barrier,
            matrix_load_to_shared,
            structured_register_load,
            wait_local_memory,
            warp_group_id,
        )
    except ImportError:
        return False
    options = {
        "structured_register_prepack_address",
        "structured_register_unclamped_column",
        "priority_after_first_mmac",
        "buffer_store_scalar_offset",
        "interleave_global_lds_reads",
        "interleave_global_lds_reads_fences",
        "tupled_register_fences",
    }
    return (
        options <= HIPOptions.__dataclass_fields__.keys()
        and all(
            callable(op)
            for op in (
                cta_barrier,
                matrix_load_to_shared,
                structured_register_load,
                wait_local_memory,
                warp_group_id,
            )
        )
        and all(
            callable(getattr(hcu, name, None))
            for name in (
                "create_local_memory_wait",
                "create_cta_barrier",
                "has_warp_specialize",
                "create_warp_group_id",
            )
        )
    )


@dataclass(frozen=True)
class CooperativeMMConfig:
    group_m: int = 0
    tupled_fences: bool = False

    def __post_init__(self):
        if type(self.group_m) is not int or self.group_m < 0:
            raise ValueError("CTA grouping must be a nonnegative integer")
        if type(self.tupled_fences) is not bool:
            raise ValueError("tupled_fences must be a boolean")

    def supports(self, a, b, c):
        if not _supported() or any(t.ndim != 2 for t in (a, b, c)):
            return False
        m, k = a.shape
        n = b.shape[1]
        begin, end = c.data_ptr(), c.data_ptr() + 2 * m * n
        disjoint = all(
            end <= t.data_ptr() or t.data_ptr() + 2 * count <= begin
            for t, count in ((a, m * k), (b, k * n))
        )
        return (
            m > 0
            and n > 0
            and 64 <= k < 8192
            and k % 64 == 0
            and b.shape[0] == k
            and c.shape == (m, n)
            and a.stride() == (k, 1)
            and b.stride() == (1, k)
            and c.stride() == (n, 1)
            and a.dtype == b.dtype == c.dtype
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.device == b.device == c.device
            and a.device.type == "cuda"
            and all(t.data_ptr() % 16 == 0 for t in (a, b, c))
            and not c.requires_grad
            and (not c.is_inference() or torch.is_inference_mode_enabled())
            and disjoint
            and max(m, n) < 2**31
            # Each CTA rebases C in i64; only its local byte offsets are i32.
            and 512 * n < 2**31
            and (not self.group_m or triton.cdiv(m, 256) * triton.cdiv(n, 256) < 2**31)
        )


def launch_cooperative_mm(a, b, c, config):
    if not config.supports(a, b, c):
        raise ValueError("unsupported cooperative matrix storage or pipeline extent")
    from .mm_cooperative_kernel import _mm_cooperative_kernel

    m, k = a.shape
    n = b.shape[1]
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.arch != "gfx936":
            raise ValueError("cooperative GEMM currently requires gfx936")
        return _mm_cooperative_kernel[(triton.cdiv(m, 256), triton.cdiv(n, 256))](
            a,
            b,
            c,
            m,
            n,
            k,
            GROUP_M=config.group_m,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
            loop_unroll_stage="post-pipeline",
            schedule_hint="local-prefetch",
            optimize_epilogue=True,
            lds_dma_alias_workaround=False,
            disable_llvm_loop_unroll=True,
            structured_register_prepack_address=True,
            structured_register_unclamped_column=True,
            priority_after_first_mmac=True,
            buffer_store_scalar_offset=True,
            interleave_global_lds_reads=True,
            interleave_global_lds_reads_fences=True,
            tupled_register_fences=config.tupled_fences,
            rebase_buffer_store=False,
            terminal_readonly_publication=False,
        )


@lru_cache(maxsize=1)
def cooperative_mm_configs():
    """Two measured traversal/scheduling strategies for this single algorithm."""
    return (CooperativeMMConfig(), CooperativeMMConfig(4, True))
