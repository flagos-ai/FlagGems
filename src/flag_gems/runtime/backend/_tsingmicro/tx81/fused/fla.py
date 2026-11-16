# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TXDA (TsingMicro) FLA operator implementations.
#
# `chunk_gated_delta_rule` (the GatedDeltaNet / Qwen3-Next prefill op) is the
# TXDA vendor-specific entry point.  The kernels live under the sibling ``FLA/``
# package, which mirrors the generic ``flag_gems.fused.FLA`` structure but is
# tuned for the TXDA device: TXDA's 3MB on-chip SPM lets us use larger block
# sizes (``check_shared_mem`` returns True, so ``BKV_LIST=[64,128]``) instead of
# the conservative ``[32,64]`` the generic path falls back to off-NVIDIA, and the
# autotune search is extended with ``num_warps=1`` / ``num_stages=1`` because the
# TXDA Triton backend prefers fewer warps (measured ~1.12-1.15x vs generic).

from __future__ import annotations

import functools
import inspect
import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _fmt_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, stride={tuple(value.stride())}, "
            f"dtype={value.dtype})"
        )
    return repr(value)


def log_args(func):
    """Print each argument on one line and the wall-clock execution time in ms.

    Gated behind ``FLAG_GEMS_LOG_FLA=1`` so the per-call stdout formatting/flush
    doesn't tax the serving hot path.
    """
    enabled = os.getenv("FLAG_GEMS_LOG_FLA", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        prefix = f"[{func.__name__}]"
        for name, value in bound.arguments.items():
            print(f"{prefix} {name}: {_fmt_value(value)}", flush=True)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{prefix} elapsed_ms={elapsed_ms:.3f}", flush=True)

    return wrapper


@log_args
def chunk_gated_delta_rule(
    q,
    k,
    v,
    beta,
    g,
    BT: int = 64,
    initial_state=None,
    output_final_state: bool = False,
    cu_seqlens=None,
    head_first: bool = True,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
):
    """TXDA chunk gated delta rule forward.

    Signature matches the generic ``flag_gems.fused.chunk_gated_delta_rule``
    public wrapper.  Returns ``(o, final_state)``.
    """
    logger.debug("GEMS_TSINGMICRO CHUNK_GATED_DELTA_RULE")
    from .FLA.chunk_gated_delta_rule import chunk_gated_delta_rule as _txda_impl

    return _txda_impl(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        BT=BT,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@log_args
def chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale: float,
    initial_state,
    output_final_state: bool,
    cu_seqlens=None,
):
    """TXDA low-level ``chunk_gated_delta_rule_fwd``.

    Signature matches the generic ``flag_gems.fused.FLA.chunk_gated_delta_rule_fwd``
    (arg order ``q, k, v, g, beta``).  Returns the 7-tuple
    ``(g, o, A, final_state, None, None, None)``.
    """
    logger.debug("GEMS_TSINGMICRO CHUNK_GATED_DELTA_RULE_FWD")
    from .FLA.chunk import chunk_gated_delta_rule_fwd as _txda_impl_fwd

    return _txda_impl_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
