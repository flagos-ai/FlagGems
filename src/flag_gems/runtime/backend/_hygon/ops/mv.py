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

"""Hygon Triton/TLE implementation of ``aten.mv``.

The reduction is kept as a vectorized Triton reduction because the current
gfx936 lowering of ``tl.dot`` adds shared-memory movement for GEMV instead of
using a matrix instruction.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


_SMALL_MV_CONFIGS = {
    (256, 1, 1, 1, 2),
    (256, 1, 2, 2, 2),
    (256, 2, 8, 1, 2),
    (1024, 2, 2, 1, 2),
    (2048, 1, 4, 1, 2),
    (2048, 2, 4, 1, 3),
    (4096, 1, 4, 1, 2),
    (4096, 2, 8, 2, 2),
    (4096, 4, 4, 2, 2),
}

_MID_MV_CONFIGS = {
    (256, 1, 1, 1, 2),
    (256, 1, 1, 2, 2),
    (256, 1, 1, 3, 2),
    (256, 1, 2, 1, 2),
    (256, 1, 2, 2, 2),
    (512, 1, 1, 1, 2),
    (512, 1, 1, 2, 2),
    (512, 1, 1, 3, 2),
    (1024, 1, 1, 1, 2),
    (1024, 1, 2, 1, 2),
    (2048, 1, 1, 1, 2),
    (2048, 1, 2, 1, 2),
    (2048, 2, 4, 2, 2),
    (2048, 2, 8, 1, 2),
    (2048, 4, 2, 1, 2),
    (2048, 4, 8, 1, 2),
    (4096, 1, 4, 1, 2),
    (4096, 2, 8, 2, 2),
    (4096, 4, 4, 2, 2),
}

_WIDE_MV_CONFIGS = {
    (512, 1, 1, 1, 2),
    (1024, 1, 2, 1, 2),
    (2048, 1, 4, 1, 2),
    (2048, 1, 4, 2, 2),
    (2048, 2, 2, 1, 2),
    (2048, 2, 4, 1, 2),
    (2048, 4, 4, 1, 2),
    (4096, 1, 4, 1, 2),
}


def _prune_mv_configs(configs, named_args, **kwargs):
    """Remove configurations that are a poor fit for the GEMV occupancy class.

    The expanded Hygon search space is intentionally broad so it can cover
    arbitrary layouts and reduction lengths.  For the contiguous model GEMVs,
    profiling shows that narrower output tiles and smaller workgroups avoid
    register/LDS overhead.  Keep this fast path limited to the dtypes and two
    reduction lengths measured here; all other shapes and layouts retain the
    complete search space.  FlagTune deliberately bypasses this shortcut so
    its proposer can rank the full YAML configuration space.
    """

    # Do not encode a second, hand-picked search space in FlagTune.  In that
    # mode the expanded YAML candidates are the contract and FlagTune owns the
    # trade-off between tuning time and the selected kernel configuration.
    if (
        runtime.resolve_tuning_mode("mv", supports_cost_model=False)
        is not runtime.TuningMode.DEFAULT
    ):
        return configs

    try:
        matrix = named_args["A"]
        dtype = matrix.dtype
        n = int(named_args["N"])
        m = int(named_args["M"])
        strides = tuple(
            int(named_args[name])
            for name in ("stride_an", "stride_am", "stride_bm", "stride_cn")
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return configs

    # The measured configuration is valid for row-major A, contiguous B and
    # contiguous output.  Strided views continue to use the full tuner space.
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return configs
    if strides != (m, 1, 1, 1) or m not in (2048, 4096):
        return configs

    # The compact sets are the winning families from sweeps across the real
    # model shapes for fp16, bf16 and fp32.
    if n < 64:
        candidates = _SMALL_MV_CONFIGS
    elif n <= 2048:
        candidates = _MID_MV_CONFIGS
    else:
        candidates = _WIDE_MV_CONFIGS

    pruned = []
    for config in configs:
        meta = config.kwargs
        key = (
            meta.get("BLOCK_M"),
            meta.get("BLOCK_N"),
            config.num_warps,
            meta.get("LOOP_STAGES"),
            config.num_stages,
        )
        if key not in candidates or meta.get("BLOCK_N", n + 1) > n:
            continue
        pruned.append(config)

    # Keep tuning functional if a future config schema changes underneath the
    # heuristic.  A conservative fallback is preferable to rejecting a valid
    # kernel launch.
    return pruned or configs


@libentry()
@libtuner(
    # Keep the default config fallback for callers that do not opt into
    # FlagTune.  The real-model acceptance path sets FLAGTUNE_INCLUDE=mv,
    # which switches this tuner to mv_hygon_expand.yaml at first launch.
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N", "stride_an", "stride_am", "stride_bm"],
    strategy=runtime.common.DEFAULT_STRATEGIES["mv_hygon"],
    benchmark_mode="event",
    warmup=5,
    rep=50,
    prune_configs_by={"early_config_prune": _prune_mv_configs},
    flagtune_op_name="mv",
    flagtune_expand_op_name="mv_hygon",
)
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M: tl.constexpr,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    """Accumulate K tiles before reducing each output row.

    ``A`` has shape ``[N, M]`` and ``B`` has shape ``[M]``.  Each program
    computes ``BLOCK_N`` output rows and walks the reduction dimension in
    ``BLOCK_M`` element tiles.
    """

    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_m = tl.arange(0, BLOCK_M)
    n_mask = offset_n < N

    # Accumulate corresponding elements across K tiles before reducing.  This
    # avoids a cross-warp reduction on every iteration of a multi-tile row.
    a_ptrs = A + offset_n[:, None] * stride_an + offset_m[None, :] * stride_am
    b_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # Use Triton's loop iterator so LOOP_STAGES is emitted as a loop
    # pipeline attribute.  The JIT parser accepts keyword arguments on a
    # builtin ``range`` call but ignores them when lowering that iterator.
    for m in tl.range(0, M, BLOCK_M, num_stages=LOOP_STAGES):
        m_mask = m + offset_m < M
        a = tl.load(
            a_ptrs,
            mask=n_mask[:, None] & m_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(b_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b[None, :]
        a_ptrs += BLOCK_M * stride_am
        b_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    c_ptrs = C + offset_n * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=n_mask)


def mv(inp, vec):
    """Hygon ``mv`` entry point matching the ATen signature."""

    logger.debug("GEMS_HYGON MV")
    assert inp.dim() == 2 and vec.dim() == 1, "mv expects a matrix and a vector"
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"

    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)

    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
        )

    return out


__all__ = ["mv", "mv_kernel"]
