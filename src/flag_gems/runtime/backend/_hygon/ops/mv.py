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


@libentry()
@libtuner(
    # Keep the default config fallback for callers that do not opt into
    # FlagTune.  The real-model acceptance path sets FLAGTUNE_INCLUDE=mv,
    # which switches this tuner to mv_hygon_expand.yaml at first launch.
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N", "stride_an", "stride_am", "stride_bm"],
    strategy=runtime.common.DEFAULT_STRATEGIES["mv_hygon"],
    benchmark_mode="replay",
    warmup=5,
    rep=50,
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

    out = torch.empty((inp.shape[0],), device=inp.device, dtype=inp.dtype)
    return _mv_out(inp, vec, out)


def _mv_out(inp, vec, out):
    """Launch the same reduction into an existing, possibly strided vector."""
    assert inp.dim() == 2 and vec.dim() == out.dim() == 1
    N, M = inp.shape
    assert vec.shape[0] == M and out.shape[0] == N
    from .mv_mmac import try_mv_mmac

    mv_kernel._apply_flagtune()
    tuner = mv_kernel.fn
    if tuner._run_mode == "normal":
        tuning_key = (tuner.config_table_name, id(tuner.configs),
                      tuple(tuner._benchmark_protocol))
        if try_mv_mmac(inp, vec, out,
                       lambda output: _launch_ordinary_mv(inp, vec, output),
                       tuning_key=tuning_key):
            return out
    _launch_ordinary_mv(inp, vec, out)
    return out


def _launch_ordinary_mv(inp, vec, out):
    N, M = inp.shape
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
