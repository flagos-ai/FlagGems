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

"""Hygon Triton/TLE MV, including the matrix-instruction candidate
that reuses the CTA-local split-K GEMM implementation.
"""

import logging
import statistics
import threading
from dataclasses import dataclass
from functools import cache, lru_cache
from math import inf, isfinite

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

from .gemm_utils import (
    benchmark_gemm_candidate,
    capture_gemm_fallback,
    gemm_replay_benchmarker,
    try_gemm_candidate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupedMVConfig:
    block_n: int
    block_k: int


_GROUPED_MV_CONFIGS = tuple(
    GroupedMVConfig(block_n, block_k)
    for block_n, block_k in ((1, 1024), (2, 1024), (2, 2048), (4, 1024))
)


def grouped_mv_candidates(a, b, compute_units):
    """Offer vector reduction for underfilled MMAC rows and contiguous K."""
    m, k = a.shape
    n = b.shape[1]
    if not (1 < m < 8 and b.stride() == (1, k)):
        return ()
    return tuple(
        config
        for config in _GROUPED_MV_CONFIGS
        if config.block_n <= n
        and config.block_k <= k
        and m * triton.cdiv(n, config.block_n) <= 4 * compute_units
    )


@libentry(cache_miss_fallback=capture_gemm_fallback)
@libtuner(
    # Keep the default config fallback for callers that do not opt into
    # FlagTune.  The real-model acceptance path sets FLAGTUNE_INCLUDE=mv,
    # which selects the MV section of mm_hygon_expand.yaml at first launch.
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N", "stride_an", "stride_am", "stride_bm"],
    strategy=runtime.common.DEFAULT_STRATEGIES["mv_hygon"],
    benchmark_mode="replay",
    warmup=5,
    rep=10,
    replay_refinement=(25, 100),
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
    GROUP_VEC_STRIDE: tl.constexpr = 0,
    GROUP_OUT_STRIDE: tl.constexpr = 0,
):
    """Accumulate K tiles before reducing each output row.

    ``A`` has shape ``[N, M]`` and ``B`` has shape ``[M]``.  Each program
    computes ``BLOCK_N`` output rows and walks the reduction dimension in
    ``BLOCK_M`` element tiles.
    """

    if GROUP_VEC_STRIDE:
        row = ext.program_id(1)
        B += row * GROUP_VEC_STRIDE
        C += row * GROUP_OUT_STRIDE
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


@lru_cache(maxsize=1)
def _fixed_mv_entry():
    return libentry()(mv_kernel.fn.fn)


def launch_grouped_mv(a, b, c, config):
    """Run one GEMV per A row against the shared matrix B transpose."""
    m, k = a.shape
    n = b.shape[1]
    with torch_device_fn.device(a.device):
        return _fixed_mv_entry()[(triton.cdiv(n, config.block_n), m)](
            b,
            a,
            c,
            n,
            k,
            b.stride(1),
            b.stride(0),
            a.stride(1),
            c.stride(1),
            BLOCK_N=config.block_n,
            BLOCK_M=config.block_k,
            LOOP_STAGES=1,
            GROUP_VEC_STRIDE=a.stride(0),
            GROUP_OUT_STRIDE=c.stride(0),
            num_warps=4,
        )


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
    _dispatch_hygon_mv(inp, vec, out)
    return out


def _dispatch_hygon_mv(inp, vec, out):
    """Choose the vector reduction or the MMAC candidate."""
    mv_kernel._apply_flagtune()
    tuner = mv_kernel.fn
    if tuner._run_mode == "normal":
        tuning_key = (tuner.config_table_name, id(tuner.configs),
                      tuple(tuner._benchmark_protocol))
        if try_mv_mmac(inp, vec, out,
                       lambda output: _launch_ordinary_mv(inp, vec, output),
                       tuning_key=tuning_key):
            return
    _launch_ordinary_mv(inp, vec, out)


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


# MV matrix-instruction candidate and replay selection.


_mv_mmac_choices = {}
_mv_mmac_measurements = {}
_mv_mmac_lock = threading.RLock()


@cache
def _mmac_config():
    from .mm import LocalSplitMMConfig

    return LocalSplitMMConfig((16, 16), num_stages=2)


def _mv_mmac_eligible(inp, vec, out):
    from .mm import local_split_mm_configs

    n, k = inp.shape
    config = _mmac_config()
    if (
        vec.shape != (k,)
        or out.shape != (n,)
        or vec.stride() != (1,)
        or out.stride() != (1,)
        or config not in local_split_mm_configs()
    ):
        return False
    if not config.supports(vec.view(1, k), inp.t(), out.view(1, n)):
        return False
    with torch_device_fn.device(inp.device):
        target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx936"


def _launch_mmac(inp, vec, out):
    from .mm import launch_local_split_mm

    n, k = inp.shape
    return launch_local_split_mm(
        vec.view(1, k), inp.t(), out.view(1, n), _mmac_config()
    )


def try_mv_mmac(inp, vec, out, ordinary_with_output, *, tuning_key):
    if not _mv_mmac_eligible(inp, vec, out):
        return False
    key = (
        inp.device,
        inp.dtype,
        vec.dtype,
        out.dtype,
        _mmac_config(),
        tuple(inp.shape),
        tuple(inp.stride()),
        tuple(vec.stride()),
        tuple(out.stride()),
        tuning_key,
        "mv-mmac-ordinary-tl-family-v3",
    )
    with torch_device_fn.device(inp.device):
        choice = _mv_mmac_choices.get(key)
        if choice is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            with _mv_mmac_lock:
                choice = _mv_mmac_choices.get(key)
                if choice is None:
                    # Resolve the unchanged Default/Expanded tuner first.
                    ordinary_with_output(out)
                    succeeded, _ = try_gemm_candidate(_launch_mmac, inp, vec, out)
                    if not succeeded:
                        _mv_mmac_choices[key] = 0
                        return True
                    def compare(bench):
                        samples = ([], [])
                        for repeat in range(3):
                            for index in (0, 1) if repeat % 2 == 0 else (1, 0):

                                def fresh_output():
                                    output = torch.empty_like(out)
                                    if index:
                                        _launch_mmac(inp, vec, output)
                                    else:
                                        ordinary_with_output(output)
                                    return output

                                samples[index].append(
                                    benchmark_gemm_candidate(
                                        bench, fresh_output, optional=index != 0,
                                    )
                                )
                        return tuple(
                            statistics.median(x) if all(map(isfinite, x)) else inf
                            for x in samples
                        )

                    screened = compare(gemm_replay_benchmarker(5, 10, 3))
                    medians = screened
                    if all(map(isfinite, screened)) and max(screened) <= 1.05 * min(screened):
                        medians = compare(gemm_replay_benchmarker(25, 100, 10))
                    choice = min(range(2), key=medians.__getitem__)
                    _mv_mmac_measurements[key] = {
                        "variants": ("ordinary", "mmac_k256_bn16_s2"),
                        "screened_ms": screened,
                        "allocating_ms": medians,
                        "choice": choice,
                    }
                    _mv_mmac_choices[key] = choice
                    logger.debug(
                        "Hygon MV family choice=%s allocating=%s ms", choice, medians
                    )
        if choice:
            succeeded, _ = try_gemm_candidate(_launch_mmac, inp, vec, out)
            if succeeded:
                return True
            with _mv_mmac_lock:
                _mv_mmac_choices[key] = 0
                _mv_mmac_measurements[key]["choice"] = 0
        ordinary_with_output(out)
    return True


__all__ = ["mv", "mv_kernel"]
