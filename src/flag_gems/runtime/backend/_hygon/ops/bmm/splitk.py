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


"""Shared split-K path for GEMMs with too few output tiles to fill the device."""

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTunerRunMode


@triton.jit
def _partials(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BATCH: tl.constexpr,
    SPLITS: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    split = tl.program_id(2)
    gn = tl.cdiv(N, TN)
    rm = tile // gn * TM + tl.arange(0, TM)
    rn = tile % gn * TN + tl.arange(0, TN)
    rk = tl.arange(0, TK)
    segment: tl.constexpr = triton.cdiv(K, SPLITS * TK) * TK
    start = split * segment
    acc = tl.zeros((TM, TN), tl.float32)
    for offset in tl.range(0, segment, TK, num_stages=2):
        kk = start + offset + rk
        a = tl.load(
            A
            + batch.to(tl.int64) * AB
            + rm[:, None].to(tl.int64) * AM
            + kk[None, :].to(tl.int64) * AK,
            (rm[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            B
            + batch.to(tl.int64) * BB
            + kk[:, None].to(tl.int64) * BK
            + rn[None, :].to(tl.int64) * BN,
            (kk[:, None] < K) & (rn[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    offsets = (
        (split * BATCH + batch).to(tl.int64) * M * N
        + rm[:, None].to(tl.int64) * N
        + rn[None, :]
    )
    tl.store(P + offsets, acc, (rm[:, None] < M) & (rn[None, :] < N))


@libentry(cache_binding=True)
@libtuner(
    configs=runtime.get_tuned_config("bmm_splitk"),
    key=["BATCH", "M", "N", "K", "AB", "AM", "AK", "BB", "BK", "BN", "SPLITS"],
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def _small_partials(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BATCH: tl.constexpr,
    SPLITS: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
):
    _partials(A, B, P, M, N, K, AB, AM, AK, BB, BK, BN, BATCH, SPLITS, TM, TN, TK)


@triton.jit
def _finish_splitk(
    P,
    O,
    Bias,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    size: tl.constexpr = BATCH * M * N
    split = tl.arange(0, SPLITS)
    values = tl.load(
        P + split[:, None].to(tl.int64) * size + idx[None, :],
        idx[None, :] < size,
        other=0,
    )
    result = tl.sum(values, axis=0)
    batch = idx // (M * N)
    row = idx // N % M
    col = idx % N
    result *= ALPHA
    if HAS_BIAS and BETA != 0:
        result += BETA * tl.load(
            Bias
            + batch.to(tl.int64) * CB
            + row.to(tl.int64) * CM
            + col.to(tl.int64) * CN,
            idx < size,
            other=0,
        ).to(tl.float32)
    tl.store(
        O + batch.to(tl.int64) * OB + row.to(tl.int64) * OM + col.to(tl.int64) * ON,
        result,
        idx < size,
    )


def _batch_strides(tensor):
    strides = tensor.stride()
    return strides if tensor.ndim == 3 else (0, *strides)


def _reduction_constants(out, batch, m, n, splits, bias, alpha, beta, bias_strides):
    if bias_strides is None:
        bias_strides = _batch_strides(bias) if bias is not None else (0, 0, 0)
    return (
        m,
        n,
        batch,
        *_batch_strides(out),
        splits,
        256,
        *bias_strides,
        alpha,
        beta,
        bias is not None,
    )


def _launch_reduction(partials, out, bias, args, grid):
    return _finish_splitk.run(
        partials,
        out,
        bias,
        *args,
        grid=grid,
        warmup=False,
        num_warps=4,
        num_stages=1,
    )


def launch_splitk(a, b, out, config):
    batch, m, k = a.shape
    n = b.shape[2]
    splits, tm, tn, tk = config
    partials = torch.empty((splits, batch, m, n), dtype=torch.float32, device=a.device)
    _partials[(triton.cdiv(m, tm) * triton.cdiv(n, tn), batch, splits)](
        a,
        b,
        partials,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        batch,
        splits,
        tm,
        tn,
        tk,
        num_warps=4,
        num_stages=2,
        allow_flush_denorm=True,
    )
    args = _reduction_constants(out, batch, m, n, splits, None, 1, 0, None)
    _launch_reduction(partials, out, out, args, ((batch * m * n + 255) // 256, 1, 1))


def small_splitk_count(batch, m, n, k, dtype):
    """Return the split count, or zero if the ordinary GEMM should run."""
    if (
        min(batch, m, n, k) <= 0
        or dtype not in (torch.float16, torch.bfloat16)
        or k < 1024
        or min(m, n) > 128
        or max(m, n) > 4096
    ):
        return 0
    tiles = batch * ((m + 31) // 32) * ((n + 63) // 64)
    if tiles >= 128:
        return 0
    requested = (127 + tiles) // tiles
    splits = min(32, 1 << (requested - 1).bit_length())
    splits = min(splits, (1 << (k - 1).bit_length()) // 256)
    return splits if splits >= 2 else 0


_SMALL_SPLITK_LAUNCHES = {}


def _small_splitk_grid(meta, batch, m, n, splits):
    tm, tn = meta["TM"], meta["TN"]
    return (((m + tm - 1) // tm) * ((n + tn - 1) // tn), batch, splits)


def _splitk_launch_key(tensors, partial_args, finish_args, alpha, beta):
    tuner = _small_partials.fn
    specialize = _small_partials._tensor_spec_fn
    if (
        specialize is None
        or tuner._run_mode != LibTunerRunMode.NORMAL
        or getattr(tuner, "shared_config_pre_hook", None) is not None
        or type(alpha) not in (int, float, bool)
        or type(beta) not in (int, float, bool)
        or any(type(t) is not torch.Tensor for t in tensors)
    ):
        return None
    signatures = tuple((t.device, t.dtype, specialize(t, align=True)) for t in tensors)
    return signatures, partial_args, finish_args, type(alpha), type(beta)


def launch_small_splitk(
    a, b, out, bias=None, alpha=1, beta=0, *, splits=None, bias_strides=None
):
    """Allocate fresh FP32 partials and apply the epilogue in the reduction."""
    m, k = a.shape[-2:]
    n = b.shape[-1]
    batch = a.shape[0] if a.ndim == 3 else 1
    if splits is None:
        splits = small_splitk_count(batch, m, n, k, a.dtype)
    if not splits:
        return False
    partials = torch.empty((splits, batch, m, n), dtype=torch.float32, device=a.device)
    args = (m, n, k, *_batch_strides(a), *_batch_strides(b), batch, splits)
    finish_args = _reduction_constants(
        out, batch, m, n, splits, bias, alpha, beta, bias_strides
    )
    finish_grid = ((batch * m * n + 255) // 256, 1, 1)
    bias_arg = bias if bias is not None else out
    key = _splitk_launch_key(
        (a, b, partials, out, bias_arg), args, finish_args, alpha, beta
    )
    plan = _SMALL_SPLITK_LAUNCHES.get(key)
    if plan is not None:
        first, second, tile = plan
        stream = driver.active.get_current_stream(a.device.index)
        first(a, b, partials, *args, *tile, stream=stream)
        second(partials, out, bias_arg, *finish_args, stream=stream)
        return True

    first, config = _small_partials.run(
        a,
        b,
        partials,
        *args,
        grid=lambda meta: _small_splitk_grid(meta, batch, m, n, splits),
        warmup=False,
    )
    second = _launch_reduction(partials, out, bias_arg, finish_args, finish_grid)
    if key is not None and all(c.pre_hook is None for c in _small_partials.fn.configs):
        if len(_SMALL_SPLITK_LAUNCHES) >= 128:
            _SMALL_SPLITK_LAUNCHES.clear()
        # Keep only launchers and compile-time tiles. Every call passes current
        # tensors and its stream, including a freshly allocated partial buffer.
        grid = _small_splitk_grid(config, batch, m, n, splits)
        tile = (config["TM"], config["TN"], config["TK"])
        _SMALL_SPLITK_LAUNCHES[key] = (first[grid], second[finish_grid], tile)
    return True
