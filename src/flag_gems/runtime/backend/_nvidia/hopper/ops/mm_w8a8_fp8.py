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

from collections import OrderedDict
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_EXPAND_YAML = str(Path(__file__).resolve().parent.parent / "mm_hopper_expand.yaml")
_TMA_SPACE = "mm_w8a8_fp8_block_scaled"
_LOAD_SPACE = "mm_w8a8_fp8_block_scaled_splitk"
_TUNE_KEY = ["M", "N", "K", "stride_am", "stride_bk"]
_LAUNCH_CACHE = OrderedDict()


def _launch_tuned(kernel, grid, args, pre_hook=None):
    """Reuse the tuner's compiled result, never tensor contents or fixed tiles."""
    tuner = kernel.fn
    if getattr(tuner._run_mode, "value", "normal") != "normal":
        _LAUNCH_CACHE.clear()
        return kernel[grid](*args)
    mode = runtime.resolve_tuning_mode("mm_w8a8_fp8", supports_cost_model=False)
    if mode is not tuner._flagtune_mode:
        kernel._apply_flagtune()
    key = [kernel, mode, id(tuner.configs), tuner._benchmark_protocol]
    for arg in args:
        if isinstance(arg, TensorDescriptor):
            key.append(
                (
                    arg.base.dtype,
                    arg.base.device,
                    tuple(arg.shape),
                    tuple(arg.strides),
                    arg.base.data_ptr() % 16,
                )
            )
        elif isinstance(arg, torch.Tensor):
            # Shapes and strides are explicit kernel arguments on the load path.
            key.append((arg.dtype, arg.device, arg.data_ptr() % 16))
        else:
            key.append(arg)
    key = tuple(key)
    cached = _LAUNCH_CACHE.get(key)
    if cached is None:
        compiled, meta = kernel[grid](*args)
        names = tuple(kernel.signature.parameters)[len(args) :]
        tail = tuple(meta[name] for name in names)
        launch_grid = (tuple(grid(meta)) + (1, 1))[:3]
        cached = (compiled, meta, tail, launch_grid)
        _LAUNCH_CACHE[key] = cached
        if len(_LAUNCH_CACHE) > 128:
            _LAUNCH_CACHE.popitem(last=False)
    else:
        _LAUNCH_CACHE.move_to_end(key)
        compiled, meta, tail, launch_grid = cached
        if pre_hook is not None:
            pre_hook({**dict(zip(kernel.arg_names, args)), **meta})
        compiled[launch_grid](*args, *tail)
    return cached[:2]


def _set_tma_blocks(args):
    BLOCK_M = args["BLOCK_M"]
    BLOCK_N = args["BLOCK_N"]
    BLOCK_K = args["BLOCK_K"]
    args["A"].block_shape = [max(64, BLOCK_M), max(32, BLOCK_K)]
    args["B"].block_shape = [max(16, BLOCK_N), max(32, BLOCK_K)]
    args["C"].block_shape = [max(64, BLOCK_M), max(16, BLOCK_N)]


_TMA_CONFIGS = runtime.ops_get_configs(
    _TMA_SPACE, pre_hook=_set_tma_blocks, yaml_path=_EXPAND_YAML
)
_LOAD_CONFIGS = runtime.ops_get_configs(_LOAD_SPACE, yaml_path=_EXPAND_YAML)
_MAX_SPLIT_K = max(config.kwargs["SPLIT_K"] for config in _LOAD_CONFIGS)


def _prune_load_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    return [
        config
        for config in configs
        if config.kwargs["SPLIT_K"] <= args["SPLIT_CAPACITY"]
        and (
            not args["PRECISE"]
            or (config.kwargs["BLOCK_K"] == 32 and config.num_stages == 2)
        )
    ]


@libentry()
@libtuner(
    configs=_TMA_CONFIGS[:1],
    key=_TUNE_KEY,
    warmup=5,
    rep=10,
    flagtune_op_name="mm_w8a8_fp8",
    flagtune_expand_op_name=_TMA_SPACE,
    flagtune_yaml_path=_EXPAND_YAML,
    flagtune_pre_hook=_set_tma_blocks,
)
@triton.jit
def _mm_w8a8_fp8_tma(
    A,
    B,
    C,
    SA,
    SB,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_bk,
    AS: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    BM: tl.constexpr = max(64, BLOCK_M)
    BN: tl.constexpr = max(16, BLOCK_N)
    BK: tl.constexpr = max(32, BLOCK_K)
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pm = group_id * GROUP_M + pid % group_size
    pn = (pid % width) // group_size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    sa = tl.load(SA + rm * AS, rm < M, 0.0)
    sb = tl.load(SB + rn * BS, rn < N, 0.0)
    acc = tl.zeros((BM, BN), tl.float32)
    # Per-tensor/row/column scales are loaded once, outside the K loop.
    for kk in range(tl.cdiv(K, BK)):
        a = A.load([pm * BM, kk * BK])
        b = B.load([pn * BN, kk * BK]).T
        acc = tl.dot(a, b, acc=acc)
    acc = acc * sa[:, None] * sb[None, :]
    C.store([pm * BM, pn * BN], acc.to(C.dtype))


@libentry()
@libtuner(
    configs=_LOAD_CONFIGS[:1],
    key=_TUNE_KEY,
    warmup=5,
    rep=10,
    prune_configs_by={"early_config_prune": _prune_load_configs},
    flagtune_op_name="mm_w8a8_fp8",
    flagtune_expand_op_name=_LOAD_SPACE,
    flagtune_yaml_path=_EXPAND_YAML,
)
@triton.jit
def _mm_w8a8_fp8_load(
    A,
    B,
    C,
    Partial,
    SA,
    SB,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_bk,
    AK: tl.constexpr,
    stride_bn: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    PRECISE: tl.constexpr,
    SPLIT_CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # FlagTree may shrink a proposed tile; retain native FP8 MMA dimensions.
    BM: tl.constexpr = max(64, BLOCK_M)
    BN: tl.constexpr = max(16, BLOCK_N)
    BK: tl.constexpr = max(32, BLOCK_K)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    if PRECISE:
        # A full FP8 WGMMA chunk can lose low bits before FP32 promotion.
        # Zero-pad single products to the native K=32 instruction and force
        # separate FP32 additions; never convert the operands to FP16.
        for kk in range(tl.cdiv(K, SPLIT_K)):
            idx = kk * SPLIT_K + split + rk
            a = tl.load(
                A + rm[:, None] * stride_am + idx[None, :] * AK,
                (rm[:, None] < M) & (rk[None, :] == 0) & (idx[None, :] < K),
                0.0,
            )
            b = tl.load(
                B + idx[:, None] * stride_bk + rn[None, :] * stride_bn,
                (rn[None, :] < N) & (rk[:, None] == 0) & (idx[:, None] < K),
                0.0,
            )
            partial = tl.dot(a, b, max_num_imprecise_acc=32)
            acc = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[acc, partial],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
    else:
        for block in range(tl.cdiv(K, BK * SPLIT_K)):
            idx = (block * SPLIT_K + split) * BK + rk
            a = tl.load(
                A + rm[:, None] * stride_am + idx[None, :] * AK,
                (rm[:, None] < M) & (idx[None, :] < K),
                0.0,
            )
            b = tl.load(
                B + idx[:, None] * stride_bk + rn[None, :] * stride_bn,
                (rn[None, :] < N) & (idx[:, None] < K),
                0.0,
            )
            acc = tl.dot(a, b, acc=acc, max_num_imprecise_acc=32)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT_K == 1:
        sa = tl.load(SA + rm * AS, rm < M, 0.0)
        sb = tl.load(SB + rn * BS, rn < N, 0.0)
        acc = acc * sa[:, None] * sb[None, :]
        tl.store(C + rm[:, None] * CM + rn[None, :] * CN, acc, mask)
    else:
        tl.store(Partial + split * M * N + rm[:, None] * N + rn[None, :], acc, mask)


@triton.jit
def _mm_w8a8_fp8_reduce(
    Partial,
    C,
    SA,
    SB,
    M,
    N,
    CM: tl.constexpr,
    CN: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLIT_K):
        value += tl.load(Partial + split * M * N + idx, idx < M * N, 0.0)
    sa = tl.load(SA + (idx // N) * AS, idx < M * N, 0.0)
    sb = tl.load(SB + (idx % N) * BS, idx < M * N, 0.0)
    tl.store(C + (idx // N) * CM + (idx % N) * CN, value * sa * sb, idx < M * N)


@triton.jit
def _mm_w8a8_fp8_scaled_epilogue(
    Acc,
    Out,
    Bias,
    M: tl.constexpr,
    N: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Acc + x, x < M * N, 0.0)
    value += tl.load(Bias + (x % N) * BIAS_STRIDE, x < M * N, 0.0).to(tl.float32)
    tl.store(Out + (x // N) * OM + (x % N) * ON, value, x < M * N)


def _tma_aligned(tensor):
    return (
        tensor.stride(1) == 1
        and tensor.data_ptr() % 16 == 0
        and tensor.stride(0) * tensor.element_size() % 16 == 0
    )


def _scaled_matmul(a, b, c, sa, sb):
    m, k = a.shape
    n = b.shape[1]
    as_ = 0 if sa.numel() == 1 else sa.stride(0)
    bs = 0 if sb.numel() == 1 else sb.stride(0)
    precise = c.dtype not in (torch.float16, torch.bfloat16) or k > 4096
    # Small matrices avoid host-side TMA descriptor setup.
    use_tma = (
        not precise
        and not (m <= 512 and n <= 512 and k <= 512)
        and k >= 32
        and all(_tma_aligned(t) for t in (a, b.T, c))
    )
    if use_tma:
        ad = TensorDescriptor(a, a.shape, a.stride(), [1, 1])
        bt = b.T
        bd = TensorDescriptor(bt, bt.shape, bt.stride(), [1, 1])
        cd = TensorDescriptor(c, c.shape, c.stride(), [1, 1])
        _launch_tuned(
            _mm_w8a8_fp8_tma,
            lambda meta: (
                triton.cdiv(m, max(64, meta["BLOCK_M"]))
                * triton.cdiv(n, max(16, meta["BLOCK_N"])),
            ),
            (ad, bd, cd, sa, sb, m, n, k, a.stride(0), b.stride(0), as_, bs),
            _set_tma_blocks,
        )
        return
    # Bound FP32 scratch; split-K reduction never rounds partial sums to FP8/BF16.
    capacity = (
        1
        if k < 4096
        else max(1, min(_MAX_SPLIT_K, k // 1024, (64 * 1024 * 1024) // (m * n * 4)))
    )
    partial = (
        torch.empty((capacity, m, n), device=a.device, dtype=torch.float32)
        if capacity > 1
        else c
    )
    _, meta = _launch_tuned(
        _mm_w8a8_fp8_load,
        lambda cfg: (
            triton.cdiv(m, max(64, cfg["BLOCK_M"])),
            triton.cdiv(n, max(16, cfg["BLOCK_N"])),
            cfg["SPLIT_K"],
        ),
        (
            a,
            b,
            c,
            partial,
            sa,
            sb,
            m,
            n,
            k,
            a.stride(0),
            b.stride(0),
            a.stride(1),
            b.stride(1),
            *c.stride(),
            as_,
            bs,
            precise,
            capacity,
        ),
    )
    if meta["SPLIT_K"] > 1:
        _mm_w8a8_fp8_reduce[(triton.cdiv(m * n, 512),)](
            partial, c, sa, sb, m, n, *c.stride(), as_, bs, meta["SPLIT_K"], 512
        )


def _mm_w8a8_fp8_scale_vector(scale, x, axis, name):
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
        raise TypeError(f"{name} must be a float32 tensor")
    if scale.device != x.device:
        raise ValueError(f"{name} must be on the same device as the inputs")
    size = x.shape[axis]
    if scale.numel() == 1 and scale.ndim <= 2:
        return scale
    shape = (size, 1) if axis == 0 else (1, size)
    if scale.shape == (size,) or scale.shape == shape:
        return scale if scale.ndim == 1 else scale.reshape(size)
    raise ValueError(f"{name} must be a scalar, ({size},), or {shape}")


def mm_w8a8_fp8(
    input,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out=None,
):
    """FP8-only matmul with the torch._scaled_mm argument interface.

    Input scales are required FP32 tensors: scalar or per-row A/per-column B.
    Output defaults to input.dtype. Like CUDA torch._scaled_mm in PyTorch 2.11,
    scale_result is validated but does not affect the result.
    use_fast_accum is accepted; both settings retain FP32 accumulation.
    """
    a, b = input, mat2
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("mm_w8a8_fp8 expects two-dimensional inputs")
    if out is None:
        dtype = a.dtype if out_dtype is None else out_dtype
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=dtype)
    return mm_w8a8_fp8_out(
        a, b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out=out
    )


def mm_w8a8_fp8_out(
    input,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    """Out variant with the same required scales and optional arguments."""
    a, b = input, mat2
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("mm_w8a8_fp8 expects compatible two-dimensional inputs")
    if a.device != b.device or a.device != out.device:
        raise ValueError("inputs and out must be on the same device")
    if a.dtype not in _FP8_DTYPES or b.dtype not in _FP8_DTYPES:
        raise TypeError("mm_w8a8_fp8 requires FP8 inputs")
    if a.dtype == b.dtype == torch.float8_e5m2:
        raise TypeError("Hopper does not support E5M2 x E5M2")
    if out.dtype not in (*_FP8_DTYPES, torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("unsupported output dtype")
    if out_dtype is not None and out_dtype != out.dtype:
        raise ValueError("out_dtype must match out.dtype")
    if not isinstance(use_fast_accum, bool):
        raise TypeError("use_fast_accum must be a bool")
    sa = _mm_w8a8_fp8_scale_vector(scale_a, a, 0, "scale_a")
    sb = _mm_w8a8_fp8_scale_vector(scale_b, b, 1, "scale_b")
    if scale_result is not None:
        _mm_w8a8_fp8_scale_vector(scale_result, a, 0, "scale_result")
        if scale_result.numel() != 1:
            raise ValueError("scale_result must be a float32 scalar")
    if bias is not None:
        if out.dtype == torch.float32:
            raise TypeError(
                "CUDA torch._scaled_mm does not support bias with FP32 output"
            )
        bias_dtypes = (
            (torch.float16, torch.bfloat16)
            if out.dtype in _FP8_DTYPES
            else (out.dtype,)
        )
        if not isinstance(bias, torch.Tensor) or bias.dtype not in bias_dtypes:
            raise TypeError("bias has an unsupported dtype for the output")
        if bias.device != a.device or bias.numel() != b.shape[1]:
            raise ValueError("bias must contain N elements on the input device")
        bias = bias.reshape(-1)
    m, k = a.shape
    n = b.shape[1]
    if tuple(out.shape) != (m, n):
        out.resize_(m, n)
    if out.numel() == 0:
        return out
    if k == 0:
        return out.zero_()
    with torch_device_fn.device(a.device):
        acc = (
            torch.empty((m, n), device=a.device, dtype=torch.float32)
            if bias is not None
            else out
        )
        _scaled_matmul(a, b, acc, sa, sb)
        if bias is not None:
            _mm_w8a8_fp8_scaled_epilogue[(triton.cdiv(m * n, 512),)](
                acc, out, bias, m, n, *out.stride(), bias.stride(0), 512
            )
    return out
