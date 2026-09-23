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
from copy import deepcopy
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.triton_version_utils import HAS_TLE

_CONFIG_YAML = str(
    Path(__file__).resolve().parent.parent / "mm_w8a8_fp8" / "config.yaml"
)
_TUNE_KEY = [
    "M",
    "N",
    "K",
    "AM",
    "AK",
    "BK_STRIDE",
    "BN_STRIDE",
    "DESCRIPTOR",
    "SPLIT_K",
]
_WS_LAUNCH_CACHE = OrderedDict()


def _launch_ws_tuned(kernel, grid, args):
    """Cache compiled launch metadata, while keeping tuning and tensor data live."""
    tuner = kernel.fn
    if getattr(tuner._run_mode, "value", "normal") != "normal":
        _WS_LAUNCH_CACHE.clear()
        return kernel[grid](*args, enable_backend_opt=True)
    kernel._apply_flagtune()
    a, b = args[0], args[1]
    out, sa, sb, m, n, k = args[-6:]
    # The caller validates aligned E4M3 descriptors, scalar FP32 scales and a
    # contiguous BF16 output. Only runtime layout and pointer specialization vary.
    key = (
        kernel,
        tuner._flagtune_selection_token,
        id(tuner.configs),
        tuner.configs_hash,
        tuner._benchmark_protocol,
        a.base.device,
        m,
        n,
        k,
        tuple(a.strides),
        tuple(b.strides),
        out.data_ptr() % 16,
        sa.data_ptr() % 16,
        sb.data_ptr() % 16,
    )
    cached = _WS_LAUNCH_CACHE.get(key)
    if cached is None:
        compiled, meta = kernel[grid](*args, enable_backend_opt=True)
        tail = tuple(
            meta[name] for name in tuple(kernel.signature.parameters)[len(args) :]
        )
        launch_grid = (tuple(grid(meta)) + (1, 1))[:3]
        cached = (compiled, meta, tail, compiled[launch_grid])
        _WS_LAUNCH_CACHE[key] = cached
        if len(_WS_LAUNCH_CACHE) > 128:
            _WS_LAUNCH_CACHE.popitem(last=False)
    else:
        _WS_LAUNCH_CACHE.move_to_end(key)
        compiled, meta, tail, launch = cached
        args[0].block_shape = [meta["BM"], meta["BK"]]
        args[1].block_shape = [meta["BN"], meta["BK"]]
        if "FRAGMENTED" in meta:
            args[2].block_shape = [max(32, meta["BN"] // 4), meta["BK"]]
        launch(*args, *tail)
    return cached[:2]


def _set_descriptor_blocks(args):
    if args["DESCRIPTOR"]:
        args["A"].block_shape = [args["BLOCK_M"], args["BLOCK_K"]]
        args["B"].block_shape = [args["BLOCK_N"], args["BLOCK_K"]]


def _prune_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    candidates = []
    for config in configs:
        meta = config.kwargs
        bm, bn, bk = (meta[x] for x in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        if bm > max(16, triton.next_power_of_2(m)) or bn > max(
            64, triton.next_power_of_2(n)
        ):
            continue
        if bk > max(32, triton.next_power_of_2(k)):
            continue
        if min(m, n) >= 128 and k >= 256 and bm < 32:
            continue
        if not args["DESCRIPTOR"]:
            # Large TME tiles do not help the masked-load path. Keep its
            # established geometry and tune K blocking and pipeline depth.
            load_m = min(64, max(16, triton.next_power_of_2(m)))
            load_n = min(128, max(64, triton.next_power_of_2(n)))
            if k <= 128 or m <= 32:
                load_n = 64
            if k >= 2048 and m <= 128 and n <= 512:
                load_m, load_n = (16 if m <= 64 else 32), 64
            if (bm, bn, config.num_warps) != (load_m, load_n, 4):
                continue
            # Deep pipelines on unaligned K pitches cause pathological
            # compilation in this backend; retain its bounded two-stage path.
            if k % 16 and (
                bk != max(32, min(256, triton.next_power_of_2(k)))
                or config.num_stages > 2
            ):
                continue
        if k <= 128 and (
            config.num_stages != 1 or bk != max(32, triton.next_power_of_2(k))
        ):
            continue
        if config.num_stages > triton.cdiv(k, bk * args["SPLIT_K"]) + 1:
            continue
        if meta["PERSISTENT"] and (
            not args["DESCRIPTOR"]
            or args["SPLIT_K"] != 1
            or bm * bn < 65536
            or triton.cdiv(m, bm) * triton.cdiv(n, bn) < 2 * args["NUM_SMS"]
        ):
            continue
        # FlagTree's automatic block adjustment mutates candidate Configs.
        # Keep the YAML space intact for subsequent, differently sized inputs.
        candidates.append(deepcopy(config))
    return candidates


_DEFAULT_CONFIGS = runtime.ops_get_configs(
    "mm_w8a8_fp8_musa_default", yaml_path=_CONFIG_YAML, pre_hook=_set_descriptor_blocks
)


@libentry()
@libtuner(
    configs=_DEFAULT_CONFIGS,
    key=_TUNE_KEY,
    strategy=["default"] * len(_TUNE_KEY),
    warmup=5,
    rep=10,
    prune_configs_by={"early_config_prune": _prune_configs},
    flagtune_op_name="mm_w8a8_fp8",
    flagtune_expand_op_name="mm_w8a8_fp8_musa",
    flagtune_yaml_path=_CONFIG_YAML,
    flagtune_pre_hook=_set_descriptor_blocks,
)
@triton.jit
def mm_w8a8_fp8_kernel(
    A,
    B,
    C,
    SA,
    SB,
    Bias,
    SR,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK_STRIDE: tl.constexpr,
    BN_STRIDE: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    DESCRIPTOR: tl.constexpr,
    SA_STRIDE: tl.constexpr,
    SB_STRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    HAS_SR: tl.constexpr,
    SR_STRIDE: tl.constexpr,
    GROUP_M: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    first = tl.program_id(0)
    count = tl.cdiv(grid_m * grid_n - first, NUM_SMS) if PERSISTENT else 1
    for tile in range(count):
        pid = first + tile * NUM_SMS
        group = pid // (GROUP_M * grid_n)
        group_m = tl.minimum(GROUP_M, grid_m - group * GROUP_M)
        pm = group * GROUP_M + pid % group_m
        pn = pid % (GROUP_M * grid_n) // group_m
        rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        split = tl.program_id(1) if SPLIT_K > 1 else 0
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for block in tl.range(split, tl.cdiv(K, BLOCK_K), SPLIT_K):
            if DESCRIPTOR:
                a = tl.load_tensor_descriptor(A, [pm * BLOCK_M, block * BLOCK_K])
                bt = tl.load_tensor_descriptor(B, [pn * BLOCK_N, block * BLOCK_K])
            else:
                k = block * BLOCK_K + rk
                a = tl.load(
                    A + rm[:, None].to(tl.int64) * AM + k[None, :].to(tl.int64) * AK,
                    (rm[:, None] < M) & (k[None, :] < K),
                    0.0,
                )
                bt = tl.load(
                    B
                    + rn[:, None].to(tl.int64) * BN_STRIDE
                    + k[None, :].to(tl.int64) * BK_STRIDE,
                    (rn[:, None] < N) & (k[None, :] < K),
                    0.0,
                )
            acc = tl.dot(a, tl.trans(bt), acc)
        if SA_STRIDE == 0:
            acc *= tl.load(SA)
        else:
            acc *= tl.load(SA + rm * SA_STRIDE, rm < M, 0.0)[:, None]
        if SB_STRIDE == 0:
            acc *= tl.load(SB)
        else:
            acc *= tl.load(SB + rn * SB_STRIDE, rn < N, 0.0)[None, :]
        if SPLIT_K > 1:
            ptr = C + split * M * N + rm[:, None] * N + rn[None, :]
            tl.store(ptr, acc, (rm[:, None] < M) & (rn[None, :] < N))
        else:
            if HAS_BIAS:
                acc += tl.load(Bias + rn * BIAS_STRIDE, rn < N, 0.0)[None, :].to(
                    tl.float32
                )
            if HAS_SR:
                if SR_STRIDE == 0:
                    acc /= tl.load(SR)
                else:
                    acc /= tl.load(SR + rm * SR_STRIDE, rm < M, 1.0)[:, None]
            if CM >= 0 and CN >= 0 and (M - 1) * CM + (N - 1) * CN < 2147483648:
                ptr = C + rm[:, None] * CM + rn[None, :] * CN
            else:
                ptr = C + rm[:, None].to(tl.int64) * CM + rn[None, :].to(tl.int64) * CN
            tl.store(ptr, acc, (rm[:, None] < M) & (rn[None, :] < N))


@libentry()
@triton.jit
def _reduce_split_k(
    P,
    C,
    Bias,
    SR,
    M: tl.constexpr,
    N: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    HAS_SR: tl.constexpr,
    SR_STRIDE: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(P + s * M * N + x, x < M * N, 0.0)
    if HAS_BIAS:
        acc += tl.load(Bias + (x % N) * BIAS_STRIDE, x < M * N, 0.0).to(tl.float32)
    if HAS_SR:
        if SR_STRIDE == 0:
            acc /= tl.load(SR)
        else:
            acc /= tl.load(SR + (x // N) * SR_STRIDE, x < M * N, 1.0)
    tl.store(C + (x // N).to(tl.int64) * CM + (x % N).to(tl.int64) * CN, acc, x < M * N)


def _select_split_k(m, n, k, descriptor):
    if k >= 2048 and m <= 128 and n <= 512:
        # Increase CTA count only for skinny grids. Partial sums stay in FP32.
        bm = 16 if m <= 64 else 32
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, 64)
        return min(
            32,
            triton.next_power_of_2(triton.cdiv(120, tiles)),
            triton.next_power_of_2(triton.cdiv(k, 256)),
        )
    if descriptor and k >= 8192 and m <= 512 and m * n <= 512 * 1024:
        # Bound workspace/reduction traffic while filling underoccupied TME
        # grids. Wider M tiles amortize loads when a full tile is available.
        bm = min(64 if n <= 512 else 128, max(32, triton.next_power_of_2(m)))
        bn = 64 if m <= 64 else 128
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        target_ctas = 128 if m <= 64 else 64
        return min(16, triton.next_power_of_2(triton.cdiv(target_ctas, tiles)))
    return 1


def _launch(a, b, out, sa, sb, sa_stride, sb_stride, bias, sr, sr_stride):
    m, k = a.shape
    n = b.shape[1]
    # E5M2 descriptor loads fail S5000 accuracy checks. Masked FP8 loads also
    # cover unaligned and non-contiguous inputs without BF16 dequantization.
    descriptor = (
        a.dtype == b.dtype == torch.float8_e4m3fn
        and a.stride(1) == 1
        and b.stride(0) == 1
        and a.stride(0) > 0
        and b.stride(1) > 0
        and a.stride(0) % 16 == 0
        and b.stride(1) % 16 == 0
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
        and k >= 16
    )
    split = _select_split_k(m, n, k, descriptor)
    if (
        HAS_TLE
        and descriptor
        and split == 1
        and min(m, n, k) >= 64
        and (k < 2048 or (m >= 128 and n >= 256))
        and sa_stride == sb_stride == 0
        and bias is None
        and sr is None
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
    ):
        from ._mm_w8a8_fp8_ws import fragmented_kernel, ws_multi_kernel

        aa = TensorDescriptor(a, [m, k], list(a.stride()), [64, 64])
        bb = TensorDescriptor(b, [n, k], [b.stride(1), b.stride(0)], [64, 64])

        # Medium dense tiles can avoid a third CTA wave with an N=128+32 split.
        # FlagTune compares this geometry with the original equal-width layout.
        if 1536 <= min(m, n) and max(m, n) <= 3072 and 1024 <= k <= 2048:
            bsmall = TensorDescriptor(b, [n, k], [b.stride(1), b.stride(0)], [32, 64])

            def fragmented_grid(meta):
                tn = (
                    meta["BN"] + meta["BN"] // 4
                    if meta["FRAGMENTED"]
                    else meta["BN"] * meta["NC"]
                )
                return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, tn),)

            _launch_ws_tuned(
                fragmented_kernel,
                fragmented_grid,
                (aa, bb, bsmall, out, sa, sb, m, n, k),
            )
            return out

        def ws_grid(meta):
            return (
                triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"] * meta["NC"]),
            )

        _launch_ws_tuned(ws_multi_kernel, ws_grid, (aa, bb, out, sa, sb, m, n, k))
        return out
    if descriptor:
        # The config pre-hook installs the actual tile before every launch.
        aa = TensorDescriptor(a, [m, k], list(a.stride()), [16, 32])
        bb = TensorDescriptor(b, [n, k], [b.stride(1), b.stride(0)], [64, 32])
    else:
        aa, bb = a, b
    partial = (
        torch.empty((split, m, n), device=a.device, dtype=torch.float32)
        if split > 1
        else out
    )
    epilogue = dict(
        HAS_BIAS=bias is not None,
        BIAS_STRIDE=bias.stride(0) if bias is not None else 0,
        HAS_SR=sr is not None,
        SR_STRIDE=sr_stride,
    )
    # SQMMA instruction scheduling improves dense TME tiles on S5000.
    compiler_options = {"enable_backend_opt": True} if descriptor else {}
    num_sms = torch_device_fn.get_device_properties(a.device).multi_processor_count

    def grid(meta):
        tiles = triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"])
        return (min(num_sms, tiles) if meta["PERSISTENT"] else tiles, split)

    kernel = mm_w8a8_fp8_kernel
    if k == 0:
        # There is only an epilogue to execute. Bypass the autotuner, whose
        # automatic block adjustment would otherwise shrink BLOCK_K to zero.
        kernel = kernel.fn.fn
        compiler_options.update(_DEFAULT_CONFIGS[0].all_kwargs())

    kernel[grid](
        aa,
        bb,
        partial,
        sa,
        sb,
        bias,
        sr,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        num_sms,
        SPLIT_K=split,
        DESCRIPTOR=descriptor,
        SA_STRIDE=sa_stride,
        SB_STRIDE=sb_stride,
        **epilogue,
        **compiler_options,
    )
    if split > 1:
        _reduce_split_k[(triton.cdiv(m * n, 512),)](
            partial,
            out,
            bias,
            sr,
            m,
            n,
            out.stride(0),
            out.stride(1),
            SPLIT_K=split,
            BLOCK=512,
            **epilogue,
            num_warps=4,
        )
    return out


def _scale_stride(scale, x, axis, name):
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
        raise TypeError(f"{name} must be a float32 tensor")
    if scale.device != x.device:
        raise ValueError(f"{name} must be on the same device as the inputs")
    if scale.numel() == 1 and scale.ndim <= 2:
        return 0
    size = x.shape[axis]
    shape = (size, 1) if axis == 0 else (1, size)
    if scale.shape == (size,):
        return scale.stride(0)
    if scale.shape == shape:
        return scale.stride(axis)
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
    """FP8 matmul with the torch._scaled_mm interface on MThreads.

    scale_a is scalar, (M,), or (M, 1); scale_b is scalar, (N,), or (1, N).
    Scales must be float32 tensors on the input device. Following torch_musa,
    scale_result divides the scaled product plus bias, for all output dtypes;
    it can be scalar, (M,), or (M, 1). Output defaults to input.dtype.
    Both use_fast_accum settings use FP32 accumulation, as in torch_musa.
    BF16/FP16 inputs and K-block scales are not supported.
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
    """torch._scaled_mm.out-compatible variant with a reusable output."""
    a, b = input, mat2
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("mm_w8a8_fp8 expects compatible two-dimensional inputs")
    if a.device != b.device or a.device != out.device:
        raise ValueError("inputs and out must be on the same device")
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if a.dtype not in fp8_dtypes or b.dtype not in fp8_dtypes:
        raise TypeError("mm_w8a8_fp8 requires FP8 inputs")
    if out.dtype not in (*fp8_dtypes, torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("unsupported output dtype")
    if out_dtype is not None and out_dtype != out.dtype:
        raise ValueError("out_dtype must match out.dtype")
    if not isinstance(use_fast_accum, bool):
        raise TypeError("use_fast_accum must be a bool")
    sa_stride = _scale_stride(scale_a, a, 0, "scale_a")
    sb_stride = _scale_stride(scale_b, b, 1, "scale_b")
    sr_stride = (
        _scale_stride(scale_result, a, 0, "scale_result")
        if scale_result is not None
        else 0
    )
    if bias is not None:
        bias_dtypes = (
            (torch.float32,)
            if out.dtype == torch.float32
            else (torch.float16, torch.bfloat16)
        )
        if not isinstance(bias, torch.Tensor) or bias.dtype not in bias_dtypes:
            raise TypeError("bias has an unsupported dtype for the output")
        if bias.device != a.device or bias.numel() != b.shape[1]:
            raise ValueError("bias must contain N elements on the input device")
        bias = bias.reshape(-1)
    if tuple(out.shape) != (a.shape[0], b.shape[1]):
        out.resize_(a.shape[0], b.shape[1])
    if out.numel() == 0:
        return out
    with torch_device_fn.device(a.device):
        return _launch(
            a,
            b,
            out,
            scale_a,
            scale_b,
            sa_stride,
            sb_stride,
            bias,
            scale_result,
            sr_stride,
        )
