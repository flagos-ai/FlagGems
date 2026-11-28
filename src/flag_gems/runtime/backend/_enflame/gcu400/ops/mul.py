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

import logging
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

# Name of the active flag_gems backend device (e.g. "cuda", "musa"). The
# optimized Triton path below runs on this device; tensors on other device
# types (e.g. plain "cpu") fall through to the aten reference path. Gating on
# the hardcoded literal "cuda" wrongly excluded backends whose device type is
# not "cuda" (e.g. mthreads reports "musa") from the shared Triton path.
_DEVICE_NAME = runtime_device.name

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


# Fixed Grid (48 CTAs) + grid-stride: larger BLOCK_SIZE amortizes the
# per-tile overhead, and the libtuner picks per (n_elements, dtype).
_GRID_SIZE = 48

# 2D broadcast tiling: column-direction block cap, target total CTA count, and minimum column width to enable this path.
# When columns are too narrow, in-row vectors cannot fill a full vector load, the row-loop iteration count explodes, and
# it becomes much slower than the generic kernel,
# so narrow columns still use the original path.
_TILE_MAX_BLOCK = 2048
_TILE_TARGET_CTAS = 64
_TILE_MIN_COLS = 128


def mul_get_configs():
    return [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=4, num_stages=4),
    ]


def mul_broadcast_get_configs():
    return [
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=4, num_stages=4),
    ]


@libentry()
@libtuner(
    configs=mul_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(
        tl.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x & y if IS_BOOL else x * y
        tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_scalar_kernel(
    x_ptr,
    output_ptr,
    scalar,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(
        tl.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "n_cols", "dtype"],
    strategy=["align32", "default", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul_broadcast_2d",
)
@triton.jit
def mul_broadcast_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    n_cols,
    a_s0: tl.constexpr,
    a_s1: tl.constexpr,
    b_s0: tl.constexpr,
    b_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(
        tl.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        row = offsets // n_cols
        col = offsets - row * n_cols
        a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask=mask)
        b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask=mask)
        out = a & b if IS_BOOL else a * b
        tl.store(out_ptr + row * out_s0 + col * out_s1, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_generic_nd_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    A_STRIDE: tl.constexpr,
    B_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(
        tl.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        linear = offsets
        a_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        b_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        for dim in tl.static_range(NDIM - 1, -1, -1):
            idx = linear % SHAPE[dim]
            linear = linear // SHAPE[dim]
            a_offsets += idx * A_STRIDE[dim]
            b_offsets += idx * B_STRIDE[dim]
            out_offsets += idx * OUT_STRIDE[dim]

        a = tl.load(a_ptr + a_offsets, mask=mask)
        b = tl.load(b_ptr + b_offsets, mask=mask)
        out = a & b if IS_BOOL else a * b
        tl.store(out_ptr + out_offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_generic_nd_runtime_meta_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    meta_ptr,
    n_elements,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(
        tl.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        linear = offsets
        a_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        b_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        for dim in tl.static_range(NDIM - 1, -1, -1):
            shape_dim = tl.load(meta_ptr + dim)
            a_stride_dim = tl.load(meta_ptr + NDIM + dim)
            b_stride_dim = tl.load(meta_ptr + 2 * NDIM + dim)
            out_stride_dim = tl.load(meta_ptr + 3 * NDIM + dim)
            idx = linear % shape_dim
            linear = linear // shape_dim
            a_offsets += idx * a_stride_dim
            b_offsets += idx * b_stride_dim
            out_offsets += idx * out_stride_dim

        a = tl.load(a_ptr + a_offsets, mask=mask)
        b = tl.load(b_ptr + b_offsets, mask=mask)
        out = a & b if IS_BOOL else a * b
        tl.store(out_ptr + out_offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_complex_generic_nd_kernel(
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    out_r_ptr,
    out_i_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    AR_STRIDE: tl.constexpr,
    AI_STRIDE: tl.constexpr,
    BR_STRIDE: tl.constexpr,
    BI_STRIDE: tl.constexpr,
    OUT_R_STRIDE: tl.constexpr,
    OUT_I_STRIDE: tl.constexpr,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(
        tl.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        linear = offsets
        ar_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        ai_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        br_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        bi_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_r_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_i_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        for dim in tl.static_range(NDIM - 1, -1, -1):
            idx = linear % SHAPE[dim]
            linear = linear // SHAPE[dim]
            ar_offsets += idx * AR_STRIDE[dim]
            ai_offsets += idx * AI_STRIDE[dim]
            br_offsets += idx * BR_STRIDE[dim]
            bi_offsets += idx * BI_STRIDE[dim]
            out_r_offsets += idx * OUT_R_STRIDE[dim]
            out_i_offsets += idx * OUT_I_STRIDE[dim]

        ar = tl.load(ar_ptr + ar_offsets, mask=mask, other=0.0)
        ai = tl.load(ai_ptr + ai_offsets, mask=mask, other=0.0)
        br = tl.load(br_ptr + br_offsets, mask=mask, other=0.0)
        bi = tl.load(bi_ptr + bi_offsets, mask=mask, other=0.0)

        # complex32's real/imag parts are fp16; all-fp16 multiply-add pushes a few elements beyond tolerance;
        # promote to fp32 for intermediates then cast back to the original dtype, matching the reference rounding.
        ar_f32 = ar.to(tl.float32)
        ai_f32 = ai.to(tl.float32)
        br_f32 = br.to(tl.float32)
        bi_f32 = bi.to(tl.float32)
        out_r = (ar_f32 * br_f32 - ai_f32 * bi_f32).to(ar.dtype)
        out_i = (ar_f32 * bi_f32 + ai_f32 * br_f32).to(ar.dtype)
        tl.store(out_r_ptr + out_r_offsets, out_r, mask=mask)
        tl.store(out_i_ptr + out_i_offsets, out_i, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_complex_generic_nd_runtime_meta_kernel(
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    out_r_ptr,
    out_i_ptr,
    meta_ptr,
    n_elements,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(
        tl.program_id(0), num_tile, GRID_SIZE, num_stages=NUM_STAGES
    ):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        linear = offsets
        ar_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        ai_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        br_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        bi_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_r_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_i_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

        for dim in tl.static_range(NDIM - 1, -1, -1):
            shape_dim = tl.load(meta_ptr + dim)
            ar_stride_dim = tl.load(meta_ptr + NDIM + dim)
            ai_stride_dim = tl.load(meta_ptr + 2 * NDIM + dim)
            br_stride_dim = tl.load(meta_ptr + 3 * NDIM + dim)
            bi_stride_dim = tl.load(meta_ptr + 4 * NDIM + dim)
            out_r_stride_dim = tl.load(meta_ptr + 5 * NDIM + dim)
            out_i_stride_dim = tl.load(meta_ptr + 6 * NDIM + dim)
            idx = linear % shape_dim
            linear = linear // shape_dim
            ar_offsets += idx * ar_stride_dim
            ai_offsets += idx * ai_stride_dim
            br_offsets += idx * br_stride_dim
            bi_offsets += idx * bi_stride_dim
            out_r_offsets += idx * out_r_stride_dim
            out_i_offsets += idx * out_i_stride_dim

        ar = tl.load(ar_ptr + ar_offsets, mask=mask, other=0.0)
        ai = tl.load(ai_ptr + ai_offsets, mask=mask, other=0.0)
        br = tl.load(br_ptr + br_offsets, mask=mask, other=0.0)
        bi = tl.load(bi_ptr + bi_offsets, mask=mask, other=0.0)

        # complex32's real/imag parts are fp16; all-fp16 multiply-add pushes a few elements beyond tolerance;
        # promote to fp32 for intermediates then cast back to the original dtype, matching the reference rounding.
        ar_f32 = ar.to(tl.float32)
        ai_f32 = ai.to(tl.float32)
        br_f32 = br.to(tl.float32)
        bi_f32 = bi.to(tl.float32)
        out_r = (ar_f32 * br_f32 - ai_f32 * bi_f32).to(ar.dtype)
        out_i = (ar_f32 * bi_f32 + ai_f32 * br_f32).to(ar.dtype)
        tl.store(out_r_ptr + out_r_offsets, out_r, mask=mask)
        tl.store(out_i_ptr + out_i_offsets, out_i, mask=mask)


def _is_tensor_or_number(value) -> bool:
    return isinstance(value, torch.Tensor) or isinstance(value, Number)


def _select_device(a, b):
    for value in (a, b):
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _as_tensor(value, *, device, dtype):
    if isinstance(value, torch.Tensor):
        if value.device != device or value.dtype != dtype:
            return value.to(device=device, dtype=dtype)
        return value
    return torch.tensor(value, device=device, dtype=dtype)


def _result_dtype(a, b):
    # Same-dtype tensor pairs dominate real workloads; torch.result_type is a
    # comparatively expensive dispatch, so short-circuit the common case.
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.dtype is b.dtype:
            return a.dtype
    return torch.result_type(a, b)


def _is_bool_dtype(dtype):
    return dtype is torch.bool


def _triton_version_lt(major, minor):
    version = triton.__version__.split("+", 1)[0]
    parts = version.split(".")
    try:
        current = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return False
    return current < (major, minor)


def _needs_runtime_meta_for_constexpr_tuple():
    if _triton_version_lt(3, 3):
        return True
    try:
        import triton.language.core as tl_core

        frontend_tuple = getattr(tl_core, "tuple", None)
        return frontend_tuple is None or not hasattr(frontend_tuple, "__getitem__")
    except Exception:
        return False


_DTYPE_NAME_CACHE = {}


def _dtype_name(dtype):
    name = _DTYPE_NAME_CACHE.get(dtype)
    if name is None:
        name = str(dtype).split(".")[-1]
        _DTYPE_NAME_CACHE[dtype] = name
    return name


def _broadcast_shape(a_t, b_t):
    a_shape = a_t.shape
    if a_shape == b_t.shape:
        return tuple(a_shape)
    return tuple(torch.broadcast_shapes(tuple(a_shape), tuple(b_t.shape)))


def _broadcasted_stride(shape, stride, out_shape):
    if not out_shape:
        return ()
    offset = len(out_shape) - len(shape)
    result = []
    for out_dim, out_size in enumerate(out_shape):
        in_dim = out_dim - offset
        if in_dim < 0:
            result.append(0)
            continue
        size = shape[in_dim]
        result.append(0 if size == 1 and out_size != 1 else stride[in_dim])
    return tuple(result)


def _real_output(a_t, b_t, out=None):
    out_shape = _broadcast_shape(a_t, b_t)
    if out is None:
        return torch.empty(out_shape, device=a_t.device, dtype=a_t.dtype)
    if tuple(out.shape) != out_shape:
        raise RuntimeError(
            f"output with shape {tuple(out.shape)} cannot be broadcast to {out_shape}"
        )
    if out.dtype != a_t.dtype:
        raise RuntimeError(
            f"output dtype {out.dtype} does not match result dtype {a_t.dtype}"
        )
    if out.device != a_t.device:
        raise RuntimeError("output must be on the same device as inputs")
    return out


def _can_use_contiguous_tensor_tensor(a_t, b_t, out):
    return (
        tuple(a_t.shape) == tuple(b_t.shape) == tuple(out.shape)
        and a_t.is_contiguous()
        and b_t.is_contiguous()
        and out.is_contiguous()
        and not a_t.is_complex()
        and not b_t.is_complex()
    )


def _can_use_contiguous_scalar(tensor, out):
    return tensor.is_contiguous() and out.is_contiguous() and not tensor.is_complex()


def _real_layout(out_shape, a_t, b_t, out):
    a_stride = _broadcasted_stride(tuple(a_t.shape), tuple(a_t.stride()), out_shape)
    b_stride = _broadcasted_stride(tuple(b_t.shape), tuple(b_t.stride()), out_shape)
    out_stride = tuple(out.stride()) if out_shape else ()
    return a_stride, b_stride, out_stride


def _can_use_2d_broadcast(out_shape):
    return len(out_shape) == 2


@triton.jit
def mul_contig_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NS: tl.constexpr,
    EVEN: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(tl.program_id(0), num_tile, GRID_SIZE, num_stages=NS):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            x = tl.load(x_ptr + offsets)
            y = tl.load(y_ptr + offsets)
            out = x & y if IS_BOOL else x * y
            tl.store(output_ptr + offsets, out)
        else:
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            out = x & y if IS_BOOL else x * y
            tl.store(output_ptr + offsets, out, mask=mask)


@triton.jit
def mul_scalar_contig_kernel(
    x_ptr,
    output_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NS: tl.constexpr,
    EVEN: tl.constexpr,
):
    num_tile = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(tl.program_id(0), num_tile, GRID_SIZE, num_stages=NS):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            x = tl.load(x_ptr + offsets)
            out = x & scalar if IS_BOOL else x * scalar
            tl.store(output_ptr + offsets, out)
        else:
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            out = x & scalar if IS_BOOL else x * scalar
            tl.store(output_ptr + offsets, out, mask=mask)


def _pick_block(n_elements, element_size, scalar_path=False):
    """Deterministic block size for the contiguous-path kernels.

    A tile of ``BLOCK_SIZE`` elements must pull one tile from each input and
    push one tile to the output, so the 16-bit types (half the bytes per tile)
    need more elements per tile than 32-bit ones to keep the memory pipeline
    busy. The tensor-by-scalar kernel only has two memory streams instead of
    three, so it can afford a 2x larger tile at the same byte budget. Tiny
    inputs stay on the smallest block to avoid masked tail waste.
    """
    if n_elements <= 4096:
        return 1024
    if element_size <= 2:
        return 16384 if scalar_path else 8192
    return 32768


_HUGE_STREAM_ELEMENTS = 1 << 25

# Wide-stream bands re-probed 2026-11-17 (GCU400, same-process round-robin
# do_bench w1000/r100 vs native anchor): the 2-byte tensor-tensor stream and
# both scalar wide bands win with bigger tiles + shallower pipelining than the
# mid-stream tiers below.
_TT_WIDE_2B_N = 1 << 28  # gs48 B131072/nw2/st2: 2^30 f16 2.214 vs 2.271 ms
_SCALAR_WIDE_N = 1 << 22  # f16 B131072/nw4/st2, f32 B65536/nw4/st2


def _pick_tensor_tensor_launch(n_elements, element_size):
    """(BLOCK_SIZE, num_warps, num_stages) for the tensor-by-tensor kernel.

    Probe sweep on GCU400 (do_bench warmup=1000ms rep=100ms median, 3-round
    round-robin vs native anchor): the wide 2-byte stream (>=2^28 elements)
    runs 2.5% faster with a 131072-wide tile at nw2/st2 (2.214 vs 2.271 ms at
    2^30 f16) than the 65536/nw2/st4 mid-stream config; the 131072 tile is
    768KB per program so those launches run 24-way, which the sweep shows is
    still the fastest shape at that size. All other tiers keep their probe
    history: nw4 collapses on huge streams (1G fp16 BLOCK 32768: nw4 3161us
    vs nw1 2181us, +45%) while nw1 gives back ~1.5% on mid streams, so mid
    streams (<32M elements) run nw2 and huge streams run nw1. Mid-stream
    2-byte types now use BLOCK 16384: a 2026-11 re-probe at 2^24 elements
    beats BLOCK 65536 in all three dtypes (fp16 1.006x vs 0.996x, bf16
    1.005x vs 0.996x, num_stages 2/4 indistinguishable), because the 16K
    tile spreads the 16M stream over 1024 tiles instead of 256. 4-byte
    types keep BLOCK 32768. 8-byte types keep the 384KB on-chip tile budget
    of the validated 4-byte config. Tiny inputs stay on BLOCK 1024 at the
    launch floor (~3.3us).
    """
    if element_size == 2 and n_elements >= _TT_WIDE_2B_N:
        return 131072, 2, 2
    if n_elements <= 4096:
        return 1024, 4, 4
    if element_size >= 8:
        block = 16384
    elif element_size <= 2:
        block = 16384
    else:
        block = 32768
    if n_elements >= _HUGE_STREAM_ELEMENTS:
        return block, 1, 4
    return block, 2, 4


def _pick_scalar_launch(n_elements, element_size):
    """(BLOCK_SIZE, num_warps, num_stages) for the tensor-by-scalar kernel.

    Re-probed 2026-11-17 (same-process round-robin vs native anchor): the
    two-stream profile saturates with wider tiles than the old
    ``_pick_block`` ladder -- 2-byte streams >=2^22 elements run
    B131072/nw4/st2 (2^30 f16 1.374 vs 1.659 ms, -17%; 2^24 f16 27.3 vs
    29.8us, -8%) and 4-byte streams run B65536/nw4/st2 (2^30 f32 2.775 vs
    2.949 ms, -6%). 1/8-byte and smaller streams keep the historical
    ``_pick_block`` picks (a 16384-wide tile collapses on huge streams: the
    old config lost 13% to native at 2^30 f16).
    """
    if n_elements >= _SCALAR_WIDE_N:
        if element_size == 2:
            return 131072, 4, 2
        if element_size == 4:
            return 65536, 4, 2
    return _pick_block(n_elements, element_size, scalar_path=True), 4, 4


def _grid_for(n_elements):
    """Fixed reference block for the grid size so the kernel GRID_SIZE
    argument always matches grid dim 0, independent of which BLOCK_SIZE the
    libtuner picks.
    """
    tiles = (n_elements + 1023) // 1024
    return _GRID_SIZE if tiles > _GRID_SIZE else tiles


def _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = (_grid_for(n_elements),)
    with torch_device_fn.device(output.device):
        mul_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
            GRID_SIZE=grid[0],
            NUM_STAGES=4,
        )
    return output


def _fast_contiguous_mul(a, b, out):
    """Same-shape, same-dtype, contiguous fast path.

    Skips result-type inference, broadcast inference and tensor coercion, which
    together cost several times the kernel launch itself on small inputs.
    Returns ``None`` when the expensive generic path must handle the call.
    """
    if a.device.type != _DEVICE_NAME or b.device != a.device:
        return None
    dtype = a.dtype
    if b.dtype is not dtype or a.shape != b.shape or dtype.is_complex:
        return None
    if not (a.is_contiguous() and b.is_contiguous()):
        return None
    if out is None:
        output = torch.empty(a.shape, device=a.device, dtype=dtype)
    else:
        if (
            out.shape != a.shape
            or out.dtype is not dtype
            or out.device != a.device
            or not out.is_contiguous()
        ):
            return None
        output = out
    n_elements = output.numel()
    if n_elements == 0:
        return output
    block_size, num_warps, num_stages = _pick_tensor_tensor_launch(
        n_elements, a.element_size()
    )
    tiles = (n_elements + block_size - 1) // block_size
    grid = (_GRID_SIZE if tiles > _GRID_SIZE else tiles,)
    launch_kwargs = {"num_warps": num_warps}
    if num_stages == 4:
        launch_kwargs["num_stages"] = 4
    with torch_device_fn.device(output.device):
        mul_contig_kernel[grid](
            a,
            b,
            output,
            n_elements,
            BLOCK_SIZE=block_size,
            IS_BOOL=dtype is torch.bool,
            GRID_SIZE=grid[0],
            NS=num_stages,
            EVEN=n_elements % block_size == 0,
            **launch_kwargs,
        )
    return output


def _launch_scalar(tensor, scalar, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = (_grid_for(n_elements),)
    with torch_device_fn.device(output.device):
        mul_scalar_kernel[grid](
            tensor,
            output,
            scalar,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
            GRID_SIZE=grid[0],
            NUM_STAGES=4,
        )
    return output


def _fast_scalar_mul(tensor, scalar, out):
    """Contiguous tensor-by-number fast path with no result-type promotion.

    Returns ``None`` when the generic path must take over.
    """
    if not isinstance(scalar, Number) or isinstance(scalar, complex):
        return None
    if tensor.device.type != _DEVICE_NAME or not tensor.is_contiguous():
        return None
    dtype = _result_dtype(tensor, scalar)
    if dtype is not tensor.dtype:
        return None
    if out is None:
        output = torch.empty(tensor.shape, device=tensor.device, dtype=dtype)
    else:
        if (
            out.shape != tensor.shape
            or out.dtype is not dtype
            or out.device != tensor.device
            or not out.is_contiguous()
        ):
            return None
        output = out
    n_elements = output.numel()
    if n_elements == 0:
        return output
    block_size, num_warps, num_stages = _pick_scalar_launch(
        n_elements, tensor.element_size()
    )
    grid = (_grid_for(n_elements),)
    launch_kwargs = {"num_warps": num_warps}
    if num_stages == 4:
        launch_kwargs["num_stages"] = 4
    with torch_device_fn.device(output.device):
        mul_scalar_contig_kernel[grid](
            tensor,
            output,
            scalar,
            n_elements,
            BLOCK_SIZE=block_size,
            IS_BOOL=dtype is torch.bool,
            GRID_SIZE=grid[0],
            NS=num_stages,
            EVEN=n_elements % block_size == 0,
            **launch_kwargs,
        )
    return output


@libentry()
@triton.jit
def mul_broadcast_2d_tile_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    a_s0: tl.constexpr,
    a_s1: tl.constexpr,
    b_s0: tl.constexpr,
    b_s1: tl.constexpr,
    A_ROW_CONST: tl.constexpr,
    A_ROW_SCALAR: tl.constexpr,
    B_ROW_CONST: tl.constexpr,
    B_ROW_SCALAR: tl.constexpr,
    dtype: tl.constexpr,
    IS_BOOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_ROWS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Broadcast multiply using a 2-D "row scalar x in-row vector" tiling.

    Compared with the generic broadcast kernel:
    - the row comes from the row loop on program_id(1), removing per-element // and % n_cols;
    - the broadcast axis (stride 0) is either loaded once outside the loop or degrades to a
      scalar load plus broadcast, so it is no longer gathered element-wise;
    - the output is written row-contiguously with coalesced column access.
    """
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    cols = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = cols < n_cols
    if A_ROW_CONST:
        a_row_vec = tl.load(a_ptr + cols * a_s1, mask=col_mask)
    if B_ROW_CONST:
        b_row_vec = tl.load(b_ptr + cols * b_s1, mask=col_mask)
    for row in tl.range(pid_r, n_rows, GRID_ROWS, num_stages=NUM_STAGES):
        if A_ROW_CONST:
            a = a_row_vec
        elif A_ROW_SCALAR:
            a = tl.load(a_ptr + row * a_s0)
        else:
            a = tl.load(a_ptr + row * a_s0 + cols * a_s1, mask=col_mask)
        if B_ROW_CONST:
            b = b_row_vec
        elif B_ROW_SCALAR:
            b = tl.load(b_ptr + row * b_s0)
        else:
            b = tl.load(b_ptr + row * b_s0 + cols * b_s1, mask=col_mask)
        out = a & b if IS_BOOL else a * b
        tl.store(out_ptr + row * n_cols + cols, out, mask=col_mask)


def _can_use_2d_tile(out_shape, out_stride):
    """2-D tiling requires: exactly 2-D, contiguous output rows, and a not-too-narrow column width."""
    if len(out_shape) != 2:
        return False
    n_rows, n_cols = out_shape
    if n_rows <= 0 or n_cols < _TILE_MIN_COLS:
        return False
    return tuple(out_stride) == (n_cols, 1)


def _launch_2d_tile(a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype):
    n_rows, n_cols = out_shape
    block_size = min(_TILE_MAX_BLOCK, max(32, triton.next_power_of_2(n_cols)))
    grid_cols = triton.cdiv(n_cols, block_size)
    # Split along rows only until the "total program count is enough to fill the device"; remaining rows are consumed by
    # the row loop
    grid_rows = max(1, min(n_rows, -(-_TILE_TARGET_CTAS // grid_cols)))
    with torch_device_fn.device(output.device):
        mul_broadcast_2d_tile_kernel[(grid_cols, grid_rows)](
            a_t,
            b_t,
            output,
            n_rows,
            n_cols,
            a_s0=a_stride[0],
            a_s1=a_stride[1],
            b_s0=b_stride[0],
            b_s1=b_stride[1],
            A_ROW_CONST=a_stride[0] == 0,
            A_ROW_SCALAR=a_stride[1] == 0,
            B_ROW_CONST=b_stride[0] == 0,
            B_ROW_SCALAR=b_stride[1] == 0,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
            BLOCK_SIZE=block_size,
            GRID_ROWS=grid_rows,
            NUM_STAGES=4,
        )
    return output


def _launch_2d_broadcast(
    a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    # Fixed reference block for the grid size so the kernel GRID_SIZE
    # argument (same expression below) always matches grid dim 0,
    # independent of which BLOCK_SIZE the libtuner picks.
    grid = lambda meta: (min(_GRID_SIZE, triton.cdiv(n_elements, 1024)), 1, 1)
    with torch_device_fn.device(output.device):
        mul_broadcast_2d_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            out_shape[1],
            a_s0=a_stride[0],
            a_s1=a_stride[1],
            b_s0=b_stride[0],
            b_s1=b_stride[1],
            out_s0=out_stride[0],
            out_s1=out_stride[1],
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
            GRID_SIZE=min(_GRID_SIZE, triton.cdiv(n_elements, 1024)),
            NUM_STAGES=4,
        )
    return output


def _launch_generic(a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    shape = out_shape or (1,)
    ndim = len(shape)
    a_stride = a_stride or (0,)
    b_stride = b_stride or (0,)
    out_stride = out_stride or (0,)
    # Fixed reference block for the grid size so the kernel GRID_SIZE
    # argument (same expression below) always matches grid dim 0,
    # independent of which BLOCK_SIZE the libtuner picks.
    grid = lambda meta: (min(_GRID_SIZE, triton.cdiv(n_elements, 1024)), 1, 1)
    if _needs_runtime_meta_for_constexpr_tuple():
        meta = torch.tensor(
            shape + a_stride + b_stride + out_stride,
            device=output.device,
            dtype=torch.int64,
        )
        with torch_device_fn.device(output.device):
            mul_generic_nd_runtime_meta_kernel[grid](
                a_t,
                b_t,
                output,
                meta,
                n_elements,
                NDIM=ndim,
                dtype=_dtype_name(dtype),
                IS_BOOL=_is_bool_dtype(dtype),
                GRID_SIZE=min(_GRID_SIZE, triton.cdiv(n_elements, 1024)),
                NUM_STAGES=4,
            )
        return output

    with torch_device_fn.device(output.device):
        mul_generic_nd_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            SHAPE=shape,
            A_STRIDE=a_stride,
            B_STRIDE=b_stride,
            OUT_STRIDE=out_stride,
            NDIM=ndim,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
            GRID_SIZE=min(_GRID_SIZE, triton.cdiv(n_elements, 1024)),
            NUM_STAGES=4,
        )
    return output


def mul_broadcast_func(a, b, out=None):
    if not (_is_tensor_or_number(a) and _is_tensor_or_number(b)):
        raise TypeError("mul expects tensor or scalar inputs")

    # Fast paths first: they avoid result-type inference, broadcast inference
    # and tensor coercion. On small tensors those costs dominate the launch.
    if isinstance(a, torch.Tensor):
        if isinstance(b, torch.Tensor):
            result = _fast_contiguous_mul(a, b, out)
        else:
            result = _fast_scalar_mul(a, b, out)
        if result is not None:
            return result
    elif isinstance(b, torch.Tensor):
        result = _fast_scalar_mul(b, a, out)
        if result is not None:
            return result

    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        if out is not None:
            return torch.ops.aten.mul.out.redispatch(_FALLBACK_KEYSET, a, b, out=out)
        return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

    dtype = _result_dtype(a, b)

    if (
        isinstance(a, torch.Tensor)
        and not isinstance(b, torch.Tensor)
        and not isinstance(b, complex)
    ):
        a_t = _as_tensor(a, device=device, dtype=dtype)
        output = _real_output(a_t, a_t.new_empty(()), out=out)
        if _can_use_contiguous_scalar(a_t, output):
            return _launch_scalar(a_t, b, output, dtype)

    if (
        isinstance(b, torch.Tensor)
        and not isinstance(a, torch.Tensor)
        and not isinstance(a, complex)
    ):
        b_t = _as_tensor(b, device=device, dtype=dtype)
        output = _real_output(b_t.new_empty(()), b_t, out=out)
        if _can_use_contiguous_scalar(b_t, output):
            return _launch_scalar(b_t, a, output, dtype)

    a_t = _as_tensor(a, device=device, dtype=dtype)
    b_t = _as_tensor(b, device=device, dtype=dtype)
    output = _real_output(a_t, b_t, out=out)

    if _can_use_contiguous_tensor_tensor(a_t, b_t, output):
        return _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype)

    out_shape = tuple(output.shape)
    a_stride, b_stride, out_stride = _real_layout(out_shape, a_t, b_t, output)
    if _can_use_2d_tile(out_shape, out_stride):
        return _launch_2d_tile(
            a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
        )
    if _can_use_2d_broadcast(out_shape):
        return _launch_2d_broadcast(
            a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
        )
    return _launch_generic(
        a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
    )


def _complex_parts(value, *, device, complex_dtype):
    tensor = _as_tensor(value, device=device, dtype=complex_dtype)
    real_view = torch.view_as_real(tensor)
    return real_view[..., 0], real_view[..., 1]


def _complex_output(out_shape, *, device, dtype, out=None):
    if out is None:
        return torch.empty(out_shape, device=device, dtype=dtype)
    if tuple(out.shape) != out_shape:
        raise RuntimeError(
            f"output with shape {tuple(out.shape)} cannot be broadcast to {out_shape}"
        )
    if out.dtype != dtype:
        raise RuntimeError(
            f"output dtype {out.dtype} does not match result dtype {dtype}"
        )
    if out.device != device:
        raise RuntimeError("output must be on the same device as inputs")
    return out


def _complex_layout(out_shape, tensors, out_r, out_i):
    strides = [
        _broadcasted_stride(tuple(tensor.shape), tuple(tensor.stride()), out_shape)
        for tensor in tensors
    ]
    return (
        strides,
        tuple(out_r.stride()) if out_shape else (),
        tuple(out_i.stride()) if out_shape else (),
    )


def _launch_complex_generic(
    ar, ai, br, bi, output, out_shape, strides, out_r_stride, out_i_stride, dtype
):
    n_elements = volume(out_shape)
    if n_elements == 0:
        return output
    out_view = torch.view_as_real(output)
    out_r = out_view[..., 0]
    out_i = out_view[..., 1]
    shape = out_shape or (1,)
    ndim = len(shape)
    strides = [stride or (0,) for stride in strides]
    out_r_stride = out_r_stride or (0,)
    out_i_stride = out_i_stride or (0,)
    # Fixed reference block for the grid size so the kernel GRID_SIZE
    # argument (same expression below) always matches grid dim 0,
    # independent of which BLOCK_SIZE the libtuner picks.
    grid = lambda meta: (min(_GRID_SIZE, triton.cdiv(n_elements, 1024)), 1, 1)
    if _needs_runtime_meta_for_constexpr_tuple():
        meta = torch.tensor(
            shape
            + strides[0]
            + strides[1]
            + strides[2]
            + strides[3]
            + out_r_stride
            + out_i_stride,
            device=output.device,
            dtype=torch.int64,
        )
        with torch_device_fn.device(output.device):
            mul_complex_generic_nd_runtime_meta_kernel[grid](
                ar,
                ai,
                br,
                bi,
                out_r,
                out_i,
                meta,
                n_elements,
                NDIM=ndim,
                dtype=_dtype_name(dtype),
                GRID_SIZE=min(_GRID_SIZE, triton.cdiv(n_elements, 1024)),
                NUM_STAGES=4,
            )
        return output

    with torch_device_fn.device(output.device):
        mul_complex_generic_nd_kernel[grid](
            ar,
            ai,
            br,
            bi,
            out_r,
            out_i,
            n_elements,
            SHAPE=shape,
            AR_STRIDE=strides[0],
            AI_STRIDE=strides[1],
            BR_STRIDE=strides[2],
            BI_STRIDE=strides[3],
            OUT_R_STRIDE=out_r_stride,
            OUT_I_STRIDE=out_i_stride,
            NDIM=ndim,
            dtype=_dtype_name(dtype),
            GRID_SIZE=min(_GRID_SIZE, triton.cdiv(n_elements, 1024)),
            NUM_STAGES=4,
        )
    return output


def mul_complex_broadcast_func(a, b, out=None):
    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        if out is not None:
            return torch.ops.aten.mul.out.redispatch(_FALLBACK_KEYSET, a, b, out=out)
        return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

    dtype = _result_dtype(a, b)
    ar, ai = _complex_parts(a, device=device, complex_dtype=dtype)
    br, bi = _complex_parts(b, device=device, complex_dtype=dtype)
    out_shape = tuple(
        torch.broadcast_shapes(
            tuple(ar.shape), tuple(ai.shape), tuple(br.shape), tuple(bi.shape)
        )
    )
    output = _complex_output(out_shape, device=device, dtype=dtype, out=out)
    out_view = torch.view_as_real(output)
    strides, out_r_stride, out_i_stride = _complex_layout(
        out_shape, [ar, ai, br, bi], out_view[..., 0], out_view[..., 1]
    )
    return _launch_complex_generic(
        ar, ai, br, bi, output, out_shape, strides, out_r_stride, out_i_stride, dtype
    )


def mul(A, B, *, out=None):
    logger.debug("GEMS_ENFLAME MUL")
    if isinstance(A, torch.Tensor) or isinstance(B, torch.Tensor):
        if (
            (isinstance(A, torch.Tensor) and A.is_complex())
            or (isinstance(B, torch.Tensor) and B.is_complex())
            or isinstance(A, complex)
            or isinstance(B, complex)
        ):
            return mul_complex_broadcast_func(A, B, out=out)
        return mul_broadcast_func(A, B, out=out)
    return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_ENFLAME MUL_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("mul_ expects the first argument to be a tensor")
    dtype = _result_dtype(A, B)
    if dtype != A.dtype:
        raise RuntimeError(
            f"result type {dtype} cannot be cast to inplace dtype {A.dtype}"
        )
    if (
        A.is_complex()
        or (isinstance(B, torch.Tensor) and B.is_complex())
        or isinstance(B, complex)
    ):
        return mul_complex_broadcast_func(A, B, out=A)
    return mul_broadcast_func(A, B, out=A)
