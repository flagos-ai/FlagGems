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
#
# Ascend implementation imported from the tuned operator batch
# host entry points below mirror the generic
# flag_gems names so the backend registrar can pick them up.

import logging
import math
import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Rounding uses the symmetric magic constant ``1.5 * 2**23``: ``(y + C) - C``
# rounds to nearest-even in a single add/subtract pair for ``|y| < 2**22`` and
# works for both signs, so it replaces the previous ``|y|``-based chain and its
# abs / negate / select (3 vector ops per element).  Beyond ``2**22`` the sum
# leaves the ulp==1 window, but such magnitudes are far outside every
# quantisation range and saturate in the clamp below anyway.  inf/-inf/NaN pass
# through unchanged, which is what the range checks below expect.
_MAGIC_ROUND = tl.constexpr(12582912.0)
# 910B4: 40 vector cores, 168 MiB L2.  One program per vector core driving a
# grid-stride loop beats one program per tile by 1.6-1.9x at 16.7M/67M elements
# (per-program fixed cost is ~67 ns).  Launching more programs than cores is
# worse than leaving cores idle, so the grid never exceeds the core count, which
# is read from the device (the value is SoC-specific) and cached.
_CORES_FALLBACK = 40
_CORES_CACHE = {}
# Largest tile that fits the 196 KB UB: 8192 fp32 lanes for 16-bit inputs (one
# upcast stream), 4096 for fp32 inputs (fp32 sources stay 32-bit); 16384
# overflows with this many live temporaries.
_BLOCK_MAX_16BIT = 8192
_BLOCK_MAX_32BIT = 4096
_BLOCK_MIN = 1024
# Tile offsets are int32 inside the kernel, so one launch covers at most this
# many elements; larger tensors are split into slices.
_MAX_ELEMS_PER_LAUNCH = 1 << 30


@triton.jit
def _fake_quantize_cachemask_body(
    input_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    base,
    scale,
    inv_scale,
    q_lo,
    q_hi,
    BLOCK_SIZE: tl.constexpr,
):
    # q_lo / q_hi are quant_min - zero_point and quant_max - zero_point, folded
    # on the host so the kernel never adds/subtracts zero_point.  The clamp is
    # expressed as two selects reusing the two range compares that also produce
    # the cachemask, so the range test costs no extra vector ops.
    offsets = base + tl.arange(0, BLOCK_SIZE)
    active = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=active).to(tl.float32)
    y = value * inv_scale.to(tl.float32)
    rounded = (y + _MAGIC_ROUND) - _MAGIC_ROUND
    below_hi = rounded <= q_hi
    above_lo = rounded >= q_lo
    cachemask = above_lo & below_hi
    quantized = tl.where(below_hi, rounded, q_hi)
    quantized = tl.where(above_lo, quantized, q_lo)
    output = quantized * scale.to(tl.float32)
    tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=active)
    tl.store(mask_ptr + offsets, cachemask, mask=active)


@libentry()
@triton.jit(
    do_not_specialize=["n_elements", "num_tiles", "scale", "inv_scale", "q_lo", "q_hi"]
)
def fake_quantize_per_tensor_affine_cachemask_kernel(
    input_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    num_tiles,
    scale,
    inv_scale,
    q_lo,
    q_hi,
    BLOCK_SIZE: tl.constexpr,
    GRID_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if GRID_STRIDE:
        for tile in tl.range(pid, num_tiles, tl.num_programs(0)):
            _fake_quantize_cachemask_body(
                input_ptr,
                output_ptr,
                mask_ptr,
                n_elements,
                tile * BLOCK_SIZE,
                scale,
                inv_scale,
                q_lo,
                q_hi,
                BLOCK_SIZE,
            )
    else:
        _fake_quantize_cachemask_body(
            input_ptr,
            output_ptr,
            mask_ptr,
            n_elements,
            pid * BLOCK_SIZE,
            scale,
            inv_scale,
            q_lo,
            q_hi,
            BLOCK_SIZE,
        )


def _validate_and_convert_qparams(scale, zero_point, quant_min, quant_max):
    if quant_min > quant_max:
        raise RuntimeError("quant_min must be less than or equal to quant_max")
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError("zero_point must be between quant_min and quant_max")
    scale = struct.unpack("f", struct.pack("f", float(scale)))[0]
    if scale == 0.0:
        inv_scale = math.copysign(math.inf, scale)
    else:
        inv_scale = struct.unpack("f", struct.pack("f", 1.0 / scale))[0]
    return (scale, inv_scale)


def _pick_block(n_elements, element_size):
    """Widest UB-fitting tile, narrowed so small tensors still get >=16 programs.

    Small tensors are launch/program-count bound (n=16: 1024 lanes -> 1.4us,
    8192 lanes -> 3.1us); for n >= 128K the widest tile wins by >5x over the old
    1024-lane one because the per-tile cost dominates otherwise.
    """
    block_max = _BLOCK_MAX_16BIT if element_size <= 2 else _BLOCK_MAX_32BIT
    block_size = _BLOCK_MIN
    while block_size < block_max and block_size * 16 <= n_elements:
        block_size <<= 1
    return block_size


def _vector_cores(device):
    """Vector-core count of ``device`` (cached); 40 on a 910B4."""
    key = device.index
    cores = _CORES_CACHE.get(key)
    if cores is None:
        cores = _CORES_FALLBACK
        try:
            props = torch_device_fn.get_device_properties(device)
            cores = int(getattr(props, "vector_core_num", 0)) or _CORES_FALLBACK
        except Exception:  # pragma: no cover - backend without the property
            pass
        _CORES_CACHE[key] = cores
    return cores


def _pick_grid(num_tiles, cores):
    """Fewest programs (<= vector cores) that still balance the tile loop.

    ``ceil(tiles / cores)`` iterations are needed, so spreading those iterations
    over ``ceil(tiles / iters)`` programs keeps every program busy and never
    launches more programs than there are vector cores (extra programs serialize
    into a second wave and cost 20-50%).
    """
    iters = (num_tiles + cores - 1) // cores
    return (num_tiles + iters - 1) // iters


def _launch_tile_range(inp, out, mask, n, scale, inv_scale, q_lo, q_hi):
    """One launch covering ``n`` contiguous elements of already-flat tensors."""
    block_size = _pick_block(n, inp.element_size())
    num_tiles = (n + block_size - 1) // block_size
    grid = _pick_grid(num_tiles, _vector_cores(inp.device))
    # grid <= vector cores, so the flattened coreDim can never overflow.
    fake_quantize_per_tensor_affine_cachemask_kernel[(grid,)](
        inp,
        out,
        mask,
        n,
        num_tiles,
        scale,
        inv_scale,
        q_lo,
        q_hi,
        BLOCK_SIZE=block_size,
        GRID_STRIDE=grid < num_tiles,
    )


def _launch(input, output, cachemask, scale, zero_point, quant_min, quant_max):
    scale, inv_scale = _validate_and_convert_qparams(
        scale, zero_point, quant_min, quant_max
    )
    contiguous_input = _strict_contiguous(input)
    n_elements = contiguous_input.numel()
    if n_elements == 0:
        return
    contiguous_output = output
    if not output.is_contiguous():
        contiguous_output = _strict_empty_like(contiguous_input)
    contiguous_mask = cachemask
    if not cachemask.is_contiguous():
        contiguous_mask = _strict_empty_like(contiguous_input, dtype=torch.bool)
    q_lo = float(int(quant_min) - int(zero_point))
    q_hi = float(int(quant_max) - int(zero_point))
    with torch_device_fn.device(input.device):
        if n_elements <= _MAX_ELEMS_PER_LAUNCH:
            _launch_tile_range(
                contiguous_input,
                contiguous_output,
                contiguous_mask,
                n_elements,
                scale,
                inv_scale,
                q_lo,
                q_hi,
            )
        else:
            # Tile offsets are int32 in the kernel; a single launch above
            # _MAX_ELEMS_PER_LAUNCH elements could overflow them.
            flat_input = _strict_flatten(contiguous_input)
            flat_output = _strict_flatten(contiguous_output)
            flat_mask = _strict_flatten(contiguous_mask)
            for start in range(0, n_elements, _MAX_ELEMS_PER_LAUNCH):
                end = min(start + _MAX_ELEMS_PER_LAUNCH, n_elements)
                _launch_tile_range(
                    flat_input[start:end],
                    flat_output[start:end],
                    flat_mask[start:end],
                    end - start,
                    scale,
                    inv_scale,
                    q_lo,
                    q_hi,
                )
    if contiguous_output is not output:
        _strict_copy_into(output, contiguous_output)
    if contiguous_mask is not cachemask:
        _strict_copy_into(cachemask, contiguous_mask)


def fake_quantize_per_tensor_affine_cachemask(
    input, scale, zero_point, quant_min, quant_max
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK")
    contiguous_input = _strict_contiguous(input)
    output = _strict_empty_like(contiguous_input)
    cachemask = _strict_empty_like(contiguous_input, dtype=torch.bool)
    _launch(
        contiguous_input, output, cachemask, scale, zero_point, quant_min, quant_max
    )
    return (output, cachemask)


def fake_quantize_per_tensor_affine_cachemask_out(
    input, scale, zero_point, quant_min, quant_max, *, out0, out1
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_OUT")
    if out0.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, but got {out0.dtype} instead"
        )
    if out1.dtype != torch.bool:
        raise RuntimeError(
            f"Expected out tensor to have dtype bool, but got {out1.dtype} instead"
        )
    if out0.device != input.device or out1.device != input.device:
        raise RuntimeError("input and out tensors must be on the same device")
    out0.resize_(input.shape)
    out1.resize_(input.shape)
    _launch(input, out0, out1, scale, zero_point, quant_min, quant_max)
    return (out0, out1)


run = fake_quantize_per_tensor_affine_cachemask_out

"""候选内嵌的Triton存储操作：复制、连续化、填充、转换均执行Triton核。"""


@triton.jit
def _strict_copy_kernel(
    X,
    Y,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remain = i
    xp = tl.full((BLOCK,), 0, tl.int64)
    yp = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remain % SHAPE[d]
        remain = remain // SHAPE[d]
        xp += coord * XS[d]
        yp += coord * YS[d]
    value = tl.load(X + xp, i < N, other=0)
    tl.store(Y + yp, value, i < N)


@triton.jit
def _strict_fill_kernel(Y, N: tl.constexpr, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, VALUE, i < N)


@triton.jit
def _strict_random_kernel(Y, N: tl.constexpr, SEED, OFFSET, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # 备用随机采样路径；全部随机数计算在Triton内，生成器仅在宿主管理状态。
    value = tl.rand(SEED, (i.to(tl.uint64) + OFFSET).to(tl.uint32))
    tl.store(Y + i, value, i < N)


def _strict_empty_like(x, *, dtype=None, **kwargs):
    return torch.empty(
        tuple(x.shape), device=x.device, dtype=x.dtype if dtype is None else dtype
    )


def _strict_copy_into(dst, src):
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError("strict copy requires equal shapes")
    if dst.numel():
        _strict_copy_kernel[(triton.cdiv(dst.numel(), 1024),)](
            src,
            dst,
            dst.numel(),
            tuple(dst.shape),
            tuple(src.stride()),
            tuple(dst.stride()),
            1024,
        )
    return dst


def _strict_clone(x, **kwargs):
    return _strict_copy_into(_strict_empty_like(x), x)


def _strict_contiguous(x, *args, **kwargs):
    return x if x.is_contiguous() else _strict_clone(x)


def _strict_cast(x, dtype):
    if dtype == x.dtype:
        return x
    return _strict_copy_into(_strict_empty_like(x, dtype=dtype), x)


def _strict_reshape(x, *shape):
    # view只操作元数据，输入连续化由上面的Triton复制完成。
    return _strict_contiguous(x).view(*shape)


def _strict_flatten(x, start_dim=0, end_dim=-1):
    rank = x.ndim
    first = start_dim % rank
    last = end_dim % rank
    size = 1
    for n in x.shape[first : last + 1]:
        size *= n
    shape = tuple(x.shape[:first]) + (size,) + tuple(x.shape[last + 1 :])
    return _strict_contiguous(x).view(shape)


def _strict_full(shape, value, **kwargs):
    if isinstance(shape, int):
        shape = (shape,)
    out = torch.empty(shape, **kwargs)
    if out.numel():
        _strict_fill_kernel[(triton.cdiv(out.numel(), 1024),)](
            out, out.numel(), value, 1024
        )
    return out


def _strict_zeros(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 0, **kwargs)


def _strict_ones(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 1, **kwargs)


def _strict_rand(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    generator = kwargs.pop("generator", None)
    out = torch.empty(shape, **kwargs)
    if out.numel():
        if generator is None:
            import torch_npu  # noqa: F401

            generator = torch.npu.default_generators[out.device.index or 0]
        seed = generator.initial_seed()
        offset = generator.get_offset()
        advance = triton.cdiv(out.numel(), 4) * 4
        if offset + advance >= 2**32:
            raise RuntimeError("备用随机路径超过32位计数范围，明确拒绝而非静默重复")
        generator.set_offset(offset + advance)
        _strict_random_kernel[(triton.cdiv(out.numel(), 256),)](
            out, out.numel(), seed, offset, 256
        )
    return out
