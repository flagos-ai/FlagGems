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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 512
_BLOCK_DIM0 = 32
_BLOCK_DIM1 = 32
_MAX_TILED_RANK = 4
_MAX_TRITON_OFFSET = torch.iinfo(torch.int32).max

# gcu400 fast-path tuning (probe 2026-11-18, kernel do_bench w200/r100 on a
# clean L600; official-shape A/B vs native). The masked 32x32 tl.trans tiles
# above are kept as the correctness fallback for shapes the fast paths do not
# cover; every fast path is a single launch with EVEN (unmasked) affine tiles:
#   - 2D affine tile   [BM, BN] w1/w2 ns1, 48-CTA grid-stride: long unmasked
#     rows on both sides (compile-time contiguity, no predicate tax).
#   - fold3d tile      [BM, BX, BN] for dims (ndim-3, ndim-1) with one dim
#     between: the (mid, n) input axes and (mid, m) output axes are each one
#     contiguous run, so a 3D affine block coalesces both sides.
#   - word tile        [BM, BN, WORDS] for complex word views: the (n, w) and
#     (m, w) axis pairs are jointly contiguous on load and store.
_NPROG_CAP = 48
_MAX_TILE_BYTES = 128 * 1024
_FLAT_WORD_BLOCK = 8192


@triton.jit
def _complex_copy_kernel(
    input,
    out,
    n_words,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    WORDS: tl.constexpr,
    CONJ: tl.constexpr,
    NEG: tl.constexpr,
    SIGN_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    word = offsets % WORDS
    index = offsets // WORDS
    source = tl.full((BLOCK_SIZE,), 0, tl.int64)
    # Older vendor compilers require explicit access to the constexpr payload.
    for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
        source += (index % SHAPE.value[dim]) * STRIDES.value[dim]
        index //= SHAPE.value[dim]
    values = tl.load(input + source * WORDS + word, offsets < n_words, other=0)
    # The last word of each IEEE component contains its sign bit.
    flip = ((word == WORDS // 2 - 1) & NEG) | ((word == WORDS - 1) & (NEG != CONJ))
    values ^= tl.where(flip, SIGN_MASK, 0).to(values.dtype)
    tl.store(out + offsets, values, offsets < n_words)


@triton.jit
def _complex_transpose_tiled_kernel(
    input,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    OUTPUT_STRIDES: tl.constexpr,
    OTHER_SHAPE: tl.constexpr,
    OTHER_INPUT_STRIDES: tl.constexpr,
    OTHER_OUTPUT_STRIDES: tl.constexpr,
    WORDS: tl.constexpr,
    CONJ: tl.constexpr,
    NEG: tl.constexpr,
    SIGN_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tiles_n = tl.cdiv(N, BLOCK)
    tiles = tl.cdiv(M, BLOCK) * tiles_n
    batch = pid // tiles
    tile = pid % tiles
    input_base = tl.full((), 0, tl.int64)
    output_base = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(OTHER_SHAPE) - 1, -1, -1):
        coord = batch % OTHER_SHAPE.value[dim]
        batch //= OTHER_SHAPE.value[dim]
        input_base += coord * OTHER_INPUT_STRIDES.value[dim]
        output_base += coord * OTHER_OUTPUT_STRIDES.value[dim]
    m = tile // tiles_n * BLOCK + tl.arange(0, BLOCK)
    n_words = tl.arange(0, BLOCK * WORDS)
    n = tile % tiles_n * BLOCK + n_words // WORDS
    word = n_words % WORDS
    source = (
        input_base
        + m[:, None] * INPUT_STRIDES.value[0]
        + n[None, :] * INPUT_STRIDES.value[1]
    )
    values = tl.load(
        input + source * WORDS + word[None, :],
        (m[:, None] < M) & (n[None, :] < N),
        other=0,
    )
    flip = ((word == WORDS // 2 - 1) & NEG) | ((word == WORDS - 1) & (NEG != CONJ))
    values ^= tl.where(flip[None, :], SIGN_MASK, 0).to(values.dtype)
    target = (
        output_base
        + n[:, None] * OUTPUT_STRIDES.value[0]
        + m[None, :] * OUTPUT_STRIDES.value[1]
    )
    tl.store(
        out + target * WORDS + word[:, None],
        tl.trans(values, 1, 0),
        (n[:, None] < N) & (m[None, :] < M),
    )


@triton.jit
def _copy_kernel(
    input,
    out,
    input_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input + offsets * input_stride, mask=mask, other=0)
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def _transpose_copy_tiled_kernel(
    input,
    out,
    input_stride_dim0,
    input_stride_dim1,
    out_stride_dim0,
    out_stride_dim1,
    input_stride_other0,
    input_stride_other1,
    out_stride_other0,
    out_stride_other1,
    other_size1,
    dim0_size,
    dim1_size,
    tiles_dim1,
    tiles_per_slice,
    BLOCK_DIM0: tl.constexpr,
    BLOCK_DIM1: tl.constexpr,
):
    program_id = tl.program_id(0)
    slice_id = program_id // tiles_per_slice
    tile_id = program_id - slice_id * tiles_per_slice
    tile_dim0 = tile_id // tiles_dim1
    tile_dim1 = tile_id - tile_dim0 * tiles_dim1

    other_index0 = slice_id // other_size1
    other_index1 = slice_id - other_index0 * other_size1
    input_base = other_index0 * input_stride_other0 + other_index1 * input_stride_other1
    out_base = other_index0 * out_stride_other0 + other_index1 * out_stride_other1

    offsets_dim0 = tile_dim0 * BLOCK_DIM0 + tl.arange(0, BLOCK_DIM0)
    offsets_dim1 = tile_dim1 * BLOCK_DIM1 + tl.arange(0, BLOCK_DIM1)
    mask_dim0 = offsets_dim0 < dim0_size
    mask_dim1 = offsets_dim1 < dim1_size

    input_offsets = (
        input_base
        + offsets_dim0[:, None] * input_stride_dim0
        + offsets_dim1[None, :] * input_stride_dim1
    )
    values = tl.load(
        input + input_offsets,
        mask=mask_dim0[:, None] & mask_dim1[None, :],
        other=0,
    )

    out_offsets = (
        out_base
        + offsets_dim1[:, None] * out_stride_dim0
        + offsets_dim0[None, :] * out_stride_dim1
    )
    tl.store(
        out + out_offsets,
        tl.trans(values, 1, 0),
        mask=mask_dim1[:, None] & mask_dim0[None, :],
    )


@triton.jit
def _flat_word_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    # Grid-stride int32/uint8 word copy (same form as the gcu400 copy.py fast
    # path): masked exactly, so it can never overrun the allocation.
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in tl.range(pid, num_tiles, GRID_SIZE):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(src_ptr + offsets, mask=mask, other=0)
        tl.store(dst_ptr + offsets, value, mask=mask)


@triton.jit
def _affine_transpose_kernel(
    input,
    out,
    s_in0,
    s_in1,
    s_out0,
    s_out1,
    s_in_o0,
    s_in_o1,
    s_out_o0,
    s_out_o1,
    other_size1,
    tiles_d1,
    tiles_per_slice,
    total_tiles,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
):
    # EVEN (unmasked) plain-affine tile: block axis 0 runs along the input's
    # dim0 (output-contiguous direction), axis 1 along the input-contiguous
    # dim1. No tl.trans: the value block is stored with swapped index roles,
    # which keeps both sides naturally coalesced.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for tile_id in tl.range(pid, total_tiles, num_progs):
        slice_id = tile_id // tiles_per_slice
        tile = tile_id - slice_id * tiles_per_slice
        td0 = tile // tiles_d1
        td1 = tile - td0 * tiles_d1
        oi0 = slice_id // other_size1
        oi1 = slice_id - oi0 * other_size1
        ibase = oi0 * s_in_o0 + oi1 * s_in_o1
        obase = oi0 * s_out_o0 + oi1 * s_out_o1
        om0 = td0 * BLOCK_D0 + tl.arange(0, BLOCK_D0)
        on1 = td1 * BLOCK_D1 + tl.arange(0, BLOCK_D1)
        src = ibase + om0[:, None] * s_in0 + on1[None, :] * s_in1
        values = tl.load(input + src)
        dst = obase + on1[None, :] * s_out0 + om0[:, None] * s_out1
        tl.store(out + dst, values)


@triton.jit
def _fold3d_transpose_kernel(
    input,
    out,
    s_m,
    s_x,
    s_n,
    o_n,
    o_x,
    o_m,
    s_b,
    o_b,
    tiles_n,
    tiles_x,
    tiles_per_b,
    total_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # For dims (ndim-3, ndim-1) with exactly one dim between, the (mid, n)
    # axes are one contiguous run on the input and the (mid, m) axes one
    # contiguous run on the output. A 3D affine block over (m, mid, n)
    # therefore coalesces both sides without any in-register transpose.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for tile_id in tl.range(pid, total_tiles, num_progs):
        b0 = tile_id // tiles_per_b
        rem = tile_id - b0 * tiles_per_b
        tx = rem // tiles_n
        tn = rem - tx * tiles_n
        om = tl.arange(0, BLOCK_M)
        ox = tx * BLOCK_X + tl.arange(0, BLOCK_X)
        on = tn * BLOCK_N + tl.arange(0, BLOCK_N)
        src = (
            b0 * s_b
            + om[:, None, None] * s_m
            + ox[None, :, None] * s_x
            + on[None, None, :] * s_n
        )
        values = tl.load(input + src)
        dst = (
            b0 * o_b
            + on[None, None, :] * o_n
            + ox[None, :, None] * o_x
            + om[:, None, None] * o_m
        )
        tl.store(out + dst, values)


@triton.jit
def _word_transpose_kernel(
    input,
    out,
    s_in0,
    s_in1,
    s_out0,
    s_out1,
    s_in_o0,
    s_in_o1,
    s_out_o0,
    s_out_o1,
    other_size1,
    tiles_d1,
    tiles_per_slice,
    total_tiles,
    WORDS: tl.constexpr,
    CONJ: tl.constexpr,
    NEG: tl.constexpr,
    SIGN_MASK: tl.constexpr,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
):
    # Complex word view as a 3D affine (m, n, word) block. With a unit
    # element stride along dim1 the (n, word) axes load as one contiguous
    # run and the (m, word) axes store as one contiguous run. Conj/neg sign
    # bits live in the last word of each IEEE component.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for tile_id in tl.range(pid, total_tiles, num_progs):
        slice_id = tile_id // tiles_per_slice
        tile = tile_id - slice_id * tiles_per_slice
        td0 = tile // tiles_d1
        td1 = tile - td0 * tiles_d1
        oi0 = slice_id // other_size1
        oi1 = slice_id - oi0 * other_size1
        ibase = oi0 * s_in_o0 + oi1 * s_in_o1
        obase = oi0 * s_out_o0 + oi1 * s_out_o1
        om0 = td0 * BLOCK_D0 + tl.arange(0, BLOCK_D0)
        on1 = td1 * BLOCK_D1 + tl.arange(0, BLOCK_D1)
        w = tl.arange(0, WORDS)
        src = (
            ibase
            + om0[:, None, None] * s_in0
            + on1[None, :, None] * s_in1
            + w[None, None, :]
        )
        values = tl.load(input + src)
        if NEG or CONJ:
            word = w[None, None, :]
            flip = ((word == WORDS // 2 - 1) & NEG) | (
                (word == WORDS - 1) & (NEG != CONJ)
            )
            values ^= tl.where(flip, SIGN_MASK, 0).to(values.dtype)
        dst = (
            obase
            + on1[None, :, None] * s_out0
            + om0[:, None, None] * s_out1
            + w[None, None, :]
        )
        tl.store(out + dst, values)


@triton.jit
def _fold4d_word_transpose_kernel(
    input,
    out,
    s_m,
    s_x,
    s_n,
    o_n,
    o_x,
    o_m,
    s_b,
    o_b,
    tiles_n,
    tiles_x,
    tiles_per_b,
    total_tiles,
    WORDS: tl.constexpr,
    CONJ: tl.constexpr,
    NEG: tl.constexpr,
    SIGN_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Word-view companion of _fold3d_transpose_kernel for dims (ndim-3,
    # ndim-1) with one dim between: the (mid, n) input axes and (mid, m)
    # output axes are contiguous runs in word space, with the word axis
    # jointly contiguous on both sides.
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for tile_id in tl.range(pid, total_tiles, num_progs):
        b0 = tile_id // tiles_per_b
        rem = tile_id - b0 * tiles_per_b
        tx = rem // tiles_n
        tn = rem - tx * tiles_n
        om = tl.arange(0, BLOCK_M)
        ox = tx * BLOCK_X + tl.arange(0, BLOCK_X)
        on = tn * BLOCK_N + tl.arange(0, BLOCK_N)
        w = tl.arange(0, WORDS)
        src = (
            b0 * s_b
            + om[:, None, None, None] * s_m
            + ox[None, :, None, None] * s_x
            + on[None, None, :, None] * s_n
            + w[None, None, None, :]
        )
        values = tl.load(input + src)
        if NEG or CONJ:
            word = w[None, None, None, :]
            flip = ((word == WORDS // 2 - 1) & NEG) | (
                (word == WORDS - 1) & (NEG != CONJ)
            )
            values ^= tl.where(flip, SIGN_MASK, 0).to(values.dtype)
        dst = (
            b0 * o_b
            + on[None, None, :, None] * o_n
            + ox[None, :, None, None] * o_x
            + om[:, None, None, None] * o_m
            + w[None, None, None, :]
        )
        tl.store(out + dst, values)


def _normalize_dim(dim: int, ndim: int) -> int:
    lower = -ndim if ndim > 0 else -1
    upper = ndim - 1 if ndim > 0 else 0
    if dim < lower or dim > upper:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[{lower}, {upper}], but got {dim})"
        )
    if ndim == 0:
        return 0
    return dim % ndim


def _is_float8(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8_")


def _has_lazy_metadata(input: torch.Tensor) -> bool:
    is_neg = getattr(input, "is_neg", lambda: False)
    return input.is_conj() or is_neg()


_I64_NARROWED = None


def _int64_narrowed(sample: torch.Tensor) -> bool:
    """Detect torch_gcu silent int64 -> int32 word narrowing (cached once per process)."""
    global _I64_NARROWED
    if _I64_NARROWED is None:
        probe = torch.tensor([(1 << 32) + 1], dtype=torch.int64, device="cpu")
        _I64_NARROWED = probe.to(sample.device).cpu().item() != (1 << 32) + 1
    return _I64_NARROWED


def _narrowed_i32_alias(t: torch.Tensor) -> torch.Tensor:
    """Narrowed-layout int64/uint64 tensor -> same-shape int32 view (element i == word i)."""
    alias = torch.empty(0, dtype=torch.int32, device=t.device)
    alias.set_(t.untyped_storage(), t.storage_offset(), t.shape, t.stride())
    return alias


def _can_use_triton_storage(input: torch.Tensor) -> bool:
    if input.device.type == "cpu" or input.layout != torch.strided:
        return False
    if input.is_quantized:
        return False
    if _has_lazy_metadata(input):
        return False
    if input.numel() > _MAX_TRITON_OFFSET:
        return False
    max_input_offset = sum(
        (size - 1) * stride for size, stride in zip(input.shape, input.stride())
    )
    return max_input_offset <= _MAX_TRITON_OFFSET


def _can_use_triton(input: torch.Tensor) -> bool:
    if input.is_complex() or _is_float8(input.dtype):
        return False
    return _can_use_triton_storage(input)


def _can_use_byte_triton(input: torch.Tensor) -> bool:
    # FP8 transpose is bitwise, so uint8 pointers avoid FP8 scalar codegen.
    return (
        _is_float8(input.dtype)
        and input.element_size() == 1
        and _can_use_triton_storage(input)
    )


def _view_as_uint8(input: torch.Tensor) -> torch.Tensor:
    # dtype views require a logical dimension on some PyTorch builds.
    if input.ndim == 0:
        return input.reshape(1).view(torch.uint8)
    return input.view(torch.uint8)


def _fallback_transpose_copy(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return input.transpose(dim0, dim1).clone(memory_format=torch.contiguous_format)


def _largest_pow2_divisor(n: int, cap: int) -> int:
    p = cap
    while p > 1 and n % p != 0:
        p //= 2
    return p


def _affine_tile(d0_size: int, d1_size: int, esize: int):
    """Pick an EVEN [BM, BN] affine tile; None falls back to the masked kernel."""
    bm = _largest_pow2_divisor(d0_size, 128)
    bn = _largest_pow2_divisor(d1_size, 256)
    if bm < 8 or bn < 8:
        return None
    while bm * bn * esize > 64 * 1024:
        if bn > 64:
            bn //= 2
        elif bm > 16:
            bm //= 2
        else:
            return None
    return bm, bn


def _fold_plan(d0_size: int, mid_size: int, d1_size: int, esize: int):
    """Pick (BX, BN) for the fold kernels under a 64KB block budget."""
    bn = _largest_pow2_divisor(d1_size, 256)
    bx = _largest_pow2_divisor(mid_size, 8)
    while bx >= 8 and d0_size * bx * bn * esize > 64 * 1024:
        bx //= 2
    if bx < 8:
        return None
    return bx, bn


def _affine_num_warps(esize: int) -> int:
    # Probed sweet spot: w1 for <=2B lanes, w2 for 4-8B lanes.
    return 2 if esize >= 4 else 1


def _word_view_plan(input: torch.Tensor):
    """Return ``(n_words, word_dtype)`` for a flat word copy, or None."""
    esize = input.element_size()
    if (input.numel() * esize) % 4 == 0 and (input.storage_offset() * esize) % 4 == 0:
        return input.numel() * esize // 4, torch.int32
    if esize == 1:
        return input.numel(), torch.uint8
    return None


def _tensor_shape(t):
    return tuple(t.shape)


def _tensor_strides(t):
    return tuple(t.stride())


def _launch_flat_word_copy(src, dst, n_words, device):
    grid = (min(_NPROG_CAP, triton.cdiv(n_words, _FLAT_WORD_BLOCK)),)
    with torch_device_fn.device(device):
        _flat_word_copy_kernel[grid](
            src,
            dst,
            n_words,
            BLOCK_SIZE=_FLAT_WORD_BLOCK,
            GRID_SIZE=_NPROG_CAP,
            num_warps=4,
        )


def _launch_masked_tiled_transpose_copy(input, out, dim0, dim1, device):
    # Original masked 32x32 tl.trans kernel: correctness fallback for every
    # shape the EVEN fast paths do not cover (odd sizes, mid-dim pairs).
    other_dims = [
        dim for dim in range(len(_tensor_shape(input))) if dim not in (dim0, dim1)
    ]
    other_sizes = [_tensor_shape(input)[dim] for dim in other_dims]
    input_other_strides = [_tensor_strides(input)[dim] for dim in other_dims]
    out_other_strides = [_tensor_strides(out)[dim] for dim in other_dims]

    while len(other_dims) < _MAX_TILED_RANK - 2:
        other_sizes.insert(0, 1)
        input_other_strides.insert(0, 0)
        out_other_strides.insert(0, 0)
        other_dims.insert(0, -1)

    dim0_size = _tensor_shape(input)[dim0]
    dim1_size = _tensor_shape(input)[dim1]
    tiles_dim0 = triton.cdiv(dim0_size, _BLOCK_DIM0)
    tiles_dim1 = triton.cdiv(dim1_size, _BLOCK_DIM1)
    tiles_per_slice = tiles_dim0 * tiles_dim1
    num_slices = other_sizes[0] * other_sizes[1]
    grid = (num_slices * tiles_per_slice,)

    with torch_device_fn.device(device):
        _transpose_copy_tiled_kernel[grid](
            input,
            out,
            _tensor_strides(input)[dim0],
            _tensor_strides(input)[dim1],
            _tensor_strides(out)[dim0],
            _tensor_strides(out)[dim1],
            input_other_strides[0],
            input_other_strides[1],
            out_other_strides[0],
            out_other_strides[1],
            other_sizes[1],
            dim0_size,
            dim1_size,
            tiles_dim1,
            tiles_per_slice,
            BLOCK_DIM0=_BLOCK_DIM0,
            BLOCK_DIM1=_BLOCK_DIM1,
        )


def _launch_affine_transpose(input, out, dim0, dim1, bm, bn, esize, device):
    in_shape = _tensor_shape(input)
    in_strides = _tensor_strides(input)
    out_strides = _tensor_strides(out)
    other_dims = [dim for dim in range(len(in_shape)) if dim not in (dim0, dim1)]
    other_sizes = [in_shape[dim] for dim in other_dims]
    input_other_strides = [in_strides[dim] for dim in other_dims]
    out_other_strides = [out_strides[dim] for dim in other_dims]
    while len(other_dims) < _MAX_TILED_RANK - 2:
        other_sizes.insert(0, 1)
        input_other_strides.insert(0, 0)
        out_other_strides.insert(0, 0)
        other_dims.insert(0, -1)

    dim0_size = in_shape[dim0]
    dim1_size = in_shape[dim1]
    tiles_d0 = triton.cdiv(dim0_size, bm)
    tiles_d1 = triton.cdiv(dim1_size, bn)
    tiles_per_slice = tiles_d0 * tiles_d1
    total = other_sizes[0] * other_sizes[1] * tiles_per_slice
    grid = (min(_NPROG_CAP, total),)
    with torch_device_fn.device(device):
        _affine_transpose_kernel[grid](
            input,
            out,
            in_strides[dim0],
            in_strides[dim1],
            out_strides[dim0],
            out_strides[dim1],
            input_other_strides[0],
            input_other_strides[1],
            out_other_strides[0],
            out_other_strides[1],
            other_sizes[1],
            tiles_d1,
            tiles_per_slice,
            total,
            BLOCK_D0=bm,
            BLOCK_D1=bn,
            num_warps=_affine_num_warps(esize),
            num_stages=1,
        )


def _launch_fold3d_transpose(input, out, dim0, dim1, bx, device):
    in_shape = _tensor_shape(input)
    in_strides = _tensor_strides(input)
    out_strides = _tensor_strides(out)
    mid = dim0 + 1
    dim0_size = in_shape[dim0]
    mid_size = in_shape[mid]
    dim1_size = in_shape[dim1]
    if len(in_shape) == 4:
        batch_size = in_shape[0]
        s_b = in_strides[0]
        o_b = out_strides[0]
    else:
        batch_size = 1
        s_b = 0
        o_b = 0
    bn = _largest_pow2_divisor(dim1_size, 256)
    tiles_n = triton.cdiv(dim1_size, bn)
    tiles_x = triton.cdiv(mid_size, bx)
    tiles_per_b = tiles_n * tiles_x
    total = batch_size * tiles_per_b
    grid = (min(_NPROG_CAP, total),)
    with torch_device_fn.device(device):
        _fold3d_transpose_kernel[grid](
            input,
            out,
            in_strides[dim0],
            in_strides[mid],
            in_strides[dim1],
            out_strides[dim0],
            out_strides[mid],
            out_strides[dim1],
            s_b,
            o_b,
            tiles_n,
            tiles_x,
            tiles_per_b,
            total,
            BLOCK_M=dim0_size,
            BLOCK_X=bx,
            BLOCK_N=bn,
            num_warps=1,
            num_stages=1,
        )


def _launch_fold4d_word_transpose(
    kernel_input,
    kernel_out,
    dim0,
    dim1,
    words,
    conj,
    neg,
    sign_mask,
    bx,
    bn,
    device,
):
    in_shape = _tensor_shape(kernel_input)
    in_strides = _tensor_strides(kernel_input)
    out_strides = _tensor_strides(kernel_out)
    mid = dim0 + 1
    dim0_size = in_shape[dim0]
    mid_size = in_shape[mid]
    dim1_size = in_shape[dim1]
    if len(in_shape) == 5:
        batch_size = in_shape[0]
        s_b = in_strides[0]
        o_b = out_strides[0]
    else:
        batch_size = 1
        s_b = 0
        o_b = 0
    tiles_n = triton.cdiv(dim1_size, bn)
    tiles_x = triton.cdiv(mid_size, bx)
    tiles_per_b = tiles_n * tiles_x
    total = batch_size * tiles_per_b
    grid = (min(_NPROG_CAP, total),)
    with torch_device_fn.device(device):
        _fold4d_word_transpose_kernel[grid](
            kernel_input,
            kernel_out,
            in_strides[dim0],
            in_strides[mid],
            in_strides[dim1],
            out_strides[dim0],
            out_strides[mid],
            out_strides[dim1],
            s_b,
            o_b,
            tiles_n,
            tiles_x,
            tiles_per_b,
            total,
            WORDS=words,
            CONJ=conj,
            NEG=neg,
            SIGN_MASK=sign_mask,
            BLOCK_M=dim0_size,
            BLOCK_X=bx,
            BLOCK_N=bn,
            num_warps=1,
            num_stages=1,
        )


def _launch_word_transpose(
    kernel_input,
    kernel_out,
    dim0,
    dim1,
    words,
    conj,
    neg,
    sign_mask,
    bm,
    bn,
    device,
):
    in_shape = _tensor_shape(kernel_input)
    in_strides = _tensor_strides(kernel_input)
    out_strides = _tensor_strides(kernel_out)
    # The trailing word axis is addressed explicitly inside the kernel; the
    # slice decomposition only covers the element dims.
    other_dims = [dim for dim in range(len(in_shape) - 1) if dim not in (dim0, dim1)]
    other_sizes = [in_shape[dim] for dim in other_dims]
    input_other_strides = [in_strides[dim] for dim in other_dims]
    out_other_strides = [out_strides[dim] for dim in other_dims]
    while len(other_dims) < _MAX_TILED_RANK - 2:
        other_sizes.insert(0, 1)
        input_other_strides.insert(0, 0)
        out_other_strides.insert(0, 0)
        other_dims.insert(0, -1)

    tiles_d0 = triton.cdiv(in_shape[dim0], bm)
    tiles_d1 = triton.cdiv(in_shape[dim1], bn)
    tiles_per_slice = tiles_d0 * tiles_d1
    total = other_sizes[0] * other_sizes[1] * tiles_per_slice
    grid = (min(_NPROG_CAP, total),)
    with torch_device_fn.device(device):
        _word_transpose_kernel[grid](
            kernel_input,
            kernel_out,
            in_strides[dim0],
            in_strides[dim1],
            out_strides[dim0],
            out_strides[dim1],
            input_other_strides[0],
            input_other_strides[1],
            out_other_strides[0],
            out_other_strides[1],
            other_sizes[1],
            tiles_d1,
            tiles_per_slice,
            total,
            WORDS=words,
            CONJ=conj,
            NEG=neg,
            SIGN_MASK=sign_mask,
            BLOCK_D0=bm,
            BLOCK_D1=bn,
            num_warps=_affine_num_warps(kernel_input.element_size()),
            num_stages=1,
        )


def _launch_tiled_transpose_copy(input, out, dim0, dim1, device=None):
    """Route a dim0 != dim1 transpose to the fastest applicable kernel."""
    if device is None:
        device = input.device
    in_shape = _tensor_shape(input)
    in_strides = _tensor_strides(input)
    ndim = len(in_shape)
    dim0_size = in_shape[dim0]
    dim1_size = in_shape[dim1]
    esize = input.element_size()

    if (
        dim0_size > 1
        and dim1_size > 1
        and ndim <= _MAX_TILED_RANK
        and dim1 == ndim - 1
        and in_strides[dim1] == 1
    ):
        # fold3d: pair (ndim-3, ndim-1) with one dim between and a short
        # dim0, where plain 2D tiles leave sub-64B store runs.
        if (
            ndim >= 3
            and dim0 == ndim - 3
            and dim0_size in (2, 4, 8, 16, 32)
            and in_strides[dim0 + 1] == dim1_size
        ):
            plan = _fold_plan(dim0_size, in_shape[dim0 + 1], dim1_size, esize)
            if plan is not None:
                bx, bn = plan
                _launch_fold3d_transpose(input, out, dim0, dim1, bx, device)
                return out
        tile = _affine_tile(dim0_size, dim1_size, esize)
        if tile is not None:
            bm, bn = tile
            _launch_affine_transpose(input, out, dim0, dim1, bm, bn, esize, device)
            return out

    _launch_masked_tiled_transpose_copy(input, out, dim0, dim1, device)
    return out


def _launch_copy(input: torch.Tensor, out: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = input.device
    if input.is_contiguous():
        plan = _word_view_plan(input)
        if plan is not None:
            n_words, word_dtype = plan
            _launch_flat_word_copy(
                input.reshape(-1).view(word_dtype),
                out.reshape(-1).view(word_dtype),
                n_words,
                device,
            )
            return out
    n_elements = out.numel()
    input_stride = input.stride(0) if input.ndim == 1 else 1
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    with torch_device_fn.device(device):
        _copy_kernel[grid](
            input,
            out,
            input_stride,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    return out


def _complex_word_views(input, out, word_bits, words):
    word_dtype = {16: torch.uint16, 32: torch.uint32, 64: torch.uint64}[word_bits]
    if words == 1:
        # Same-element-size reinterpretation: no trailing word axis, so the
        # generic real-tensor routing sees the true rank.
        kernel_input = StridedBuffer(
            input,
            shape=tuple(input.shape),
            strides=tuple(input.stride()),
            dtype=word_dtype,
        )
        kernel_out = StridedBuffer(
            out,
            shape=tuple(out.shape),
            strides=tuple(out.stride()),
            dtype=word_dtype,
        )
    else:
        kernel_input = StridedBuffer(
            input,
            shape=(*input.shape, words),
            strides=(*(stride * words for stride in input.stride()), 1),
            dtype=word_dtype,
        )
        kernel_out = StridedBuffer(
            out,
            shape=(*out.shape, words),
            strides=(*(stride * words for stride in out.stride()), 1),
            dtype=word_dtype,
        )
    return kernel_input, kernel_out


def _launch_complex_copy(input, out, dim0, dim1):
    # Integer pointers preserve all bits and do not require complex/FP64
    # arithmetic. The default word geometry uses the widest unit words so
    # complex32/64 land on the plain affine kernel; complex128 (no unit-word
    # view) and conj/neg metadata use 32-bit words with an explicit axis.
    esize = input.element_size()
    has_metadata = _has_lazy_metadata(input)
    if has_metadata:
        word_bits = 16 if esize == 4 else 32
    elif esize <= 4:
        word_bits = esize * 8
    elif esize == 8:
        word_bits = 64
    else:
        word_bits = 32
    word_dtype = {16: torch.uint16, 32: torch.uint32, 64: torch.uint64}[word_bits]
    words = esize * 8 // word_bits
    sign_mask = (1 << (word_bits - 1)) if has_metadata else 0
    kernel_input, kernel_out = _complex_word_views(input, out, word_bits, words)
    with torch_device_fn.device(input.device):
        if dim0 != dim1:
            dim0_size = input.shape[dim0]
            dim1_size = input.shape[dim1]
            fast_shape = (
                dim0_size > 1
                and dim1_size > 1
                and input.ndim <= _MAX_TILED_RANK
                and dim1 == input.ndim - 1
                and input.stride(dim1) == 1
            )
            if fast_shape and (
                input.ndim >= 3
                and dim0 == input.ndim - 3
                and dim0_size in (2, 4, 8, 16, 32)
                and input.stride(dim0 + 1) == dim1_size
            ):
                # Between-dims pattern: always run the word fold with 32-bit
                # words (64-bit fold tiles are pathological on gcu400).
                fold_bits = 32
                fold_words = max(1, esize * 8 // fold_bits)
                if fold_words == 1:
                    # complex32 reinterprets as one 32-bit word: fold3d /
                    # affine are safe on 4-byte lanes; anything else goes to
                    # the complex fallback kernel (the real masked kernel
                    # must not see StridedBuffer pointers).
                    plan = _fold_plan(
                        dim0_size, input.shape[dim0 + 1], dim1_size, esize
                    )
                    if plan is not None:
                        _launch_fold3d_transpose(
                            kernel_input, kernel_out, dim0, dim1, plan[0], input.device
                        )
                        return out
                    tile = _affine_tile(dim0_size, dim1_size, esize)
                    if tile is not None:
                        bm, bn = tile
                        _launch_affine_transpose(
                            kernel_input,
                            kernel_out,
                            dim0,
                            dim1,
                            bm,
                            bn,
                            esize,
                            input.device,
                        )
                        return out
                    _launch_complex_tiled_fallback(
                        input, out, dim0, dim1, words, word_bits
                    )
                    return out
                plan = _fold_plan(dim0_size, input.shape[dim0 + 1], dim1_size, esize)
                if plan is not None:
                    fold_in, fold_out = _complex_word_views(
                        input, out, fold_bits, fold_words
                    )
                    bx, bn = plan
                    _launch_fold4d_word_transpose(
                        fold_in,
                        fold_out,
                        dim0,
                        dim1,
                        fold_words,
                        input.is_conj(),
                        input.is_neg(),
                        (1 << 31) if has_metadata else 0,
                        bx,
                        bn,
                        input.device,
                    )
                    return out
                tile = _affine_tile(dim0_size, dim1_size, esize * 2)
                if tile is not None:
                    bm, bn = tile
                    _launch_word_transpose(
                        kernel_input,
                        kernel_out,
                        dim0,
                        dim1,
                        words,
                        input.is_conj(),
                        input.is_neg(),
                        sign_mask,
                        bm,
                        bn,
                        input.device,
                    )
                    return out
                _launch_complex_tiled_fallback(input, out, dim0, dim1, words, word_bits)
                return out
            if words == 1:
                if word_bits == 64 and input.numel() <= 16384:
                    # Small 8-byte-word tiles under-fill the grid; the 32-bit
                    # word kernel with small blocks parallelizes better.
                    small_bm = _largest_pow2_divisor(dim0_size, 16)
                    small_bn = _largest_pow2_divisor(dim1_size, 64)
                    if (
                        fast_shape
                        and small_bm >= 8
                        and small_bn >= 8
                        and small_bm * small_bn * 8 <= 64 * 1024
                    ):
                        small_in, small_out = _complex_word_views(input, out, 32, 2)
                        _launch_word_transpose(
                            small_in,
                            small_out,
                            dim0,
                            dim1,
                            2,
                            False,
                            False,
                            0,
                            small_bm,
                            small_bn,
                            input.device,
                        )
                        return out
                tile = _affine_tile(dim0_size, dim1_size, esize) if fast_shape else None
                if tile is not None:
                    bm, bn = tile
                    _launch_affine_transpose(
                        kernel_input,
                        kernel_out,
                        dim0,
                        dim1,
                        bm,
                        bn,
                        esize,
                        input.device,
                    )
                else:
                    # The real-dtype masked kernel lowers StridedBuffer
                    # pointers to block-pointer (DTE) transfers whose tail
                    # DMA rounds up to the tile width; the complex fallback
                    # kernel's int64 offset form is the safe generic path.
                    _launch_complex_tiled_fallback(
                        input, out, dim0, dim1, words, word_bits
                    )
            else:
                if fast_shape:
                    tile = _affine_tile(dim0_size, dim1_size, esize * 2)
                    if tile is not None:
                        bm, bn = tile
                        _launch_word_transpose(
                            kernel_input,
                            kernel_out,
                            dim0,
                            dim1,
                            words,
                            input.is_conj(),
                            input.is_neg(),
                            sign_mask,
                            bm,
                            bn,
                            input.device,
                        )
                        return out
                _launch_complex_tiled_fallback(input, out, dim0, dim1, words, word_bits)
        else:
            if input.is_contiguous() and not has_metadata:
                n_words = input.numel() * words
                flat_in = StridedBuffer(
                    input,
                    shape=(n_words,),
                    strides=(1,),
                    dtype=word_dtype,
                )
                flat_out = StridedBuffer(
                    out,
                    shape=(n_words,),
                    strides=(1,),
                    dtype=word_dtype,
                )
                _launch_flat_word_copy(flat_in, flat_out, n_words, input.device)
            else:
                _launch_complex_flat_fallback(input, out, dim0, dim1, words, word_bits)
    return out


def _launch_complex_tiled_fallback(input, out, dim0, dim1, words, word_bits):
    other_dims = [dim for dim in range(input.ndim) if dim not in (dim0, dim1)]
    grid = (
        triton.cdiv(input.shape[dim0], _BLOCK_DIM0)
        * triton.cdiv(input.shape[dim1], _BLOCK_DIM1)
        * (input.numel() // input.shape[dim0] // input.shape[dim1]),
    )
    kernel_input = StridedBuffer(
        input,
        shape=(*input.shape, words),
        strides=(*(stride * words for stride in input.stride()), 1),
        dtype={16: torch.uint16, 32: torch.uint32, 64: torch.uint64}[word_bits],
    )
    kernel_out = StridedBuffer(
        out,
        shape=(*out.shape, words),
        strides=(*(stride * words for stride in out.stride()), 1),
        dtype=kernel_input.dtype,
    )
    with torch_device_fn.device(input.device):
        _complex_transpose_tiled_kernel[grid](
            kernel_input,
            kernel_out,
            input.shape[dim0],
            input.shape[dim1],
            (input.stride(dim0), input.stride(dim1)),
            (out.stride(dim0), out.stride(dim1)),
            tuple(input.shape[dim] for dim in other_dims),
            tuple(input.stride(dim) for dim in other_dims),
            tuple(out.stride(dim) for dim in other_dims),
            words,
            input.is_conj(),
            input.is_neg(),
            (1 << (word_bits - 1)) if _has_lazy_metadata(input) else 0,
            _BLOCK_DIM0,
        )


def _launch_complex_flat_fallback(input, out, dim0, dim1, words, word_bits):
    shape = (input.numel(),) if input.is_contiguous() else tuple(input.shape)
    strides = (1,) if input.is_contiguous() else input.stride()
    n_words = input.numel() * words
    kernel_input = StridedBuffer(
        input,
        shape=(*input.shape, words),
        strides=(*(stride * words for stride in input.stride()), 1),
        dtype={16: torch.uint16, 32: torch.uint32, 64: torch.uint64}[word_bits],
    )
    kernel_out = StridedBuffer(
        out,
        shape=(*out.shape, words),
        strides=(*(stride * words for stride in out.stride()), 1),
        dtype=kernel_input.dtype,
    )
    with torch_device_fn.device(input.device):
        _complex_copy_kernel[(triton.cdiv(n_words, _BLOCK_SIZE),)](
            kernel_input,
            kernel_out,
            n_words,
            shape,
            strides,
            words,
            input.is_conj(),
            input.is_neg(),
            (1 << (word_bits - 1)) if _has_lazy_metadata(input) else 0,
            _BLOCK_SIZE,
        )


def _narrowed_int64_transpose_copy(
    input: torch.Tensor,
    out_shape: list,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    """Handle int64/uint64 tensors whose storage torch_gcu narrowed to word layout.

    The word view is only addressable for contiguous zero-offset storage; any
    other layout (or Triton-unsuitable storage) goes through the native path,
    which is narrowing-aware.
    """
    if not _can_use_triton_storage(input) or not (
        input.is_contiguous() and input.storage_offset() == 0
    ):
        return _fallback_transpose_copy(input, dim0, dim1)
    kernel_input = _narrowed_i32_alias(input)
    if input.ndim <= 1 or dim0 == dim1:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        _launch_copy(kernel_input, _narrowed_i32_alias(out))
        return out
    if input.ndim <= _MAX_TILED_RANK:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        _launch_tiled_transpose_copy(kernel_input, _narrowed_i32_alias(out), dim0, dim1)
        return out
    return _fallback_transpose_copy(input, dim0, dim1)


def transpose_copy(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Return a contiguous copy with ``dim0`` and ``dim1`` swapped."""
    logger.debug("GEMS_ENFLAME TRANSPOSE_COPY")

    normalized_dim0 = _normalize_dim(dim0, input.ndim)
    normalized_dim1 = _normalize_dim(dim1, input.ndim)

    out_shape = list(input.shape)
    if input.ndim > 0:
        out_shape[normalized_dim0], out_shape[normalized_dim1] = (
            out_shape[normalized_dim1],
            out_shape[normalized_dim0],
        )
    if input.numel() == 0:
        return torch.empty(out_shape, dtype=input.dtype, device=input.device)

    if (
        input.is_complex()
        and input.device.type != "cpu"
        and input.layout == torch.strided
    ):
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        return _launch_complex_copy(input, out, normalized_dim0, normalized_dim1)

    if (
        input.dtype in (torch.int64, torch.uint64)
        and input.device.type != "cpu"
        and _int64_narrowed(input)
    ):
        return _narrowed_int64_transpose_copy(
            input, out_shape, normalized_dim0, normalized_dim1
        )

    use_byte_triton = _can_use_byte_triton(input)
    if not use_byte_triton and not _can_use_triton(input):
        return _fallback_transpose_copy(input, normalized_dim0, normalized_dim1)

    if input.ndim <= 1:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        kernel_input = _view_as_uint8(input) if use_byte_triton else input
        kernel_out = _view_as_uint8(out) if use_byte_triton else out
        _launch_copy(kernel_input, kernel_out)
        return out

    if normalized_dim0 == normalized_dim1:
        if input.is_contiguous():
            out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
            kernel_input = _view_as_uint8(input) if use_byte_triton else input
            kernel_out = _view_as_uint8(out) if use_byte_triton else out
            _launch_copy(kernel_input, kernel_out)
            return out
        return _fallback_transpose_copy(input, normalized_dim0, normalized_dim1)

    if input.ndim <= _MAX_TILED_RANK:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        kernel_input = _view_as_uint8(input) if use_byte_triton else input
        kernel_out = _view_as_uint8(out) if use_byte_triton else out
        _launch_tiled_transpose_copy(
            kernel_input, kernel_out, normalized_dim0, normalized_dim1
        )
        return out
    return _fallback_transpose_copy(input, normalized_dim0, normalized_dim1)
