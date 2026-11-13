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

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_LEAD_RANK = 5
_WORD_BLOCK = 8192
_STRIDED_BLOCK = 1024
_NUM_WARPS = 4
# GCU400 sweet spot for streaming word kernels: a fixed, modest program count
# with a grid-stride loop beats one tiny program per block by ~3x (97us vs
# 230us on (4096,4096) cfloat) and stays within the P19 grid.x limit.
_GRID_CAP = 192
_I32_SAFE_WORDS = 2**31 - _WORD_BLOCK - 1

# complex dtype -> int32 words per complex element
_INT32_WORDS = {
    torch.complex32: 1,
    torch.complex64: 2,
    torch.complex128: 4,
}
_SIGN = -2147483648  # 0x80000000

# (dtype, flip_real) -> (pos_mod, pos_rem, xor) for the flat word kernel.
# pos_mod == 0 selects every word. The lazy negative bit composes with the
# conjugate: neg(conj(raw)) keeps the imaginary lane and flips the real lane.
_FLAT_XOR = {
    (torch.complex32, False): (0, 0, _SIGN),
    (torch.complex32, True): (0, 0, 32768),  # low fp16 half = real part
    (torch.complex64, False): (1, 1, _SIGN),
    (torch.complex64, True): (1, 0, _SIGN),
    (torch.complex128, False): (3, 3, _SIGN),
    (torch.complex128, True): (3, 1, _SIGN),
}

# (dtype, flip_real) -> (word lane to flip, xor) for the strided kernel.
_STRIDED_XOR = {
    (torch.complex32, False): (0, _SIGN),
    (torch.complex32, True): (0, 32768),
    (torch.complex64, False): (1, _SIGN),
    (torch.complex64, True): (0, _SIGN),
    (torch.complex128, False): (3, _SIGN),
    (torch.complex128, True): (1, _SIGN),
}


@libentry()
@triton.jit(do_not_specialize=["n_words"])
def _resolve_conj_word_kernel(
    src_ptr,
    dst_ptr,
    n_words,
    POS_MOD: tl.constexpr,
    POS_REM: tl.constexpr,
    XORV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize a conjugated (optionally negative) complex tensor through a
    flat int32 word stream, one kernel launch.

    A contiguous complex tensor is an interleaved word stream where each
    element keeps its lanes in adjacent words; conjugation flips the sign bit
    of the imaginary words and a lazy negative bit flips the real ones. All
    flips are int32 XORs: fp64 arithmetic is rejected by the GCU compiler
    (P04) and int64 lanes are unreliable (P21), while int32 word-level bit
    operations are verified bit-exact, including the predicated
    ``where(pos, v ^ mask, v)`` form used here.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (n_words + BLOCK_SIZE - 1) // BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    for block_id in tl.range(pid, num_blocks, num_pids):
        j = block_id * BLOCK_SIZE + arange
        mask = j < n_words
        v = tl.load(src_ptr + j, mask=mask)
        r = tl.where((j & POS_MOD) == POS_REM, v ^ XORV, v)
        tl.store(dst_ptr + j, r, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _resolve_conj_strided_kernel(
    src_ptr,
    dst_ptr,
    d0,
    d1,
    d2,
    d3,
    d4,
    s0,
    s1,
    s2,
    s3,
    s4,
    n_elements,
    RANK: tl.constexpr,
    WORDS: tl.constexpr,
    XOR_LANE: tl.constexpr,
    XORV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize a conjugated (optionally negative) complex view of any
    layout into a fresh contiguous tensor, one kernel launch.

    ``src_ptr`` addresses the input's raw int32 word storage: the conjugate
    and negative bits live in the tensor metadata only, so the (real, imag)
    lanes of element ``i`` sit at the word slots starting at its flat position
    (strides are pre-scaled to word units on the host). The sign flip is an
    int32 XOR on the lane selected by the flip kind. Flat indices stay int64
    throughout: int64 index arithmetic is the env-free form proven by the
    view_as_real_copy family, and only the data lanes must remain int32.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    dims = (d0, d1, d2, d3, d4)
    strides = (s0, s1, s2, s3, s4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        i = block_id.to(tl.int64) * BLOCK_SIZE + arange.to(tl.int64)
        mask = i < n_elements
        if RANK == 1:
            base = i * strides[0]
        else:
            rem = i
            base = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
            for j in tl.static_range(RANK - 1, -1, -1):
                idx = rem % dims[j]
                rem = rem // dims[j]
                base += idx * strides[j]
        w0 = tl.load(src_ptr + base, mask=mask)
        if XOR_LANE == 0:
            w0 = w0 ^ XORV
        tl.store(dst_ptr + i * WORDS, w0, mask=mask)
        if WORDS >= 2:
            w1 = tl.load(src_ptr + base + 1, mask=mask)
            if XOR_LANE == 1:
                w1 = w1 ^ XORV
            tl.store(dst_ptr + i * WORDS + 1, w1, mask=mask)
        if WORDS == 4:
            w2 = tl.load(src_ptr + base + 2, mask=mask)
            tl.store(dst_ptr + i * WORDS + 2, w2, mask=mask)
            w3 = tl.load(src_ptr + base + 3, mask=mask)
            if XOR_LANE == 3:
                w3 = w3 ^ XORV
            tl.store(dst_ptr + i * WORDS + 3, w3, mask=mask)


def _raw_int32_view(t: torch.Tensor, words: int) -> torch.Tensor:
    """Bind a fresh int32 window over ``t``'s raw storage, zero dispatch.

    ``torch.view_as_real`` refuses lazily conjugated tensors on torch_gcu
    (the "unresolved conjugated" P21 defect), so the kernels address the raw
    words through this pure-metadata ``set_`` view instead: it exposes the
    same data pointer as the complex tensor without any aten guard. The
    window shape is only a bound-valid carrier; the kernels derive their
    offsets from the arguments passed alongside.
    """
    storage = t.untyped_storage()
    total_words = storage.nbytes() // 4
    offset_words = t.storage_offset() * words
    view = torch.empty(1, dtype=torch.int32, device=t.device)
    view.set_(storage, offset_words, (total_words - offset_words,), (1,))
    return view


def resolve_conj(A: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME RESOLVE_CONJ")
    if not A.is_conj():
        # ATen semantics: resolve_conj on a tensor without the bit returns
        # the input itself, no copy.
        return A

    if not A.is_complex() or A.device.type != flag_gems.device:
        # The conjugate bit cannot alter non-complex values, and foreign
        # devices (CPU) resolve the bit inside clone(); either way a plain
        # clone is the correct materialization.
        return A.clone()

    words = _INT32_WORDS[A.dtype]
    n = A.numel()
    out = torch.empty_like(A, memory_format=torch.contiguous_format)
    if n == 0:
        return out

    # The lazy negative bit composes with the conjugate: neg(conj(raw)) keeps
    # the imaginary lane untouched and flips the real lane instead.
    flip_real = A.is_neg()
    n_words = n * words

    if A.is_contiguous() and n_words < _I32_SAFE_WORDS:
        # Contiguous: walk the interleaved word stream directly, one XOR pass.
        pos_mod, pos_rem, xorv = _FLAT_XOR[(A.dtype, flip_real)]
        src = _raw_int32_view(A, words)
        dst = _raw_int32_view(out, words)
        grid = (min(triton.cdiv(n_words, _WORD_BLOCK), _GRID_CAP),)
        with torch_device_fn.device(A.device):
            _resolve_conj_word_kernel[grid](
                src,
                dst,
                n_words,
                POS_MOD=pos_mod,
                POS_REM=pos_rem,
                XORV=xorv,
                BLOCK_SIZE=_WORD_BLOCK,
                num_warps=_NUM_WARPS,
            )
        return out

    # Non-contiguous (or beyond the int32 word domain): decompose each
    # element's flat index over the input's dims/strides. Exotic high-rank
    # views are densified first; contiguous() keeps the lazy bit on GCU and
    # copies the raw stream (P21), so the flat pass below stays correct.
    if A.dim() > _MAX_LEAD_RANK:
        dense = A.contiguous()
        src = _raw_int32_view(dense, words)
        rank, dims, strides = 1, (n, 1, 1, 1, 1), (words, 0, 0, 0, 0)
    else:
        src = _raw_int32_view(A, words)
        rank = A.dim()
        dims = tuple(A.shape) + (1,) * (_MAX_LEAD_RANK - rank)
        strides = tuple(s * words for s in A.stride()) + (0,) * (_MAX_LEAD_RANK - rank)

    xor_lane, xorv = _STRIDED_XOR[(A.dtype, flip_real)]
    dst = _raw_int32_view(out, words)
    grid = (min(triton.cdiv(n, _STRIDED_BLOCK), _GRID_CAP),)
    with torch_device_fn.device(A.device):
        _resolve_conj_strided_kernel[grid](
            src,
            dst,
            *dims,
            *strides,
            n,
            RANK=rank,
            WORDS=words,
            XOR_LANE=xor_lane,
            XORV=xorv,
            BLOCK_SIZE=_STRIDED_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return out
