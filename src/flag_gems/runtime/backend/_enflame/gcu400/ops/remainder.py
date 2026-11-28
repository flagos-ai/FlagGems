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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# Three modulo implementations, selected by the VARIANT constexpr:
#   _NATIVE    Use the dtype's own `%` directly (int32 / floating-point takes this path)
#   _WIDEN     Widen to int32 before `%`
#   _FLOAT_DIV Compute the quotient with fp32 division then the remainder (for 16-bit integers only)
# Must be a tl.constexpr, otherwise module-level global constants cannot be referenced inside the kernel.
_NATIVE = tl.constexpr(0)
_WIDEN = tl.constexpr(1)
_FLOAT_DIV = tl.constexpr(2)

# int32 offset safe range, kept consistent with abs.py: the tail block can reach num_blocks * BLOCK - 1,
# must ensure it does not wrap around to a negative value.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)


@triton.jit
def _remainder_math(x, y, VARIANT: tl.constexpr):
    """Python-semantics remainder (the result sign follows the divisor).

    _FLOAT_DIV premise: |x| and |y| are exactly representable in fp32 (true for int8/int16), so
    the fp32 quotient differs from the exact value by < 1, the truncated quotient is off by at
    most 1, and the residual lands within one |y|; one correction normalizes it to [0, |y|) and
    the divisor sign is applied. Integer division is emulated on GCU400, so this is ~40% faster.
    """
    # Single return point: Triton unifies the dtype of all returns; multiple return points would cause the unselected
    # branch's
    # 16-bit result to conflict with the int32 result (Inconsistent return types).
    if VARIANT == _FLOAT_DIV:
        xw = x.to(tl.int32)
        yw = xw * 0 + y  # Uniformly promote scalar/16-bit operands to int32
        abs_y = tl.abs(yw)
        quotient = (xw.to(tl.float32) / yw.to(tl.float32)).to(tl.int32)
        res = xw - quotient * yw
        res = tl.where(res < 0, res + abs_y, res)
        res = tl.where(res >= abs_y, res - abs_y, res)
        res = tl.where((res != 0) & (yw < 0), res + yw, res)
    elif VARIANT == _WIDEN:
        xw = x.to(tl.int32)
        yw = xw * 0 + y
        res = xw % yw
        res = tl.where((res != 0) & ((xw < 0) ^ (yw < 0)), res + yw, res)
    else:
        res = x % y
        res = tl.where((res != 0) & ((x < 0) ^ (y < 0)), res + y, res)
    return res


@libentry()
@triton.jit(do_not_specialize=["N"])
def remainder_tt_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    VARIANT: tl.constexpr,
):
    """tensor % tensor: BLOCK tiling plus a grid-stride walk.

    EVEN means full tiles, taking the predicate-free memory path (masked accesses are clearly slower on GCU400).
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
            tl.store(out_ptr + off, _remainder_math(x, y, VARIANT))
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask)
            y = tl.load(y_ptr + off, mask=mask)
            tl.store(out_ptr + off, _remainder_math(x, y, VARIANT), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def remainder_ts_kernel(
    x_ptr,
    y_scalar,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    VARIANT: tl.constexpr,
):
    """tensor % scalar."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            tl.store(out_ptr + off, _remainder_math(x, y_scalar, VARIANT))
        else:
            mask = off < N
            x = tl.load(x_ptr + off, mask=mask)
            tl.store(out_ptr + off, _remainder_math(x, y_scalar, VARIANT), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def remainder_st_kernel(
    x_scalar,
    y_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    VARIANT: tl.constexpr,
):
    """scalar % tensor."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            y = tl.load(y_ptr + off)
            tl.store(out_ptr + off, _remainder_math(x_scalar, y, VARIANT))
        else:
            mask = off < N
            y = tl.load(y_ptr + off, mask=mask)
            tl.store(out_ptr + off, _remainder_math(x_scalar, y, VARIANT), mask=mask)


def _variant_for(dtype):
    """Picks the arithmetic implementation by dtype.

    _FLOAT_DIV needs fp32 to represent the operands exactly, which only holds for itemsize <= 2;
    int32 and wider use native `%` (measured no gain on GCU400 from promoting to int64).
    """
    if dtype.is_floating_point:
        return _NATIVE
    if dtype.itemsize <= 2:
        return _FLOAT_DIV
    return _NATIVE


def _choose_block(n_elements, elem_size):
    """Picks the tile from element count and element width.

    Same-source do_bench A/B on GCU400: 128KB-class tiles are best; int16 at 16.7M is 3% faster
    with a 64KB tile (0.0896ms vs 0.0927ms) and only needs 128KB at 512M.
    Small shapes degrade to a single next_pow2 block to save scheduling.
    """
    if n_elements <= 32768:
        return max(triton.next_power_of_2(n_elements), 64)
    if elem_size >= 4:
        return 32768
    return 32768 if n_elements <= (1 << 25) else 65536


def _choose_warps(mode):
    # tensor % tensor prefers fewer warps (more registers/tile), and the opposite for scalar mode.
    return 4 if mode in ("ts", "st") else 2


def _launch_tt(inp_a, inp_b, out, n_elements, dtype):
    variant = _variant_for(dtype)
    block = _choose_block(n_elements, dtype.itemsize)
    grid_size = min(triton.cdiv(n_elements, block), NUM_SIPS * 2)
    # The external llvm converter occasionally fails on cold compilation (same precedent in neg.py / abs.py); retrying
    # succeeds.
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp_a.device):
                remainder_tt_kernel[(grid_size,)](
                    inp_a,
                    inp_b,
                    out,
                    n_elements,
                    BLOCK=block,
                    EVEN=n_elements % block == 0,
                    VARIANT=variant,
                    num_warps=_choose_warps("tt"),
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "remainder tt kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def _launch_ts(inp, y_scalar, out, n_elements, dtype):
    variant = _variant_for(dtype)
    block = _choose_block(n_elements, dtype.itemsize)
    grid_size = min(triton.cdiv(n_elements, block), NUM_SIPS * 2)
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                remainder_ts_kernel[(grid_size,)](
                    inp,
                    y_scalar,
                    out,
                    n_elements,
                    BLOCK=block,
                    EVEN=n_elements % block == 0,
                    VARIANT=variant,
                    num_warps=_choose_warps("ts"),
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "remainder ts kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def _launch_st(x_scalar, inp, out, n_elements, dtype):
    variant = _variant_for(dtype)
    block = _choose_block(n_elements, dtype.itemsize)
    grid_size = min(triton.cdiv(n_elements, block), NUM_SIPS * 2)
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                remainder_st_kernel[(grid_size,)](
                    x_scalar,
                    inp,
                    out,
                    n_elements,
                    BLOCK=block,
                    EVEN=n_elements % block == 0,
                    VARIANT=variant,
                    num_warps=_choose_warps("st"),
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "remainder st kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


def _flat_path_ok(tensor):
    return tensor.is_contiguous() and tensor.numel() < _INT32_SAFE_LIMIT


def _public_remainder(A, B):
    from flag_gems.ops.remainder import remainder as _public

    return _public(A, B)


def remainder(A, B):
    logger.debug("GEMS_ENFLAME REMAINDER")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        inp_a = A if A.is_contiguous() else A.contiguous()
        inp_b = B if B.is_contiguous() else B.contiguous()
        if inp_a.shape != inp_b.shape:
            shape = torch.broadcast_shapes(inp_a.shape, inp_b.shape)
            inp_a = inp_a.expand(shape).contiguous()
            inp_b = inp_b.expand(shape).contiguous()
        n_elements = inp_a.numel()
        if n_elements == 0:
            return torch.empty_like(inp_a)
        if not _flat_path_ok(inp_a):
            return _public_remainder(A, B)
        out = torch.empty_like(inp_a)
        _launch_tt(inp_a, inp_b, out, n_elements, inp_a.dtype)
        return out
    elif isinstance(A, torch.Tensor):
        inp = A if A.is_contiguous() else A.contiguous()
        n_elements = inp.numel()
        if n_elements == 0:
            return torch.empty_like(inp)
        if not _flat_path_ok(inp):
            return _public_remainder(A, B)
        out = torch.empty_like(inp)
        _launch_ts(inp, B, out, n_elements, inp.dtype)
        return out
    elif isinstance(B, torch.Tensor):
        inp = B if B.is_contiguous() else B.contiguous()
        n_elements = inp.numel()
        if n_elements == 0:
            return torch.empty_like(inp)
        if not _flat_path_ok(inp):
            return _public_remainder(A, B)
        out = torch.empty_like(inp)
        _launch_st(A, inp, out, n_elements, inp.dtype)
        return out
    else:
        return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_ENFLAME REMAINDER_")
    inp_a = A if A.is_contiguous() else A.contiguous()
    n_elements = inp_a.numel()
    if n_elements == 0:
        return A
    if not _flat_path_ok(inp_a):
        from flag_gems.ops.remainder import remainder_ as _public

        return _public(A, B)
    if isinstance(B, torch.Tensor):
        inp_b = B if B.is_contiguous() else B.contiguous()
        if inp_a.shape != inp_b.shape:
            shape = torch.broadcast_shapes(inp_a.shape, inp_b.shape)
            inp_b = inp_b.expand(shape).contiguous()
        _launch_tt(inp_a, inp_b, A, n_elements, inp_a.dtype)
    else:
        _launch_ts(inp_a, B, A, n_elements, inp_a.dtype)
    return A
