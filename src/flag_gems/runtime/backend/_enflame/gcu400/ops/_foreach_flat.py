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

"""Internal helpers for the 1D flat kernels of the GCU400 ``foreach`` family.

The common ``foreach`` executor encodes the tensor list into a metadata table and launches a 2D
``grid=(block, tensor)``; GCU400 has a ``grid.y`` hardware limit of 255,
which causes ``OutOfResources`` on large shapes. Two alternative paths are provided here:

* a per-tensor 1D fixed grid + in-kernel striding kernel (plain migration baseline);
* packing the whole list's addresses / element counts into int64 metadata and processing all tensors
  in one launch with a 1D grid, with ``grid.x`` linearized over ``(tensor, part)`` to avoid
  the fixed overhead of 16 separate launches.

Only lists that are "float, contiguous, same dtype, standard broadcast shape" take the fast path; all other cases
are delegated by the caller to the common executor, keeping dtype policy / exception semantics completely unchanged.
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

# Triton 3.6 tl tensors have no __pow__; reuse the same GCU libdevice powf as the common implementation.
_pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)

FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# Compile-time dispatch code for binary operations, avoiding a copy of the kernel per operator.
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3
OP_MIN = 4
OP_MAX = 5
OP_POW = 6

# Unary operation code.
UNARY_EXP = 0
UNARY_CEIL = 1

# Shape code for the second operand.
MODE_LIST = 0
MODE_SCALAR = 1
MODE_SCALAR_LIST = 2
MODE_TENSOR = 3

# ``run_binary`` / ``run_unary`` use it to mean "this input is not suitable for the fast path".
FALLBACK = object()

# host dtype -> Triton dtype, used to turn raw addresses in metadata back into typed pointers.
_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@libentry()
@triton.jit(do_not_specialize=["N"])
def binary_flat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    OP: tl.constexpr,
    HAS_B_TENSOR: tl.constexpr,
    B_BROADCAST: tl.constexpr,
    B_SCALAR: tl.constexpr,
    ALPHA: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """1D strided computation for a single tensor; scalar / broadcast operands are dispatched at compile time."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask)
        if HAS_B_TENSOR:
            if B_BROADCAST:
                # 0-d / single-element operands are read once, avoiding repeated loads of a whole block.
                b = tl.load(b_ptr)
            else:
                b = tl.load(b_ptr + offs, mask=mask)
            if FP32_PROMOTE:
                b = b.to(tl.float32)
        elif FP32_PROMOTE:
            b = tl.full((), B_SCALAR, tl.float32)
        else:
            b = B_SCALAR
        if FP32_PROMOTE:
            # Same as the common executor: promote low precision to fp32, then implicitly cast back on store.
            a = a.to(tl.float32)
        if ALPHA != 1:
            b = b * ALPHA
        if OP == 0:
            result = a + b
        elif OP == 1:
            result = a - b
        elif OP == 2:
            result = a * b
        elif OP == 3:
            result = a / b
        elif OP == 4:
            result = tl.minimum(a, b)
        elif OP == 5:
            result = tl.maximum(a, b)
        else:
            # powf needs fp32 operands, consistent with the promotion in the common _pow_fn.
            result = _pow(a.to(tl.float32), b.to(tl.float32))
        tl.store(out_ptr + offs, result, mask=mask)


@libentry()
@triton.jit
def fused_binary_list_kernel(
    meta_ptr,
    NT: tl.constexpr,
    OP: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``.List``: process the whole paired tensor list in one launch.

    ``meta_ptr`` holds input pointers, output pointers, second-operand pointers and element counts
    (NT entries each, int64). grid.x is linearized over ``(tensor, part)``.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    b_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(A_DT))

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        if OP == 0:
            result = a + b
        elif OP == 1:
            result = a - b
        elif OP == 2:
            result = a * b
        elif OP == 3:
            result = a / b
        elif OP == 4:
            result = tl.minimum(a, b)
        elif OP == 5:
            result = tl.maximum(a, b)
        else:
            # powf needs fp32 operands, consistent with the promotion in the common _pow_fn.
            result = _pow(a.to(tl.float32), b.to(tl.float32))
        tl.store(out_ptr + offs, result, mask=mask)


@libentry()
@triton.jit
def fused_binary_scalar_kernel(
    meta_ptr,
    scalar_ptr,
    NT: tl.constexpr,
    OP: tl.constexpr,
    A_DT: tl.constexpr,
    HAS_B_TENSOR: tl.constexpr,
    B_DT: tl.constexpr,
    B_SCALAR: tl.constexpr,
    SCALAR_PER_TENSOR: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``.Scalar`` / ``.ScalarList`` / ``.Tensor``: process the whole list in one launch.

    ``HAS_B_TENSOR`` corresponds to 0-d tensor broadcast; ``SCALAR_PER_TENSOR`` corresponds to
    the per-tensor scalar array of ScalarList; other cases use the compile-time constant ``B_SCALAR``.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    b_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(B_DT))

    if SCALAR_PER_TENSOR:
        scalar = tl.load(scalar_ptr + tensor_id)
    else:
        scalar = tl.full((), B_SCALAR, A_DT)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        if HAS_B_TENSOR:
            b = tl.load(b_ptr)
        else:
            b = scalar
        if FP32_PROMOTE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        if OP == 0:
            result = a + b
        elif OP == 1:
            result = a - b
        elif OP == 2:
            result = a * b
        elif OP == 3:
            result = a / b
        elif OP == 4:
            result = tl.minimum(a, b)
        elif OP == 5:
            result = tl.maximum(a, b)
        else:
            # powf needs fp32 operands, consistent with the promotion in the common _pow_fn.
            result = _pow(a.to(tl.float32), b.to(tl.float32))
        tl.store(out_ptr + offs, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def unary_flat_kernel(
    data_ptr,
    N,
    OP: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place unary operation on a single tensor, written back with 1D striding."""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(data_ptr + offs, mask=mask)
        # The GCU backend's exp/ceil accept only fp32/fp64, so low precision must be promoted first;
        # store implicitly casts back to the output tensor's dtype.
        value = value.to(tl.float32)
        if OP == 0:
            result = tl.exp(value)
        else:
            result = tl.ceil(value)
        tl.store(data_ptr + offs, result, mask=mask)


@libentry()
@triton.jit
def fused_unary_kernel(
    meta_ptr,
    NT: tl.constexpr,
    OP: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place unary operation over the whole tensor list in one launch."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    data_raw = tl.load(meta_ptr + tensor_id)
    num_elems = tl.load(meta_ptr + NT + tensor_id)
    data_ptr = data_raw.to(tl.pointer_type(A_DT))

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        value = tl.load(data_ptr + offs, mask=mask)
        # The GCU backend's exp/ceil accept only fp32/fp64, so low precision must be promoted first;
        # store implicitly casts back to the output tensor's dtype.
        value = value.to(tl.float32)
        if OP == 0:
            result = tl.exp(value)
        else:
            result = tl.ceil(value)
        tl.store(data_ptr + offs, result, mask=mask)


def can_use_flat(self, operand, mode):
    """Check whether a list has a shape the fast path can handle safely."""
    if not self:
        return False
    if not all(isinstance(tensor, torch.Tensor) for tensor in self):
        # Non-Tensor elements: return False so the caller falls back to the common executor for matching exceptions.
        return False
    dtype = self[0].dtype
    if dtype not in FLOAT_DTYPES or dtype.is_complex:
        return False
    for tensor in self:
        if tensor.dtype != dtype or not tensor.is_contiguous() or tensor.numel() == 0:
            return False
    if mode is None:
        # Unary operations have no second operand, so checking the list itself is enough.
        return True
    if mode == MODE_LIST:
        if not isinstance(operand, (list, tuple)) or len(operand) != len(self):
            return False
        for index, other in enumerate(operand):
            if not isinstance(other, torch.Tensor) or other.dtype != dtype:
                return False
            if not other.is_contiguous():
                return False
            if other.numel() != self[index].numel() and other.numel() != 1:
                return False
    elif mode == MODE_SCALAR_LIST:
        if not isinstance(operand, (list, tuple)) or len(operand) != len(self):
            return False
        if not all(isinstance(value, (int, float)) for value in operand):
            return False
    elif mode == MODE_TENSOR:
        if not isinstance(operand, torch.Tensor) or operand.numel() != 1:
            return False
    elif not isinstance(operand, (int, float)):
        return False
    return True


def _operand_for(tensor, operand, mode, index):
    """Normalize the four operand shapes into (pointer arg, is_tensor, is_broadcast, scalar value)."""
    if mode == MODE_LIST:
        other = operand[index]
        return other, 1, int(other.numel() == 1), 0.0
    if mode == MODE_TENSOR:
        return operand, 1, 1, 0.0
    if mode == MODE_SCALAR_LIST:
        return tensor, 0, 0, float(operand[index])
    return tensor, 0, 0, float(operand)


def _fused_grid(self, fixed_grid, block_size):
    """Small lists need only NT programs; large shapes give each tensor multiple programs."""
    num_tensors = len(self)
    blocks_per_tensor = triton.cdiv(max(t.numel() for t in self), block_size)
    return max(1, min(fixed_grid, num_tensors * blocks_per_tensor))


def _fused_meta(self, outputs, inplace, b_pointers):
    """Build four int64 metadata segments: input/output/second-operand addresses + element counts."""
    out_tensors = self if inplace else outputs
    pointers = (
        [tensor.data_ptr() for tensor in self]
        + [tensor.data_ptr() for tensor in out_tensors]
        + b_pointers
        + [tensor.numel() for tensor in self]
    )
    return torch.tensor(pointers, dtype=torch.int64, device=self[0].device)


def _run_fused(
    self,
    operand,
    op_code,
    mode,
    outputs,
    inplace,
    fixed_grid,
    block_size,
    num_warps,
    num_stages,
    use_device_context,
    fp32_promote,
):
    """Select a fused kernel by operand shape and process the whole list in one launch."""
    num_tensors = len(self)
    dtype = self[0].dtype
    triton_dtype = _TL_DTYPES[dtype]
    grid_size = _fused_grid(self, fixed_grid, block_size)

    if mode == MODE_LIST:
        b_pointers = [other.data_ptr() for other in operand]
        scalar_tensor = self[0]  # Placeholder, never read in this kernel
        kernel = fused_binary_list_kernel
        extra = ()
    elif mode == MODE_TENSOR:
        b_pointers = [operand.data_ptr()] * num_tensors
        scalar_tensor = self[0]
        kernel = fused_binary_scalar_kernel
        # 0-d tensors are usually fp32 and must be read with their own dtype, not the list dtype.
        b_dtype = _TL_DTYPES.get(operand.dtype, triton_dtype)
        extra = (1, b_dtype, 0.0, 0)
    elif mode == MODE_SCALAR_LIST:
        b_pointers = [0] * num_tensors
        scalar_tensor = torch.tensor(
            operand, dtype=torch.float32, device=self[0].device
        )
        kernel = fused_binary_scalar_kernel
        extra = (0, triton_dtype, 0.0, 1)
    else:
        b_pointers = [0] * num_tensors
        scalar_tensor = self[0]
        kernel = fused_binary_scalar_kernel
        extra = (0, triton_dtype, float(operand), 0)

    meta = _fused_meta(self, outputs, inplace, b_pointers)
    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        if mode == MODE_LIST:
            kernel[(grid_size,)](
                meta,
                num_tensors,
                op_code,
                triton_dtype,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            kernel[(grid_size,)](
                meta,
                scalar_tensor,
                num_tensors,
                op_code,
                triton_dtype,
                *extra,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return None if inplace else outputs


def run_binary(
    self,
    operand,
    op_code,
    mode,
    inplace,
    alpha=1.0,
    *,
    fixed_grid=48,
    block_size=8192,
    num_warps=4,
    num_stages=1,
    use_device_context=1,
    fp32_promote=1,
    fused=1,
):
    """Execute binary foreach; return the ``FALLBACK`` sentinel when the fast path does not apply."""
    if alpha != 1 or not can_use_flat(self, operand, mode):
        return FALLBACK

    outputs = None if inplace else [torch.empty_like(tensor) for tensor in self]
    # For ``.List``, if the second operand needs broadcasting it cannot share the input's flat index.
    fused_ready = fused and (
        mode != MODE_LIST
        or all(other.numel() == tensor.numel() for tensor, other in zip(self, operand))
    )
    if fused_ready:
        return _run_fused(
            self,
            operand,
            op_code,
            mode,
            outputs,
            inplace,
            fixed_grid,
            block_size,
            num_warps,
            num_stages,
            use_device_context,
            fp32_promote,
        )

    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        for index, tensor in enumerate(self):
            num_elems = tensor.numel()
            out = tensor if inplace else outputs[index]
            b_arg, has_b_tensor, b_broadcast, b_scalar = _operand_for(
                tensor, operand, mode, index
            )
            grid = (min(triton.cdiv(num_elems, block_size), fixed_grid),)
            binary_flat_kernel[grid](
                tensor,
                b_arg,
                out,
                num_elems,
                op_code,
                has_b_tensor,
                b_broadcast,
                b_scalar,
                alpha,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return None if inplace else outputs


def _run_fused_unary(
    self,
    op_code,
    inplace,
    fixed_grid,
    block_size,
    num_warps,
    num_stages,
    use_device_context,
    fp32_promote,
):
    """Unary operation over the whole list in one launch."""
    num_tensors = len(self)
    out_tensors = self if inplace else [torch.empty_like(t) for t in self]
    if not inplace:
        for source, target in zip(self, out_tensors):
            target.copy_(source)
    pointers = [t.data_ptr() for t in out_tensors] + [t.numel() for t in self]
    meta = torch.tensor(pointers, dtype=torch.int64, device=self[0].device)
    grid_size = _fused_grid(self, fixed_grid, block_size)
    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        fused_unary_kernel[(grid_size,)](
            meta,
            num_tensors,
            op_code,
            _TL_DTYPES[self[0].dtype],
            fp32_promote,
            block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return None if inplace else out_tensors


def run_unary(
    self,
    op_code,
    inplace,
    *,
    fixed_grid=48,
    block_size=8192,
    num_warps=4,
    num_stages=1,
    use_device_context=1,
    fp32_promote=1,
    fused=1,
):
    """Execute unary foreach; return the ``FALLBACK`` sentinel when the fast path does not apply."""
    if not can_use_flat(self, None, None):
        return FALLBACK
    if fused:
        return _run_fused_unary(
            self,
            op_code,
            inplace,
            fixed_grid,
            block_size,
            num_warps,
            num_stages,
            use_device_context,
            fp32_promote,
        )

    outputs = None if inplace else [torch.empty_like(tensor) for tensor in self]
    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        for index, tensor in enumerate(self):
            num_elems = tensor.numel()
            out = tensor if inplace else outputs[index]
            if not inplace:
                # Non-in-place forms must copy the input first, then compute in place on the new tensor.
                out.copy_(tensor)
            grid = (min(triton.cdiv(num_elems, block_size), fixed_grid),)
            unary_flat_kernel[grid](
                out,
                num_elems,
                op_code,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return None if inplace else outputs
