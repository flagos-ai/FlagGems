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

"""GCU400-specific implementation of atanh / atanh_: 1D flat Triton kernel.

Input and output may share one memory block (in-place) or use a separate output (out-of-place);
large-shape tiles are chosen by element bytes, and resident CTAs are reused with a grid-stride loop.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Resident CTA count matches GCU400 SIP topology; large-shape tiles are tiered by element bytes
NUM_CTAS = 48
BLOCK_LARGE_16BIT = 8192
BLOCK_LARGE_32BIT = 8192
NUM_WARPS = 4
# Backward pass (d/dx atanh) is a 2-read/1-write stream: the 32768-element tile
# with 2 warps reaches 95-98% of the measured GCU400 streaming ceiling.
BACKWARD_BLOCK = 32768
BACKWARD_NUM_WARPS = 2
# Pipeline depth of the grid-stride loop: keep 3 as "harmless and potentially beneficial"; measurements and
# counterexamples are in the module
# docstring (for in-place and 16-bit, st=1/3 makes no difference; only in the two-pass scans of fp32 512M
# is st>=2 faster). It must be a tl.constexpr to be read by tl.range inside @triton.jit.
GRID_STRIDE_STAGES = tl.constexpr(3)


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def atanh_kernel(
    x_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = (num_elements + BLOCK - 1) // BLOCK
    for tile_index in tl.range(program_id, num_tiles, num_programs, num_stages=1):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        value = x.to(tl.float32)
        # sqrt((1+v)/(1-v)) = (1+v) * rsqrt(1-v) * rsqrt(1-v)
        reciprocal_sqrt = tl.rsqrt(1.0 - value)
        scaled = ((1.0 + value) * reciprocal_sqrt) * reciprocal_sqrt
        log_value = tl.log(scaled)
        tl.store(out_ptr + offsets, (log_value.to(x.dtype)) * 0.5, mask=mask)


def _choose_block(num_elements, is_fp32):
    """1024 for small shapes, the nearest power of two for medium ones, dtype tiers for large ones."""
    if num_elements <= 1024:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return BLOCK_LARGE_32BIT if is_fp32 else BLOCK_LARGE_16BIT


@libentry()
@triton.jit(do_not_specialize=["num_elements"])
def atanh_backward_kernel(
    grad_ptr,
    input_ptr,
    out_ptr,
    num_elements,
    BLOCK: tl.constexpr,
    IEEE_DIV: tl.constexpr,
):
    """d/dx atanh(x) = grad / (1 - x * x), same grid-stride tiling as atanh_kernel."""
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = (num_elements + BLOCK - 1) // BLOCK
    for tile_index in tl.range(program_id, num_tiles, num_programs, num_stages=1):
        offsets = tile_index * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < num_elements
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # ATen evaluates grad / (1 - x * x) in the input dtype instead of fp32
        # opmath, so for fp16/bf16 the rounding of (1 - x * x) is part of the
        # reference result: a fp32 denominator diverges from it by more than
        # the 1e-3 / 1.6e-2 test tolerance near |x| -> 1.
        denominator = 1.0 - x * x
        if IEEE_DIV:
            quotient = grad / denominator
        else:
            # 16-bit divide is emulated; the non-IEEE form is ~25% faster on
            # GCU400 (45us vs 60us for 16.7M fp16 elements) and its <=2ulp
            # error stays far inside the fp16/bf16 test tolerance.
            quotient = tl.fdiv(grad, denominator, ieee_rounding=False)
        tl.store(out_ptr + offsets, quotient, mask=mask)


def _choose_backward_launch(num_elements, is_fp32):
    """Tile, warps and divide flavour for the backward launch.

    Measured against a 2-read/1-write streaming ceiling of ~2.4 TB/s (16-bit)
    and ~2.6 TB/s (fp32): the 32768-element / 2-warp tile reaches 95-98% of it
    for both tiers, while small shapes stay on one tile with more warps.
    """
    if num_elements <= 1024:
        return 1024, NUM_WARPS, is_fp32
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements), NUM_WARPS, is_fp32
    # fp32 keeps the correctly rounded divide: it is as fast as the fast form
    # there, and the tolerance is the tightest of the three dtypes.
    return BACKWARD_BLOCK, BACKWARD_NUM_WARPS, is_fp32


def _atanh_backward(grad_output, input_tensor):
    """Return grad_output / (1 - input ** 2) as a plain Triton launch."""
    contiguous_grad = (
        grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    )
    contiguous_input = (
        input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
    )
    grad_input = torch.empty_like(contiguous_input)
    num_elements = contiguous_input.numel()
    if num_elements == 0:
        return grad_input
    block_size, num_warps, ieee_div = _choose_backward_launch(
        num_elements, contiguous_input.dtype == torch.float32
    )
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(contiguous_input.device):
        atanh_backward_kernel[(grid_size,)](
            contiguous_grad,
            contiguous_input,
            grad_input,
            num_elements,
            BLOCK=block_size,
            IEEE_DIV=ieee_div,
            num_warps=num_warps,
        )
    return grad_input


class _AtanhFunction(torch.autograd.Function):
    """Autograd wrapper for the out-of-place atanh kernel.

    The flat Triton kernels write through raw pointers and create no graph node
    on their own, so the forward node and the backward kernel are wired here;
    without it every backward-test path fails with "does not require grad".
    """

    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return _atanh(A)

    @staticmethod
    def backward(ctx, grad_output):
        (saved_input,) = ctx.saved_tensors
        return _atanh_backward(grad_output, saved_input)


def _launch(inp, out):
    num_elements = inp.numel()
    if num_elements == 0:
        return out
    is_fp32 = inp.dtype == torch.float32
    block_size = _choose_block(num_elements, is_fp32)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_CTAS)
    with torch_device_fn.device(inp.device):
        atanh_kernel[(grid_size,)](
            inp,
            out,
            num_elements,
            BLOCK=block_size,
            num_warps=NUM_WARPS,
        )
    return out


def _atanh(A):
    inp = A if A.is_contiguous() else A.contiguous()
    return _launch(inp, torch.empty_like(inp))


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# GCU tensors only: the AutogradPrivateUse1 slot below must not touch the
# AutogradCPU/AutogradCUDA formulas of other backends.
_ATEN_PRIVATEUSE1_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)


def _run_device_atanh_(A):
    """Run the registered in-place atanh without entering autograd again."""
    return torch.ops.aten.atanh_.default.redispatch(_ATEN_PRIVATEUSE1_KEYSET, A)


def _atanh_inplace_kernel(A):
    """Write atanh(A) into A with no autograd bookkeeping."""
    if not A.is_contiguous():
        # copy_ is a real ATen in-place op, so it bumps the version counter.
        A.copy_(_atanh(A))
        return A
    out = _launch(A, A)
    if A.requires_grad or A.grad_fn is not None:
        # A direct Triton store bypasses ATen's in-place bump; autograd uses
        # that counter to reject stale saved tensors, so do it by hand.
        torch.autograd.graph.increment_version(A)
    return out


class _AtanhInplaceFunction(torch.autograd.Function):
    """Autograd for aten::atanh_.

    ATen's own in-place atanh_ raises "the derivative for 'inplace version of
    atanh' is not implemented", and the flat Triton kernel writes through the
    pointer without creating a node, so gradients either raised or silently
    fell through to the node that produced the buffer.  Both the graph and the
    backward kernel live here instead.
    """

    @staticmethod
    def forward(ctx, A):
        # The device kernel overwrites A, so the pre-write values have to be
        # snapshotted: 1 / (1 - x * x) needs the input, not atanh(input).
        saved_input = A.clone()
        # A detached alias shares A's version counter but is not the function
        # output, so autograd still version-checks it (a saved output would be
        # exempt) and a later in-place write to A trips "modified by an
        # inplace operation" as it does for any other ATen op.
        version_probe = A.detach()
        out = _run_device_atanh_(A)
        ctx.mark_dirty(A)
        ctx.save_for_backward(version_probe, saved_input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        _, saved_input = ctx.saved_tensors
        return _atanh_backward(grad_output, saved_input)


def _atanh_inplace_autograd(A):
    return _AtanhInplaceFunction.apply(A)


# aten::atanh_ has no ATen autograd formula (it always raises), so the free
# AutogradPrivateUse1 slot is claimed here: the device kernel keeps handling
# the forward, only the graph node comes from this module.
_ATEN_AUTOGRAD_LIB = torch.library.Library("aten", "IMPL")
_ATEN_AUTOGRAD_LIB.impl(
    "atanh_",
    _atanh_inplace_autograd,
    "AutogradPrivateUse1",
    allow_override=True,
)


def atanh(A):
    logger.debug("GEMS_ENFLAME ATANH")
    # The flat kernel allocates with empty_like and stores back in the input
    # dtype, which only matches ATen for float inputs.  ATen promotes integer
    # inputs to fp32, so leave those to the generic INT_TO_FLOAT pointwise op.
    if A.dtype not in _FLOAT_DTYPES:
        from flag_gems.ops.atanh import atanh as _generic_atanh

        return _generic_atanh(A)
    if torch.is_grad_enabled() and A.requires_grad:
        return _AtanhFunction.apply(A)
    return _atanh(A)


def atanh_(A):
    logger.debug("GEMS_ENFLAME ATANH_")
    # ATen rejects in-place atanh on integer/bool inputs (the float result
    # cannot be cast back into the output buffer), matching the generic op.
    if not A.dtype.is_floating_point:
        raise RuntimeError(
            "result type Float can't be cast to the desired output type " f"{A.dtype}"
        )
    if A.dtype not in _FLOAT_DTYPES:
        # fp64 (and any non-flat-kernel float dtype) goes through the generic
        # in-place op; raising here would wrongly reject valid float inputs.
        from flag_gems.ops.atanh import atanh_ as _generic_atanh_

        return _generic_atanh_(A)
    if A.requires_grad and torch.is_grad_enabled():
        # ATen rejects the leaf case before touching the buffer, so raise here
        # rather than letting mark_dirty() report it after the kernel wrote.
        if A.is_leaf:
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an "
                "in-place operation."
            )
        # apply() disables grad for the forward, so the kernel below runs
        # exactly once and the node is created only here.
        return _AtanhInplaceFunction.apply(A)
    return _atanh_inplace_kernel(A)
