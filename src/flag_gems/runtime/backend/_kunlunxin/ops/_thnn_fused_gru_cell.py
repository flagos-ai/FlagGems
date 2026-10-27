# Copyright 2026, The FlagOS Contributors.
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
# Kunlunxin specialization of ``_thnn_fused_gru_cell``.
#
# ``HIDDEN_SIZE`` is a compile-time constant, and the launcher picks the grid
# shape from the measured per-shape optimum: a flat 1-D grid, or a per-row 2-D
# grid when a flat block would have to cross row boundaries.  The kernel body
# (gate algebra, bias handling, sigmoid/tanh, workspace stores) is kept verbatim
# from the generic implementation, including the order of the additions, so the
# numerics are unchanged.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_BLOCK_SIZE = 1024

# --- Flat 1-D path -----------------------------------------------------------
#
# The cost of a kernel here is dominated by the number of blocks times the number
# of memory operations, so the flat path covers as many elements per block as the
# device still parallelises; ``_flat_plan`` picks the block from the measured
# per-shape optimum.  Two equivalent kernel bodies exist because they do not
# compile over the same block range: the eager form reaches BLOCK=8192 with
# 4-byte operands, and the deferred form -- each value loaded where it is first
# used, each result stored as soon as it is computed -- is the one that compiles
# from BLOCK=2048 up with 2-byte operands.  Only combinations measured to compile
# are selected.
_FLAT_MIN_BLOCK = 512
# Widest block each form still compiles for, per operand width.
_FLAT_MAX_BLOCK = {2: 1024, 4: 8192}
# Block used by the deferred form for 2-byte operands.
_FLAT_LAZY_BLOCK = 2048


def _flat_plan(n_elements, element_bytes):
    """Choose (kernel form, block) from combinations measured to compile.

    The eager form stops growing once four CTAs would no longer fit: a block
    that spans the whole tensor leaves one CTA, and the device then runs the
    kernel ~4x slower than it runs four CTAs over the same elements (measured
    fp32 at (32, 128): 26.1 us at one CTA against 7.1 us at four).
    """
    if element_bytes == 2 and n_elements % _FLAT_LAZY_BLOCK == 0:
        return "lazy", _FLAT_LAZY_BLOCK
    limit = _FLAT_MAX_BLOCK.get(element_bytes, _FLAT_MIN_BLOCK)
    block = _FLAT_MIN_BLOCK
    while block < limit and block * 4 < n_elements:
        block *= 2
    return "eager", block


def _pick_block_size(hidden_size):
    """Pick BLOCK_SIZE from the measured per-shape optimum.

    The cost model is ``grid * n_ops * cost(BLOCK)`` with
    ``grid = batch * ceil(H / BLOCK)``.  A wide ``tl.arange`` is required to get
    a usable bandwidth out of the XPU backend (a 1024-wide block is ~2x cheaper
    per op than a 512-wide one), but when ``BLOCK > H`` the surplus lanes are
    wasted.  Measured fp32 totals at the benchmark shapes:
        H   BLOCK=512   BLOCK=1024
        128   19.47       19.15
        256   47.02       45.49
        512   33.60       54.91
       1024  121.80       58.95
    so BLOCK=1024 wins except in the 512 <= H < 1024 window where BLOCK=512
    covers a row exactly with no wasted lanes.
    """
    if hidden_size % _BLOCK_SIZE == 0:
        return _BLOCK_SIZE
    if hidden_size >= 512:
        return 512
    return _BLOCK_SIZE


@triton.jit
def _thnn_fused_gru_cell_gates(
    input_reset,
    input_gate,
    input_new,
    hidden_reset,
    hidden_gate,
    hidden_new,
    hx_value,
    bias_reset,
    bias_update,
    bias_new,
    hidden_bias_reset,
    hidden_bias_update,
    hidden_bias_new,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    if not IS_FP64:
        input_reset = input_reset.to(tl.float32)
        input_gate = input_gate.to(tl.float32)
        input_new = input_new.to(tl.float32)
        hidden_reset = hidden_reset.to(tl.float32)
        hidden_gate = hidden_gate.to(tl.float32)
        hidden_new = hidden_new.to(tl.float32)
        hx_value = hx_value.to(tl.float32)
        if HAS_BIAS:
            bias_reset = bias_reset.to(tl.float32)
            bias_update = bias_update.to(tl.float32)
            bias_new = bias_new.to(tl.float32)
            hidden_bias_reset = hidden_bias_reset.to(tl.float32)
            hidden_bias_update = hidden_bias_update.to(tl.float32)
            hidden_bias_new = hidden_bias_new.to(tl.float32)

    reset_gate = input_reset + hidden_reset
    update_gate = input_gate + hidden_gate
    if HAS_BIAS:
        reset_gate += bias_reset
        reset_gate += hidden_bias_reset
        update_gate += bias_update
        update_gate += hidden_bias_update
        input_new += bias_new
        hidden_new += hidden_bias_new

    reset_gate = tl.sigmoid(reset_gate)
    update_gate = tl.sigmoid(update_gate)
    new_gate = input_new + reset_gate * hidden_new
    new_gate = tl_extra_shim.tanh(new_gate)
    hy_value = new_gate + update_gate * (hx_value - new_gate)
    return reset_gate, update_gate, new_gate, hx_value, hidden_new, hy_value


@triton.jit
def _thnn_fused_gru_cell_kernel(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    workspace,
    HIDDEN_SIZE: tl.constexpr,
    input_stride_0,
    input_stride_1,
    hidden_stride_0,
    hidden_stride_1,
    hx_stride_0,
    hx_stride_1,
    input_bias_stride,
    hidden_bias_stride,
    hy_stride_0,
    hy_stride_1,
    workspace_stride_0,
    workspace_stride_1,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Masked variant: one block covers BLOCK_SIZE columns of a single row."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE

    input_base = row * input_stride_0 + cols * input_stride_1
    hidden_base = row * hidden_stride_0 + cols * hidden_stride_1
    input_reset = tl.load(input_gates + input_base, mask=mask, other=0.0)
    input_gate = tl.load(
        input_gates + input_base + HIDDEN_SIZE * input_stride_1, mask=mask, other=0.0
    )
    input_new = tl.load(
        input_gates + input_base + 2 * HIDDEN_SIZE * input_stride_1,
        mask=mask,
        other=0.0,
    )
    hidden_reset = tl.load(hidden_gates + hidden_base, mask=mask, other=0.0)
    hidden_gate = tl.load(
        hidden_gates + hidden_base + HIDDEN_SIZE * hidden_stride_1,
        mask=mask,
        other=0.0,
    )
    hidden_new = tl.load(
        hidden_gates + hidden_base + 2 * HIDDEN_SIZE * hidden_stride_1,
        mask=mask,
        other=0.0,
    )
    hx_value = tl.load(
        hx + row * hx_stride_0 + cols * hx_stride_1, mask=mask, other=0.0
    )

    if HAS_BIAS:
        bias_reset = tl.load(
            input_bias + cols * input_bias_stride, mask=mask, other=0.0
        )
        bias_update = tl.load(
            input_bias + HIDDEN_SIZE * input_bias_stride + cols * input_bias_stride,
            mask=mask,
            other=0.0,
        )
        bias_new = tl.load(
            input_bias + 2 * HIDDEN_SIZE * input_bias_stride + cols * input_bias_stride,
            mask=mask,
            other=0.0,
        )
        hidden_bias_reset = tl.load(
            hidden_bias + cols * hidden_bias_stride, mask=mask, other=0.0
        )
        hidden_bias_update = tl.load(
            hidden_bias + HIDDEN_SIZE * hidden_bias_stride + cols * hidden_bias_stride,
            mask=mask,
            other=0.0,
        )
        hidden_bias_new = tl.load(
            hidden_bias
            + 2 * HIDDEN_SIZE * hidden_bias_stride
            + cols * hidden_bias_stride,
            mask=mask,
            other=0.0,
        )
    else:
        zero = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        bias_reset = zero
        bias_update = zero
        bias_new = zero
        hidden_bias_reset = zero
        hidden_bias_update = zero
        hidden_bias_new = zero

    reset_gate, update_gate, new_gate, hx_value, hidden_new, hy_value = (
        _thnn_fused_gru_cell_gates(
            input_reset,
            input_gate,
            input_new,
            hidden_reset,
            hidden_gate,
            hidden_new,
            hx_value,
            bias_reset,
            bias_update,
            bias_new,
            hidden_bias_reset,
            hidden_bias_update,
            hidden_bias_new,
            HAS_BIAS,
            IS_FP64,
        )
    )

    tl.store(hy + row * hy_stride_0 + cols * hy_stride_1, hy_value, mask=mask)
    workspace_base = row * workspace_stride_0 + cols * workspace_stride_1
    tl.store(workspace + workspace_base, reset_gate, mask=mask)
    tl.store(
        workspace + workspace_base + HIDDEN_SIZE * workspace_stride_1,
        update_gate,
        mask=mask,
    )
    tl.store(
        workspace + workspace_base + 2 * HIDDEN_SIZE * workspace_stride_1,
        new_gate,
        mask=mask,
    )
    tl.store(
        workspace + workspace_base + 3 * HIDDEN_SIZE * workspace_stride_1,
        hx_value,
        mask=mask,
    )
    tl.store(
        workspace + workspace_base + 4 * HIDDEN_SIZE * workspace_stride_1,
        hidden_new,
        mask=mask,
    )


@triton.jit
def _thnn_fused_gru_cell_kernel_nomask(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    workspace,
    HIDDEN_SIZE: tl.constexpr,
    input_stride_0,
    input_stride_1,
    hidden_stride_0,
    hidden_stride_1,
    hx_stride_0,
    hx_stride_1,
    input_bias_stride,
    hidden_bias_stride,
    hy_stride_0,
    hy_stride_1,
    workspace_stride_0,
    workspace_stride_1,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Maskless variant; requires ``HIDDEN_SIZE % BLOCK_SIZE == 0``."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_base = row * input_stride_0 + cols * input_stride_1
    hidden_base = row * hidden_stride_0 + cols * hidden_stride_1
    input_reset = tl.load(input_gates + input_base)
    input_gate = tl.load(input_gates + input_base + HIDDEN_SIZE * input_stride_1)
    input_new = tl.load(input_gates + input_base + 2 * HIDDEN_SIZE * input_stride_1)
    hidden_reset = tl.load(hidden_gates + hidden_base)
    hidden_gate = tl.load(hidden_gates + hidden_base + HIDDEN_SIZE * hidden_stride_1)
    hidden_new = tl.load(hidden_gates + hidden_base + 2 * HIDDEN_SIZE * hidden_stride_1)
    hx_value = tl.load(hx + row * hx_stride_0 + cols * hx_stride_1)

    if HAS_BIAS:
        bias_reset = tl.load(input_bias + cols * input_bias_stride)
        bias_update = tl.load(
            input_bias + HIDDEN_SIZE * input_bias_stride + cols * input_bias_stride
        )
        bias_new = tl.load(
            input_bias + 2 * HIDDEN_SIZE * input_bias_stride + cols * input_bias_stride
        )
        hidden_bias_reset = tl.load(hidden_bias + cols * hidden_bias_stride)
        hidden_bias_update = tl.load(
            hidden_bias + HIDDEN_SIZE * hidden_bias_stride + cols * hidden_bias_stride
        )
        hidden_bias_new = tl.load(
            hidden_bias
            + 2 * HIDDEN_SIZE * hidden_bias_stride
            + cols * hidden_bias_stride
        )
    else:
        zero = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        bias_reset = zero
        bias_update = zero
        bias_new = zero
        hidden_bias_reset = zero
        hidden_bias_update = zero
        hidden_bias_new = zero

    reset_gate, update_gate, new_gate, hx_value, hidden_new, hy_value = (
        _thnn_fused_gru_cell_gates(
            input_reset,
            input_gate,
            input_new,
            hidden_reset,
            hidden_gate,
            hidden_new,
            hx_value,
            bias_reset,
            bias_update,
            bias_new,
            hidden_bias_reset,
            hidden_bias_update,
            hidden_bias_new,
            HAS_BIAS,
            IS_FP64,
        )
    )

    tl.store(hy + row * hy_stride_0 + cols * hy_stride_1, hy_value)
    workspace_base = row * workspace_stride_0 + cols * workspace_stride_1
    tl.store(workspace + workspace_base, reset_gate)
    tl.store(
        workspace + workspace_base + HIDDEN_SIZE * workspace_stride_1,
        update_gate,
    )
    tl.store(
        workspace + workspace_base + 2 * HIDDEN_SIZE * workspace_stride_1,
        new_gate,
    )
    tl.store(
        workspace + workspace_base + 3 * HIDDEN_SIZE * workspace_stride_1,
        hx_value,
    )
    tl.store(
        workspace + workspace_base + 4 * HIDDEN_SIZE * workspace_stride_1,
        hidden_new,
    )


@triton.jit
def _thnn_fused_gru_cell_flat_kernel_lazy(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    workspace,
    HIDDEN_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat variant that loads each value where it is first used and stores each
    result as soon as it is computed.

    Same arithmetic as ``_thnn_fused_gru_cell_flat_kernel_nomask``, including the
    order of every addition; only the placement of the loads and stores differs.
    Requires ``N % BLOCK_SIZE == 0``.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // HIDDEN_SIZE
    cols = offsets - rows * HIDDEN_SIZE
    gate_base = rows * (3 * HIDDEN_SIZE) + cols
    workspace_base = rows * (5 * HIDDEN_SIZE) + cols

    hx_value = tl.load(hx + offsets)
    if not IS_FP64:
        hx_value = hx_value.to(tl.float32)
    tl.store(workspace + workspace_base + 3 * HIDDEN_SIZE, hx_value)

    reset_gate = tl.load(input_gates + gate_base)
    hidden_reset = tl.load(hidden_gates + gate_base)
    if not IS_FP64:
        reset_gate = reset_gate.to(tl.float32)
        hidden_reset = hidden_reset.to(tl.float32)
    reset_gate = reset_gate + hidden_reset
    if HAS_BIAS:
        bias_reset = tl.load(input_bias + cols)
        hidden_bias_reset = tl.load(hidden_bias + cols)
        if not IS_FP64:
            bias_reset = bias_reset.to(tl.float32)
            hidden_bias_reset = hidden_bias_reset.to(tl.float32)
        reset_gate += bias_reset
        reset_gate += hidden_bias_reset
    reset_gate = tl.sigmoid(reset_gate)
    tl.store(workspace + workspace_base, reset_gate)

    update_gate = tl.load(input_gates + gate_base + HIDDEN_SIZE)
    hidden_gate = tl.load(hidden_gates + gate_base + HIDDEN_SIZE)
    if not IS_FP64:
        update_gate = update_gate.to(tl.float32)
        hidden_gate = hidden_gate.to(tl.float32)
    update_gate = update_gate + hidden_gate
    if HAS_BIAS:
        bias_update = tl.load(input_bias + HIDDEN_SIZE + cols)
        hidden_bias_update = tl.load(hidden_bias + HIDDEN_SIZE + cols)
        if not IS_FP64:
            bias_update = bias_update.to(tl.float32)
            hidden_bias_update = hidden_bias_update.to(tl.float32)
        update_gate += bias_update
        update_gate += hidden_bias_update
    update_gate = tl.sigmoid(update_gate)
    tl.store(workspace + workspace_base + HIDDEN_SIZE, update_gate)

    hidden_new = tl.load(hidden_gates + gate_base + 2 * HIDDEN_SIZE)
    if not IS_FP64:
        hidden_new = hidden_new.to(tl.float32)
    if HAS_BIAS:
        hidden_bias_new = tl.load(hidden_bias + 2 * HIDDEN_SIZE + cols)
        if not IS_FP64:
            hidden_bias_new = hidden_bias_new.to(tl.float32)
        hidden_new += hidden_bias_new
    tl.store(workspace + workspace_base + 4 * HIDDEN_SIZE, hidden_new)

    input_new = tl.load(input_gates + gate_base + 2 * HIDDEN_SIZE)
    if not IS_FP64:
        input_new = input_new.to(tl.float32)
    if HAS_BIAS:
        bias_new = tl.load(input_bias + 2 * HIDDEN_SIZE + cols)
        if not IS_FP64:
            bias_new = bias_new.to(tl.float32)
        input_new += bias_new
    new_gate = tl_extra_shim.tanh(input_new + reset_gate * hidden_new)
    tl.store(workspace + workspace_base + 2 * HIDDEN_SIZE, new_gate)

    tl.store(hy + offsets, new_gate + update_gate * (hx_value - new_gate))


@triton.jit
def _thnn_fused_gru_cell_flat_kernel_nomask(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    workspace,
    HIDDEN_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Maskless flat variant; requires ``N % BLOCK_SIZE == 0``."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // HIDDEN_SIZE
    cols = offsets - rows * HIDDEN_SIZE
    gate_base = rows * (3 * HIDDEN_SIZE) + cols

    input_reset = tl.load(input_gates + gate_base)
    input_gate = tl.load(input_gates + gate_base + HIDDEN_SIZE)
    input_new = tl.load(input_gates + gate_base + 2 * HIDDEN_SIZE)
    hidden_reset = tl.load(hidden_gates + gate_base)
    hidden_gate = tl.load(hidden_gates + gate_base + HIDDEN_SIZE)
    hidden_new = tl.load(hidden_gates + gate_base + 2 * HIDDEN_SIZE)
    hx_value = tl.load(hx + offsets)

    if not IS_FP64:
        input_reset = input_reset.to(tl.float32)
        input_gate = input_gate.to(tl.float32)
        input_new = input_new.to(tl.float32)
        hidden_reset = hidden_reset.to(tl.float32)
        hidden_gate = hidden_gate.to(tl.float32)
        hidden_new = hidden_new.to(tl.float32)
        hx_value = hx_value.to(tl.float32)

    if HAS_BIAS:
        bias_reset = tl.load(input_bias + cols)
        bias_update = tl.load(input_bias + HIDDEN_SIZE + cols)
        bias_new = tl.load(input_bias + 2 * HIDDEN_SIZE + cols)
        hidden_bias_reset = tl.load(hidden_bias + cols)
        hidden_bias_update = tl.load(hidden_bias + HIDDEN_SIZE + cols)
        hidden_bias_new = tl.load(hidden_bias + 2 * HIDDEN_SIZE + cols)
        if not IS_FP64:
            bias_reset = bias_reset.to(tl.float32)
            bias_update = bias_update.to(tl.float32)
            bias_new = bias_new.to(tl.float32)
            hidden_bias_reset = hidden_bias_reset.to(tl.float32)
            hidden_bias_update = hidden_bias_update.to(tl.float32)
            hidden_bias_new = hidden_bias_new.to(tl.float32)
    else:
        zero = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        bias_reset = zero
        bias_update = zero
        bias_new = zero
        hidden_bias_reset = zero
        hidden_bias_update = zero
        hidden_bias_new = zero

    reset_gate = input_reset + hidden_reset
    update_gate = input_gate + hidden_gate
    if HAS_BIAS:
        reset_gate += bias_reset
        reset_gate += hidden_bias_reset
        update_gate += bias_update
        update_gate += hidden_bias_update
        input_new += bias_new
        hidden_new += hidden_bias_new

    reset_gate = tl.sigmoid(reset_gate)
    update_gate = tl.sigmoid(update_gate)
    new_gate = input_new + reset_gate * hidden_new
    new_gate = tl_extra_shim.tanh(new_gate)
    hy_value = new_gate + update_gate * (hx_value - new_gate)

    tl.store(hy + offsets, hy_value)
    workspace_base = rows * (5 * HIDDEN_SIZE) + cols
    tl.store(workspace + workspace_base, reset_gate)
    tl.store(workspace + workspace_base + HIDDEN_SIZE, update_gate)
    tl.store(workspace + workspace_base + 2 * HIDDEN_SIZE, new_gate)
    tl.store(workspace + workspace_base + 3 * HIDDEN_SIZE, hx_value)
    tl.store(workspace + workspace_base + 4 * HIDDEN_SIZE, hidden_new)


def _validate_tensor(tensor, shape, dtype, device, name):
    if tensor.ndim != len(shape) or tuple(tensor.shape) != tuple(shape):
        raise RuntimeError(f"{name} must have shape {tuple(shape)}")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype}")
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on {device}")


def _validate_inputs(input_gates, hidden_gates, hx, input_bias, hidden_bias):
    if input_gates.ndim != 2:
        raise RuntimeError("input_gates must be a 2-D tensor")
    if input_gates.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "_thnn_fused_gru_cell only supports FP16, BF16, FP32, and FP64"
        )
    batch_size, gate_size = input_gates.shape
    if gate_size % 3 != 0:
        raise RuntimeError("input_gates.size(1) must be divisible by 3")
    hidden_size = gate_size // 3
    dtype = input_gates.dtype
    device = input_gates.device
    _validate_tensor(
        hidden_gates, (batch_size, gate_size), dtype, device, "hidden_gates"
    )
    _validate_tensor(hx, (batch_size, hidden_size), dtype, device, "hx")
    if input_bias is not None:
        _validate_tensor(input_bias, (gate_size,), dtype, device, "input_bias")
        if hidden_bias is None:
            raise RuntimeError("hidden_bias must be defined when input_bias is defined")
    if hidden_bias is not None:
        _validate_tensor(hidden_bias, (gate_size,), dtype, device, "hidden_bias")
    return batch_size, hidden_size


def _prepare_out(out, shape, dtype, device, name):
    if out.dtype != dtype:
        raise RuntimeError(f"Expected {name} to have dtype {dtype}, got {out.dtype}")
    if out.device != device:
        raise RuntimeError(f"Expected {name} to be on {device}, got {out.device}")
    if tuple(out.shape) != tuple(shape):
        out.resize_(shape)


def _shares_storage(left, right):
    return (
        left.device == right.device
        and left.untyped_storage().nbytes() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _use_flat_path(
    input_gates, hidden_gates, hx, input_bias, hidden_bias, hy, workspace
):
    """The flat 1-D kernel hard-codes the contiguous layout, so it is only used
    when every operand is contiguous and ``hidden_size`` fits a constexpr."""
    tensors = [input_gates, hidden_gates, hx, hy, workspace]
    if input_bias is not None:
        tensors += [input_bias, hidden_bias]
    return all(t.is_contiguous() for t in tensors)


def _launch_gru_cell(
    input_gates,
    hidden_gates,
    hx,
    input_bias,
    hidden_bias,
    hy,
    workspace,
):
    batch_size, hidden_size = hx.shape
    if batch_size * hidden_size == 0:
        return
    has_bias = input_bias is not None
    if _use_flat_path(
        input_gates, hidden_gates, hx, input_bias, hidden_bias, hy, workspace
    ):
        n_elements = batch_size * hidden_size
        form, block_size = _flat_plan(n_elements, input_gates.element_size())
        # Both flat forms need the block to divide the element count exactly;
        # otherwise the 2-D grid below runs.
        if n_elements % block_size == 0:
            kernel = (
                _thnn_fused_gru_cell_flat_kernel_lazy
                if form == "lazy"
                else _thnn_fused_gru_cell_flat_kernel_nomask
            )
            with torch_device_fn.device(input_gates.device):
                kernel[(n_elements // block_size,)](
                    input_gates,
                    hidden_gates,
                    hx,
                    input_bias if has_bias else input_gates,
                    hidden_bias if has_bias else hidden_gates,
                    hy,
                    workspace,
                    hidden_size,
                    HAS_BIAS=has_bias,
                    IS_FP64=input_gates.dtype == torch.float64,
                    BLOCK_SIZE=block_size,
                )
            return
    block_size = _pick_block_size(hidden_size)
    n_chunks = triton.cdiv(hidden_size, block_size)
    grid = (batch_size, n_chunks)
    maskless = hidden_size % block_size == 0
    kernel = (
        _thnn_fused_gru_cell_kernel_nomask if maskless else _thnn_fused_gru_cell_kernel
    )
    with torch_device_fn.device(input_gates.device):
        kernel[grid](
            input_gates,
            hidden_gates,
            hx,
            input_bias if has_bias else input_gates,
            hidden_bias if has_bias else hidden_gates,
            hy,
            workspace,
            hidden_size,
            input_gates.stride(0),
            input_gates.stride(1),
            hidden_gates.stride(0),
            hidden_gates.stride(1),
            hx.stride(0),
            hx.stride(1),
            input_bias.stride(0) if has_bias else 0,
            hidden_bias.stride(0) if has_bias else 0,
            hy.stride(0),
            hy.stride(1),
            workspace.stride(0),
            workspace.stride(1),
            HAS_BIAS=has_bias,
            IS_FP64=input_gates.dtype == torch.float64,
            BLOCK_SIZE=block_size,
        )


def _thnn_fused_gru_cell(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN _THNN_FUSED_GRU_CELL")
    batch_size, hidden_size = _validate_inputs(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )
    options = {"dtype": input_gates.dtype, "device": input_gates.device}
    hy = torch.empty((batch_size, hidden_size), **options)
    workspace = torch.empty((batch_size, 5 * hidden_size), **options)
    _launch_gru_cell(
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        hy,
        workspace,
    )
    return hy, workspace


def _thnn_fused_gru_cell_out(
    input_gates: torch.Tensor,
    hidden_gates: torch.Tensor,
    hx: torch.Tensor,
    input_bias: torch.Tensor = None,
    hidden_bias: torch.Tensor = None,
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _THNN_FUSED_GRU_CELL.OUT")
    batch_size, hidden_size = _validate_inputs(
        input_gates, hidden_gates, hx, input_bias, hidden_bias
    )
    if _shares_storage(out0, out1):
        raise RuntimeError("out0 and out1 must not share storage")
    aliases_input = any(
        operand is not None and _shares_storage(output, operand)
        for output in (out0, out1)
        for operand in (input_gates, hidden_gates, hx, input_bias, hidden_bias)
    )
    temporary = None
    if aliases_input:
        temporary = _thnn_fused_gru_cell(
            input_gates, hidden_gates, hx, input_bias, hidden_bias
        )
    _prepare_out(
        out0,
        (batch_size, hidden_size),
        input_gates.dtype,
        input_gates.device,
        "out0",
    )
    _prepare_out(
        out1,
        (batch_size, 5 * hidden_size),
        input_gates.dtype,
        input_gates.device,
        "out1",
    )
    if temporary is not None:
        out0.copy_(temporary[0])
        out1.copy_(temporary[1])
        return out0, out1
    _launch_gru_cell(
        input_gates,
        hidden_gates,
        hx,
        input_bias,
        hidden_bias,
        out0,
        out1,
    )
    return out0, out1
