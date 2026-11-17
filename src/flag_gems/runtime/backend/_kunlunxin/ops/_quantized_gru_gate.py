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
# GRU gate chain for the Kunlunxin ``quantized_gru`` override, kept in its own
# module: on TritonXPU a kernel that contains both ``tl.dot`` and
# ``tl.sigmoid``/``tanh`` fails to lower (``ConvertTritonSDNNToLLVM``), and on a
# *cold* compile a sigmoid/tanh-bearing kernel defined inside a module that also
# holds ``tl.dot`` kernels can abort with ``KeyError: 'tt.sigmoid'``.  Keeping
# the pointwise gate here separates it from every dot kernel.  Sigmoid is
# expressed through ``tanh`` because ``triton_xpu.vsigmoidf`` fails to legalize
# on flat 1-D/2-D tiles on this image.
#
# Semantics (PyTorch GRU):
#   r = sigmoid(g_r + recur_r)          # recur_r already carries b_hr
#   z = sigmoid(g_z + recur_z)          # recur_z already carries b_hz
#   n = tanh(g_n + r * recur_n)         # recur_n already carries b_hn, and
#                                       # b_hn must live *inside* r * (...)
#   h' = z * h_prev + (1 - z) * n

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

_tanh = tl_extra_shim.tanh


@triton.jit
def _sigmoid(x):
    return 0.5 * (1.0 + _tanh(0.5 * x))


@triton.jit
def quantized_gru_gate_kernel(
    pre_ptr,  # (3, pad_mb * GS) fp32 recurrent contributions, gate-major
    gates_ptr,  # (3, pad_m * GS) fp32 input projections, gate-major
    state_ptr,  # (pad_mb, GS) fp32 previous hidden state
    next_ptr,  # (pad_mb, GS) fp32 next hidden state (padded lanes zeroed)
    out_ptr,  # (rows, dirs * H) output view for this direction
    final_ptr,  # (batch_size, H) final hidden state for this direction
    row_off,  # first packed/unpacked row of this timestep
    active_batch,  # number of live sequences at this timestep
    batch_size,  # rows in ``final_ptr``
    pre_plane_stride,
    g_row_stride,
    g_plane_stride,
    out_row_stride,
    dir_off,
    H: tl.constexpr,
    GS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    base = rb[:, None] * GS + rn[None, :]
    pre_r = tl.load(pre_ptr + base)
    pre_z = tl.load(pre_ptr + pre_plane_stride + base)
    pre_n = tl.load(pre_ptr + 2 * pre_plane_stride + base)
    g_n = tl.load(
        gates_ptr
        + 2 * g_plane_stride
        + (row_off + rb)[:, None] * g_row_stride
        + rn[None, :]
    )
    h_prev = tl.load(state_ptr + base)

    reset = _sigmoid(pre_r)
    update = _sigmoid(pre_z)
    new = _tanh(g_n + reset * pre_n)
    hy = update * h_prev + (1.0 - update) * new

    valid = (rb < active_batch)[:, None] & (rn < H)[None, :]
    # Padded lanes must stay exactly zero so they do not contaminate the next
    # step's recurrent dot; store them unmasked (in-bounds by construction).
    tl.store(next_ptr + base, tl.where(valid, hy, 0.0))

    # Scatter the live rows into the output view.  Both indices are clamped
    # before being used as addresses and the store is masked, so no out-of-bounds
    # write can happen.
    row_c = row_off + tl.minimum(rb, active_batch - 1)
    col_c = dir_off + tl.minimum(rn, H - 1)
    tl.store(
        out_ptr + row_c[:, None] * out_row_stride + col_c[None, :],
        hy.to(out_ptr.dtype.element_ty),
        mask=valid,
    )

    # Final hidden state: rows are the *un-offset* batch indices, and only rows
    # below batch_size exist.  The last step that leaves a row active wins.
    fvalid = (rb < active_batch)[:, None] & (rb < batch_size)[:, None] & (
        rn < H
    )[None, :]
    fr = tl.minimum(rb, active_batch - 1)
    fc = tl.minimum(rn, H - 1)
    tl.store(
        final_ptr + fr[:, None] * H + fc[None, :],
        hy.to(final_ptr.dtype.element_ty),
        mask=fvalid,
    )
