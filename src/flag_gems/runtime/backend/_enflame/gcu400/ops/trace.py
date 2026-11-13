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


# ---------------------------------------------------------------------------
# torch.trace non-floating-on-gcu shim (torch_gcu P21 defect workaround)
#
# torch_gcu's native topsaten trace aborts the whole process for every
# non-floating dtype: uint8/int8/int16/int32/int64/bool/complex64 inputs die
# with SIGABRT + core dump at the call site (even for a 1x1 tensor), while
# float16/float32/float64 work.  That kills any caller of torch.trace on an
# integral gcu tensor before FlagGems code runs: tests/test_trace.py computes
# its reference via torch.trace(ref_inp) and dies at the first int32/bool
# case, and the benchmark's native arm cannot run its int16/int32 shapes at
# all.  The ENFLAME_PT_OP_DEBUG_CONFIG=fallback_cpu=all_ops route rescues
# the integer cases but not bool, because CPU aten::trace itself does not
# implement Bool ("NotImplementedError: \"trace\" not implemented for
# 'Bool'").
#
# Route non-floating gcu tensors through CPU instead: torch.trace on CPU for
# integral/complex dtypes, and torch.sum(torch.diagonal(...)) for bool (the
# exact formula the upstream test uses for its CPU bool reference), with the
# result copied back to the original device.  Floating-point dtypes,
# non-gcu tensors and every other call shape pass through to the original
# implementation untouched, so float behavior (including the benchmark's
# native arm) is unchanged.  The rescue is also skipped while a FlagGems
# registrar has aten::trace registered (flag_gems.enable() /
# use_gems() context, e.g. the benchmark's candidate arm): the dispatcher
# then routes to the FlagGems Triton kernel, which handles every integral
# dtype and bool correctly, so callers keep the fast path.  Only the
# functional form is shimmed, mirroring the complex isclose shim in add.py;
# torch.Tensor.trace keeps native behavior.  Narrow, value-preserving, and
# safe to drop once topsaten trace supports non-floating dtypes.
# ---------------------------------------------------------------------------
def _gems_trace_dispatch_active():
    try:
        import flag_gems

        return "trace" in flag_gems.all_registered_keys()
    except Exception:
        return False


def _install_native_trace_shim():
    current = torch.trace
    if getattr(current, "_flaggems_gcu_trace_safe", False):
        return

    def _gcu_safe_trace(x, *args, **kwargs):
        if (
            isinstance(x, torch.Tensor)
            and x.device.type == "gcu"
            and not x.dtype.is_floating_point
            and not _gems_trace_dispatch_active()
        ):
            target = x.device
            cpu_x = x.cpu()
            if cpu_x.dtype == torch.bool:
                out = torch.sum(torch.diagonal(cpu_x))
            else:
                out = torch.trace(cpu_x)
            return out.to(target)
        return current(x, *args, **kwargs)

    _gcu_safe_trace._flaggems_gcu_trace_safe = True
    torch.trace = _gcu_safe_trace


_install_native_trace_shim()


@libentry()
@triton.jit
def trace_kernel(
    inp_ptr,
    out_ptr,
    num_diag,
    stride0,
    stride1,
    BLOCK_SIZE: tl.constexpr,
):
    inp_dtype = inp_ptr.type.element_ty
    if inp_dtype.is_int():
        acc_dtype = tl.int64
        other_val = 0
    elif inp_dtype == tl.float64:
        acc_dtype = tl.float64
        other_val = 0.0
    else:
        acc_dtype = tl.float32
        other_val = 0.0

    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)

    diag_stride = stride0 + stride1

    for i in range(0, tl.cdiv(num_diag, BLOCK_SIZE)):
        block_start = i * BLOCK_SIZE
        current_indices = block_start + tl.arange(0, BLOCK_SIZE)

        mask = current_indices < num_diag

        ptr_offsets = current_indices * diag_stride
        current_ptrs = inp_ptr + ptr_offsets

        vals = tl.load(current_ptrs, mask=mask, other=other_val)

        acc += vals.to(acc_dtype)

    final_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr, final_sum.to(out_ptr.type.element_ty))


def trace(self):
    logger.debug("GEMS TRACE")

    if self.ndim != 2:
        raise RuntimeError(
            f"trace: expected a 2D tensor, but got a {self.ndim}D tensor"
        )

    M, N = self.shape
    stride0, stride1 = self.stride()
    num_diag = min(M, N)
    if num_diag == 0:
        if self.dtype.is_floating_point:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        else:
            return torch.tensor(0, dtype=torch.int64, device=self.device)

    if self.dtype.is_floating_point:
        output_dtype = self.dtype
    else:
        output_dtype = torch.int64
    out = torch.empty((), dtype=output_dtype, device=self.device)

    grid = (1,)
    BLOCK_SIZE = 1024
    if num_diag < BLOCK_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(num_diag)
        if BLOCK_SIZE == 0:
            BLOCK_SIZE = 1

    with torch_device_fn.device(self.device):
        trace_kernel[grid](
            self,
            out,
            num_diag,
            stride0,
            stride1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
