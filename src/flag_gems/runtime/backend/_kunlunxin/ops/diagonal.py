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
from triton.runtime import driver

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)
_prewarmed = False


# Band scatter kernel for diagonal_backward.  Writes the diagonal band of the
# (already zero-filled) output tensor from the contiguous gradient.
@triton.jit(
    do_not_specialize=[
        "grad_ptr",
        "out_ptr",
        "col_tab_ptr",
        "row_out_ptr",
        "D",
    ]
)
def diag_bwd_scatter_kernel(
    grad_ptr,
    out_ptr,
    col_tab_ptr,
    row_out_ptr,
    D,
    BLOCK: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    cols = pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    col_off = tl.load(col_tab_ptr + cols, mask=mask, other=0)
    row_off = tl.load(row_out_ptr + pid_row)
    val = tl.load(grad_ptr + pid_row * D + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row_off + col_off, val, mask=mask)


def _torch_contiguous(shape, strides):
    """PyTorch's compactness rule: a size-1 dim never constrains its stride."""
    acc = 1
    for n, s in zip(reversed(shape), reversed(strides)):
        if int(n) != 1:
            if int(s) != acc:
                return False
            acc *= int(n)
    return True


def _tle_eligible(view_shape, view_strides, src_strides, elem_size):
    """Whether `tle_copy(contiguous src, diag view)` will take a real path.

    Mirrors `utils/tle_copy.py`'s decision for the two operands this file hands
    it -- a freshly allocated contiguous source and the diagonal view of the
    output -- so the DMA path is only attempted where the vendor copy can
    express the layout, and never where its TMA descriptor build asserts.

    Only the on-chip transpose is accepted: it is the branch that beats the
    scatter kernel on the layouts this op sees (a batched diagonal whose outer
    dim is the contiguous one).  The row branch's remaining envelope reads one
    side per element, which has no advantage over the kernel above.
    """
    if elem_size == 1:
        return False
    if _torch_contiguous(view_shape, view_strides):
        return False

    dims = [
        (int(n), int(ss), int(ds))
        for n, ss, ds in zip(view_shape, src_strides, view_strides)
        if int(n) != 1
    ]
    if not dims:
        return False
    dims.sort(key=lambda d: abs(d[2]))
    merged = []
    for n, ss, ds in dims:
        if merged:
            pn, pss, pds = merged[-1]
            if ss == pss * pn and ds == pds * pn:
                merged[-1] = (pn * n, pss, pds)
                continue
        merged.append((n, ss, ds))
    if len(merged) > 5:
        return False
    cols, s_col, d_col = merged[0]
    if cols > 1 and d_col != 1:
        return False
    if cols > 1 and s_col == 0:
        return False
    if len(merged) > 1:
        rows, s_row, _ = merged[1]
    else:
        rows, s_row = 1, 0
    return cols > 1 and s_col != 1 and rows > 1 and s_row == 1


class _Plan:
    """Everything about a diagonal_backward call that depends only on
    (input_sizes, offset, dim1, dim2, dtype, device) -- so the per-call host
    path is a dict lookup plus the two unavoidable device ops.

    The dominant cost of this op under the benchmark caliber is the *host* path
    (measured 104-120 us/call at HEAD, against a ~26 us native baseline at
    (64,64)): the event window only hides the host while the host stays ahead of
    the device, which it does up to roughly 90 us/call on this box.  Recomputing
    the output strides, the diagonal geometry and the two lookup tables every
    call is a large slice of that, and none of it can change within a shape.
    """

    __slots__ = (
        "shape",
        "strides",
        "numel",
        "D",
        "rows",
        "block",
        "ncb",
        "col_tab",
        "row_tab",
        "diag_numel",
        "use_tle",
        "view_shape",
        "diag_stride",
        "base",
        "outer_shape",
        "outer_strides",
        "kernel",
    )


_plan_cache = {}
_PLAN_CACHE_MAX = 1024


def _contig_strides(shape):
    st = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        st[i] = acc
        acc *= shape[i]
    return st, acc


def _build_plan(input_sizes, offset, dim1, dim2, device, elem_size):
    """Precompute the layout/geometry (+ lazily, the two kernel lookup tables).

    Returns None for the degenerate zero-length diagonal (D == 0), which the
    caller answers with the freshly allocated (and, if non-empty, zeroed)
    output.
    """
    ndim = len(input_sizes)
    a, b = int(dim1), int(dim2)
    if a < 0:
        a += ndim
    if b < 0:
        b += ndim
    off = int(offset)
    # torch.diagonal normalises by swapping dim1/dim2 and negating the offset.
    if a > b:
        a, b = b, a
        off = -off

    strides, numel = _contig_strides(input_sizes)
    A, B = int(input_sizes[a]), int(input_sizes[b])
    if off >= 0:
        D = min(A, B - off)
        base = off * strides[b]
    else:
        D = min(A + off, B)
        base = (-off) * strides[a]
    if D <= 0:
        return None

    outer_dims = [i for i in range(ndim) if i != a and i != b]
    outer_shape = [int(input_sizes[i]) for i in outer_dims]
    outer_strides = [strides[i] for i in outer_dims]
    rows = 1
    for s in outer_shape:
        rows *= s

    diag_stride = strides[a] + strides[b]
    view_shape = outer_shape + [D]
    view_strides = outer_strides + [diag_stride]
    src_strides, _ = _contig_strides(view_shape)

    plan = _Plan()
    plan.shape = tuple(int(x) for x in input_sizes)
    plan.strides = tuple(strides)
    plan.numel = numel
    plan.D = D
    plan.rows = rows
    plan.block = 256 if D <= 256 else 512
    plan.ncb = (D + plan.block - 1) // plan.block
    plan.col_tab = None
    plan.row_tab = None
    plan.diag_numel = D * rows
    plan.use_tle = _tle_eligible(view_shape, view_strides, src_strides, elem_size)
    plan.view_shape = tuple(view_shape)
    plan.diag_stride = diag_stride
    plan.base = base
    plan.outer_shape = outer_shape
    plan.outer_strides = outer_strides
    plan.kernel = None
    return plan


def _ensure_tables(plan, device):
    """Build the scatter kernel's lookup tables (only needed off the DMA path)."""
    if plan.col_tab is not None:
        return
    if plan.rows == 1:
        plan.row_tab = torch.full((1,), plan.base, dtype=torch.int64, device=device)
    else:
        tab = torch.zeros(1, dtype=torch.int64, device=device)
        for size, stride in zip(plan.outer_shape, plan.outer_strides):
            axis = torch.arange(size, dtype=torch.int64, device=device) * stride
            tab = (tab.unsqueeze(-1) + axis.view(1, -1)).reshape(-1)
        plan.row_tab = tab + plan.base
    plan.col_tab = (
        torch.arange(plan.ncb * plan.block, dtype=torch.int64, device=device)
        * plan.diag_stride
    )


def _launch_scatter(plan, grad, out, device_index):
    """Launch the scatter kernel through a bound `CompiledKernel.run`.

    `jitfn[grid](...)` re-derives the launcher binding on every call (~19 us of
    host here, against ~8.5 us for the bound run and ~5 us for a native launch);
    with everything the kernel keys on already `do_not_specialize`d, that work
    is pure overhead.  The binding is per plan, so it is derived once and the
    per-call cost is the two allocation pointers.
    """
    kernel = plan.kernel
    if kernel is None:
        kernel = diag_bwd_scatter_kernel.warmup(
            grad,
            out,
            plan.col_tab,
            plan.row_tab,
            plan.D,
            grid=(plan.ncb, plan.rows),
            BLOCK=plan.block,
        )
        plan.kernel = kernel
    stream = driver.active.get_current_stream(device_index)
    kernel.run(
        plan.ncb,
        plan.rows,
        1,
        stream,
        kernel.function,
        kernel.packed_metadata,
        None,
        None,
        None,
        grad.data_ptr(),
        out.data_ptr(),
        plan.col_tab.data_ptr(),
        plan.row_tab.data_ptr(),
        plan.D,
    )


def _prewarm(device):
    """Precompile kernel specializations so benchmark single-rep timing is not
    polluted by the JIT compile (~100ms per specialization on XPU)."""
    global _prewarmed
    if _prewarmed:
        return
    _prewarmed = True
    try:
        for dt in (torch.float16, torch.float32, torch.bfloat16):
            # scatter kernel, rows == 1
            shape = (2, 256)
            out = torch.empty_strided(
                shape, _contig_strides(shape)[0], dtype=dt, device=device
            )
            out.zero_()
            plan = _build_plan(shape, 0, 0, 1, device, out.element_size())
            _ensure_tables(plan, device)
            diag_bwd_scatter_kernel[(plan.ncb, plan.rows)](
                out, out, plan.col_tab, plan.row_tab, plan.D, BLOCK=plan.block
            )
            # scatter kernel, rows > 1 (the batched 2-D-diagonal layout)
            shape = (2, 256, 256)
            out = torch.empty_strided(
                shape, _contig_strides(shape)[0], dtype=dt, device=device
            )
            out.zero_()
            plan = _build_plan(shape, 0, 1, 2, device, out.element_size())
            _ensure_tables(plan, device)
            diag_bwd_scatter_kernel[(plan.ncb, plan.rows)](
                out, out, plan.col_tab, plan.row_tab, plan.D, BLOCK=plan.block
            )
            # the tle transpose path (batched diagonal, outer dim contiguous)
            x = torch.empty(shape, dtype=dt, device=device)
            dst = x.diagonal(0, 0, 1)
            src = torch.empty(dst.shape, dtype=dt, device=device)
            tle_copy(src, dst)
    except Exception:  # prewarm is best-effort
        pass


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS_KUNLUNXIN DIAGONAL_BACKWARD")
    device_ = grad_output.device
    sizes = tuple(input_sizes)
    # NOTE: cannot use torch.zeros/torch.empty here: inside gem dispatch both
    # route to the kunlunxin fill kernels which are 10-20x slower than the
    # native memset for big tensors.  torch.empty_strided is not registered in
    # gems and falls through to the native allocator (~us).
    key = (sizes, int(offset), int(dim1), int(dim2), grad_output.dtype, device_)
    plan = _plan_cache.get(key)
    if plan is None:
        plan = _build_plan(
            sizes, offset, dim1, dim2, device_, grad_output.element_size()
        )
        if len(_plan_cache) >= _PLAN_CACHE_MAX:
            _plan_cache.clear()
        _plan_cache[key] = plan

    if plan is None:
        strides, _ = _contig_strides(sizes)
        grad_input = torch.empty_strided(
            sizes, tuple(strides), dtype=grad_output.dtype, device=device_
        )
        if grad_input.numel() > 0:
            grad_input.zero_()
        return grad_input

    grad_input = torch.empty_strided(
        sizes, plan.strides, dtype=grad_output.dtype, device=device_
    )
    if plan.numel > 0:
        grad_input.zero_()  # native memset (zero_ is excluded from gem dispatch)
    if grad_output.numel() == 0:
        return grad_input

    _prewarm(device_)

    band_ok = (
        grad_output.is_contiguous()
        and grad_output.numel() == plan.diag_numel
        and tuple(grad_output.shape) == plan.view_shape
    )

    if plan.use_tle and band_ok:
        diag = torch.diagonal(grad_input, int(offset), int(dim1), int(dim2))
        if tle_copy(grad_output, diag):
            return grad_input

    if band_ok:
        # BLOCK capped at 512: masked/strided scatter at BLOCK >= 1024 is
        # miscompiled by the XPU triton backend (see diagonal_copy notes).
        _ensure_tables(plan, device_)
        dev_idx = grad_output.device.index
        _launch_scatter(
            plan,
            grad_output,
            grad_input,
            torch.cuda.current_device() if dev_idx is None else dev_idx,
        )
    else:
        diag = torch.diagonal(grad_input, int(offset), int(dim1), int(dim2))
        copy_func.instantiate(grad_output.ndim)(grad_output, out0=diag)
    return grad_input
