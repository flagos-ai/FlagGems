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
from typing import List, Tuple, Union

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils.tensor_wrapper import StridedBuffer

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
# NOTE: is_cat=True makes the pointwise codegen emit buffer_size_limit=512 for the
# strided copy. Without it (default buffer_size_limit=2048), an inner-dim cat whose
# output row-stride is not vector-aligned (e.g. fp32 cat along the last dim
# producing an odd width like 25) makes the vectorized store over-run the output
# buffer -> `error code=700, illegal memory access`. is_cat helps fp16/fp32/bf16/
# int16 but int32 inner-dim cat STILL overruns even with it (config knobs
# isCloseVectorization / buffer_size_limit=256 do not help either). The generic
# gems copy_ has the SAME strided-store bug, so a torch copy_ fallback also crashes.
# See the concatenate/cat family (harness/perf_ir_3/ir-concatenate-dev7.log).
#
# Robust fix: the illegal access ONLY happens when the triton copy writes into a
# NON-CONTIGUOUS output slab (inner-dim cat). So we always arrange writes to hit a
# CONTIGUOUS dim-0 slab:
#   * dim==0 (the benchmark case): each input already maps to a contiguous block ->
#     tuned triton block-DMA copy straight into the fresh output (fast path).
#   * dim>0: permute the cat dim to the front, copy into a contiguous permuted
#     buffer (every write target is a contiguous dim-0 slab -> no overrun for any
#     dtype), then permute the result back to the original layout.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    is_cat=True,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # remove torch.Size([0]) tensors
    device = A[0].device
    dtype = A[0].dtype
    A = list(A)
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)
    if len(A) == 0:
        return torch.tensor([], device=device, dtype=dtype)
    elif len(A) == 1:
        return A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)

    nd = A[0].ndim
    if dim == 0:
        # Fast path (benchmark case): each input maps to a CONTIGUOUS output slab
        # -> tuned triton block-DMA copy straight into the fresh output.
        out0 = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
        out0_strides = out0.stride()
        start = 0
        for a in A:
            w = a.shape[0]
            in_view = StridedBuffer(a, a.shape, a.stride())
            out_view = StridedBuffer(
                out0, a.shape, out0_strides, offset=start * out0_strides[0]
            )
            copy_func.instantiate(a.ndim)(in_view, out0=out_view)
            start += w
        return out0

    # Inner-dim cat: permute the cat dim to the front so every write target is a
    # CONTIGUOUS dim-0 slab (avoids the strided-store overrun for all dtypes),
    # then permute the result back to the original layout.
    perm = [dim] + [i for i in range(nd) if i != dim]
    inv = [0] * nd
    for i, p in enumerate(perm):
        inv[p] = i
    outp_shape = [out_shape[p] for p in perm]
    outp = torch.empty(outp_shape, dtype=A[0].dtype, device=A[0].device)
    outp_strides = outp.stride()
    start = 0
    for a in A:
        ap = a.permute(perm).contiguous()
        w = ap.shape[0]
        in_view = StridedBuffer(ap, ap.shape, ap.stride())
        out_view = StridedBuffer(
            outp, ap.shape, outp_strides, offset=start * outp_strides[0]
        )
        copy_func.instantiate(ap.ndim)(in_view, out0=out_view)
        start += w
    return outp.permute(inv)


def cat_out(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    dim: int = 0,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    # cat.out was NOT overridden by kunlunxin before, so it fell back to the
    # generic ops/cat.py::cat_out, whose hand-written raw @triton.jit
    # `cat_copy_func_kernel_4` (fixed BLOCK=1024, no @libentry caching)
    # recompiles per shape/launch -> the IR dump shows ~2750
    # cat_copy_func_kernel_4 modules (43MB / 546K lines). Route cat.out through
    # the exact same tuned StridedBuffer + pointwise_dynamic `copy_func` path
    # that the kunlunxin `cat` override already uses (bounded tiles + autoGrid +
    # libentry cache), just writing into the caller-provided `out`.
    logger.debug("GEMS_KUNLUNXIN CAT_OUT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    A = list(A)
    # remove torch.Size([0]) tensors
    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)

    if len(A) == 0:
        out.resize_(0)
        return out

    if len(A) == 1:
        t = A[0]
        out.resize_(t.shape)
        out.copy_(t)
        return out

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(s[dim] for s in inp_shapes)
    if list(out.shape) != out_shape:
        out.resize_(out_shape)

    # Build the concatenation via the safe `cat` (which never writes into a strided
    # output slab), then copy into the caller-provided `out`. `out` is contiguous
    # after resize, so this final copy is a plain contiguous copy (safe for all
    # dtypes, including int32).
    result = cat(A, dim)
    out.copy_(result)
    return out
