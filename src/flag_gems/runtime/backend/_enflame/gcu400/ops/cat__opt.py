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

"""dim=0 linear fast path for cat (pure helper, no module binding).

KernelGen `cat_copy_func_kernel_4` performs 3 int64 divisions / modulos per element.
With dim==0, pre_idx is always 0 and (idx//P)*P + idx%P == idx, so
    out_idx == idx + the starting element offset of that input within the output
can drop coordinate decomposition entirely. Probe measured 2.1x (small) ~ 26.7x (large), bit-identical to torch.cat.

Called from `cat()` in `cat.py` inside the function body; returns None when the strict boundaries are not met,
so the caller falls back to the original full logic (float8 / non-zero dim / >4 inputs / dtype mismatch /
non-contiguous / shape mismatch outside the cat dimension).
"""

import torch
import triton
import triton.language as tl

__all__ = ["try_fast_cat", "try_fast_cat_out"]


@triton.jit
def _linear_cat_kernel(
    Out,
    In0,
    In1,
    In2,
    In3,
    n0,
    n1,
    n2,
    n3,
    o0,
    o1,
    o2,
    o3,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    off = tl.arange(0, BLOCK)
    idx = pid_x * BLOCK + off
    if pid_y == 0:
        p, n, o = In0, n0, o0
    elif pid_y == 1:
        p, n, o = In1, n1, o1
    elif pid_y == 2:
        p, n, o = In2, n2, o2
    else:
        p, n, o = In3, n3, o3
    mask = (idx < n) & (pid_y < NT)
    d = tl.load(p + idx, mask=mask, other=0)
    tl.store(Out + o + idx, d, mask=mask)


def _can_fast_path(A, dim):
    if not (1 <= len(A) <= 4):
        return False
    rank = A[0].ndim
    d = dim + rank if dim < 0 else dim
    if d != 0 or rank < 1:
        return False
    first = A[0]
    if str(first.dtype) == "torch.float8_e8m0fnu":
        return False
    tail = tuple(first.shape[1:])
    for t in A:
        if (
            t.dtype != first.dtype
            or t.device != first.device
            or not t.is_contiguous()
            or t.ndim != rank
            or tuple(t.shape[1:]) != tail
        ):
            return False
    return True


def try_fast_cat(A, dim):
    """Return the concatenation result; return None when the fast path does not apply."""
    if not _can_fast_path(A, dim):
        return None
    n = len(A)
    ns = [A[j].numel() if j < n else 0 for j in range(4)]
    ptrs = [A[j] if j < n else A[0] for j in range(4)]
    offs = [0] * 4
    acc = 0
    for j in range(4):
        offs[j] = acc
        acc += ns[j]
    out_shape = (sum(t.shape[0] for t in A),) + tuple(A[0].shape[1:])
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    max_n = max(ns)
    BLOCK = 16384 if max_n < 65536 else 32768
    grid = (triton.cdiv(max_n, BLOCK), n)
    _linear_cat_kernel[grid](out, *ptrs, *ns, *offs, NT=n, BLOCK=BLOCK, num_warps=4)
    return out


def try_fast_cat_out(A, dim, out):
    """dim=0 linear fast path for cat.out: write directly into the passed-in out.

    Returns False when not applicable, so `cat.py::cat_out` falls back to the original full logic
    (non-zero dim / >4 inputs / float8 / dtype mismatch / non-contiguous input /
    non-contiguous out / dtype mismatch / shape mismatch / aliasing with input).
    """
    if not _can_fast_path(A, dim):
        return False
    n = len(A)
    first = A[0]
    out_shape = (sum(t.shape[0] for t in A),) + tuple(first.shape[1:])
    if (
        out.dtype != first.dtype
        or out.device != first.device
        or not out.is_contiguous()
        or tuple(out.shape) != tuple(out_shape)
    ):
        return False
    if any(out.data_ptr() == t.data_ptr() for t in A):
        return False

    ns = [A[j].numel() if j < n else 0 for j in range(4)]
    ptrs = [A[j] if j < n else A[0] for j in range(4)]
    offs = [0] * 4
    acc = 0
    for j in range(4):
        offs[j] = acc
        acc += ns[j]
    max_n = max(ns)
    BLOCK = 16384 if max_n < 65536 else 32768
    grid = (triton.cdiv(max_n, BLOCK), n)
    _linear_cat_kernel[grid](out, *ptrs, *ns, *offs, NT=n, BLOCK=BLOCK, num_warps=4)
    return True
