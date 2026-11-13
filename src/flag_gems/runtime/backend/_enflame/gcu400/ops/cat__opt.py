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

"""cat 的 dim=0 线性快速路径（纯 helper，不做任何模块绑定）。

KernelGen `cat_copy_func_kernel_4` 对每个元素做 3 次 int64 除法/取模。
dim==0 时 pre_idx 恒为 0 且 (idx//P)*P + idx%P == idx，故
    out_idx == idx + 该输入在输出中的起始元素偏移
可完全去掉坐标分解。探针实测 2.1x(小)~26.7x(大)，逐位与 torch.cat 一致。

由 `cat.py` 的 `cat()` 在函数体内调用；不满足严格边界时返回 None，
由调用方回退原有完整逻辑（float8 / 非 0 dim / >4 输入 / dtype 不一致 /
非连续 / 除 cat 维外 shape 不一致）。
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
    """返回拼接结果；不适用快速路径时返回 None。"""
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
    """cat.out 的 dim=0 线性快速路径：直接写进传入的 out。

    不适用时返回 False，由 `cat.py::cat_out` 回退原有完整逻辑
    （非 0 dim / >4 输入 / float8 / dtype 不一致 / 非连续输入 /
    out 非连续 / dtype 不符 / shape 不符 / 与输入别名）。
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
