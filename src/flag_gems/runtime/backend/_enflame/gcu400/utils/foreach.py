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
"""GCU400 专属 ``_foreach_*`` 执行层。

结构参照 ``_kunlunxin/utils/foreach.py``：后端在 ``utils/foreach.py`` 里自带
kernel / executor / launcher，公共 ``flag_gems.utils.foreach`` 一行不改。

公共 scalar kernel 把 ``.Scalar`` 的标量声明为 ``scalar_value: tl.float64``，
GCU400 没有 f64 ABI，这个签名在后端无法 lowering。这里像昆仑芯那样完全取消
by-value 标量参数：所有标量统一放进 device 侧 ``scalar_ptr`` 数组（``.Scalar``
在 host 侧广播成每个 tensor 一份），kernel 签名里不出现任何 f64。

公共执行器的 ``grid=(nt, cdiv(max_numel, BLOCK))`` 在大 shape 下还会越过
GCU400 ``grid.y <= 255`` 的硬件上限；这里沿用昆仑芯的 grid-stride：第二维
封顶到 ``MAX_GRID_DIM``，核内按该步长跨过剩余 tile。

dtype policy / 异常语义不从公共模块复制，只以只读方式复用其纯 helper
（promotion、校验、metadata、分配、launch 计数），未改动的公共执行器则显式
转发，使本模块成为 GCU400 foreach 的单一导入点。
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import foreach as _common

from .config_utils import MAX_GRID_DIM

# 公共执行器中与后端无关的部分；以绑定而非属性链的形式引用，便于静态检查。
check_tensor_list = _common.check_tensor_list
check_dtype_supported = _common.check_dtype_supported
has_internal_overlap = _common.has_internal_overlap
_finish_inplace = _common._finish_inplace
_promote_with_scalar = _common._promote_with_scalar
_resolve_scalars = _common._resolve_scalars
_dense = _common._dense
_device_meta = _common._device_meta
_pick_block = _common._pick_block
tl_dtype = _common.tl_dtype
_NUMEL = _common._NUMEL
_DATA_PTR = _common._DATA_PTR
_STATS = _common._STATS

# 未做后端适配的公共执行器 / 常量，转发后 ops 层只需 import 本模块。
_FAST_MATH = _common._FAST_MATH
_promote_dtypes = _common._promote_dtypes
foreach_binary_list = _common.foreach_binary_list
foreach_ternary = _common.foreach_ternary


@triton.jit
def foreach_binary_scalar_kernel(
    meta_ptr,
    scalar_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
    NCTA,
):
    """``NT`` tensors against one scalar each (GCU400, f64-free).

    ``meta_ptr`` int64 layout: ``[0, NT)`` input pointers, ``[NT, 2*NT)``
    output pointers, ``[2*NT, 3*NT)`` element counts.  ``scalar_ptr`` holds one
    scalar per tensor; ``.Scalar`` is broadcast on the host and ``.ScalarList``
    is uploaded as is.  There is deliberately no by-value scalar argument --
    that is what ``scalar_value: tl.float64`` costs on a backend without f64.

    ``NCTA`` is the number of CTAs along grid axis 1; each program strides over
    the remaining tiles with it, so the axis never has to grow past the
    hardware limit.
    """
    tensor_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    a_ptr = tl.load(meta_ptr + tensor_id).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + NT + tensor_id).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + tensor_id)

    s = tl.load(scalar_ptr + tensor_id)
    s = (s * ALPHA).to(S_DT) if ALPHA != 1 else s
    s = s.to(OUT_DT)

    # ``n_elements`` is an int64 metadata field; keeping the tile count in i32
    # makes the loop variable the same type as ``tile_id`` / ``NCTA`` and keeps
    # the address arithmetic on i32, as the common executor does.
    num_tiles = tl.cdiv(n_elements, BLOCK).to(tl.int32)
    for block_id in tl.range(tile_id, num_tiles, NCTA):
        idx = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        a = tl.load(a_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, fn(a.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT), mask=mask)


def _launch_scalar_group(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    vals: List[Any],
    fn: Callable,
    in_dt: torch.dtype,
    out_dt: torch.dtype,
    device: Any,
    alpha: float = 1,
) -> None:
    """One launch per ``(device, in_dtype, out_dtype)`` bucket."""
    numels = list(map(_NUMEL, srcs))
    max_numel = max(numels)
    if max_numel == 0:
        return
    nt = len(srcs)
    ptrs = list(map(_DATA_PTR, srcs))
    values = ptrs + (ptrs if dsts is srcs else list(map(_DATA_PTR, dsts))) + numels
    meta = _device_meta(values, device)

    # The scalars ride in a device array of the *output* dtype, so integer
    # operators keep integer semantics instead of going through a float.
    s_dtype = out_dt if out_dt.is_floating_point else torch.float32
    scalar_buf = torch.tensor([float(v) for v in vals], dtype=s_dtype).to(
        device, non_blocking=True
    )

    block = _pick_block(numels)
    num_tiles = max(1, triton.cdiv(max_numel, block))
    ncta = min(num_tiles, MAX_GRID_DIM)
    foreach_binary_scalar_kernel[(nt, ncta)](
        meta,
        scalar_buf,
        fn,
        nt,
        tl_dtype(in_dt),
        tl_dtype(out_dt),
        tl_dtype(s_dtype),
        alpha,
        block,
        ncta,
        num_warps=4,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def foreach_binary_scalar(
    self: Sequence[torch.Tensor],
    scalars: Any,
    fn: Callable,
    *,
    inplace: bool = False,
    alpha: float = 1,
    allowed_dtypes: Optional[frozenset] = None,
    dtype_error: Optional[str] = None,
    dtype_error_type: type = RuntimeError,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]] = None,
) -> List[torch.Tensor]:
    """``self[i] <op> alpha * scalar`` -- serves ``.Scalar`` / ``.ScalarList``.

    The per-tensor bucketing, promotion and inplace checks mirror the common
    executor; only the launch group is GCU400's own, so no f64 scalar kernel is
    ever compiled.  ``scalars`` is either one number (broadcast to every tensor)
    or one number per tensor.
    """
    tensors = check_tensor_list(self)
    values, _, _ = _resolve_scalars(tensors, scalars)
    _STATS.reset()

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for tensor, value in zip(tensors, values):
        if allowed_dtypes is not None and tensor.dtype not in allowed_dtypes:
            check_dtype_supported(
                tensor.dtype, allowed_dtypes, dtype_error, dtype_error_type
            )
        out_dtype = _promote_with_scalar(tensor.dtype, value)
        if out_dtype_fn is not None:
            out_dtype = out_dtype_fn(out_dtype)
        src = tensor if tensor.is_contiguous() else _dense(tensor)
        if inplace:
            if out_dtype != tensor.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(tensor.dtype).replace('torch.', '').capitalize()}"
                )
            if src is not tensor and has_internal_overlap(tensor):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src
            if dst is not tensor:
                writebacks.append((tensor, dst))
            results.append(tensor)
        else:
            dst = (
                torch.empty_like(src)
                if out_dtype is src.dtype
                else torch.empty_like(src, dtype=out_dtype)
            )
            results.append(dst)
        key = (tensor.device, src.dtype, out_dtype)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [], [])
        bucket[0].append(src)
        bucket[1].append(dst)
        bucket[2].append(value)

    for (device, in_dt, out_dt), (srcs, dsts, vals) in buckets.items():
        _launch_scalar_group(srcs, dsts, vals, fn, in_dt, out_dt, device, alpha)

    _finish_inplace(results, writebacks)
    return results
