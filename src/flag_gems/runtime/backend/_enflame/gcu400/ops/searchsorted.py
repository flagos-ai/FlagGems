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

import flag_gems
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_CUDA_BLOCK_SIZE = 256
# 整数序列保守回退通用向量 kernel：整数路径已达标（1.8x），不值得为窄整型
# 与 python int 标量实参的混合比较再承担一次未验证的 GCU 风险；块宽压小以
# 避免 1 个元素时的 lane 空转。
SCALAR_BLOCK_SIZE = 64
# 1D 序列上用 int32 索引替代 int64，降低地址计算开销。
GCU400_SCALAR_INT32_INDEX = True
# GCU400 tensor 路径同样用 int32 索引（i64 地址状态链软模拟开销大），numel 超 2^31
# 自动回退 i64 保正确；标量入口走 _searchsorted_scalar_kernel（全程标量二分，
# 无 lane 空转），块宽由 _select_launch_config 按 dtype/工作量选定。
GCU400_TENSOR_INT32_INDEX = True
_GCU_BLOCK_SIZE = 512
_GCU_NUM_WARPS = 4
_ASCEND_BLOCK_SIZE = 512
_SUPPORTED_INPUT_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@triton.jit
def _searchsorted_kernel(
    sorted_sequence,
    values,
    sorter,
    out,
    total_values,
    values_per_row,
    sequence_len,
    LOG_SEQUENCE_LEN: tl.constexpr,
    RIGHT: tl.constexpr,
    HAS_SORTER: tl.constexpr,
    IS_1D_SEQUENCE: tl.constexpr,
    USE_INT32_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_values
    values_in = tl.load(values + offsets, mask=mask, other=0)

    if IS_1D_SEQUENCE:
        if USE_INT32_INDEX:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        else:
            row_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    else:
        row_offsets = (offsets // values_per_row) * sequence_len
        if USE_INT32_INDEX:
            row_offsets = row_offsets.to(tl.int32)

    if USE_INT32_INDEX:
        low = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        low = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    high = low + sequence_len

    for _ in range(LOG_SEQUENCE_LEN):
        active = mask & (low < high)
        mid = low + (high - low) // 2
        sorted_offsets = row_offsets + mid
        if HAS_SORTER:
            sorted_index = tl.load(sorter + sorted_offsets, mask=active, other=0)
            if USE_INT32_INDEX:
                sorted_index = sorted_index.to(tl.int32)
            sorted_offsets = row_offsets + sorted_index

        mid_values = tl.load(sorted_sequence + sorted_offsets, mask=active, other=0)
        if RIGHT:
            go_left = values_in < mid_values
        else:
            go_left = values_in <= mid_values

        high = tl.where(active & go_left, mid, high)
        low = tl.where(active & ~go_left, mid + 1, low)

    tl.store(out + offsets, low, mask=mask)


@triton.jit
def _searchsorted_scalar_kernel(
    sorted_sequence,
    out,
    value,
    sequence_len,
    LOG_SEQUENCE_LEN: tl.constexpr,
    RIGHT: tl.constexpr,
    OUT_INT32: tl.constexpr,
):
    """单个标量值的二分查找：全程标量运算，无 arange / 无向量 lane 空转。

    指针按 dtype 宽度偏移由 Triton 自行完成；mid 会被钳到 [0, sequence_len-1]
    以免 low>=high 时越界，此时该次比较结果被 active 屏蔽。
    """
    low = tl.zeros((), dtype=tl.int32)
    high = low + sequence_len
    last = sequence_len - 1
    for _ in range(LOG_SEQUENCE_LEN):
        active = low < high
        mid = low + (high - low) // 2
        probe = tl.minimum(mid, last)
        mid_value = tl.load(sorted_sequence + probe)
        if RIGHT:
            go_left = value < mid_value
        else:
            go_left = value <= mid_value
        high = tl.where(active & go_left, mid, high)
        low = tl.where(active & ~go_left, mid + 1, low)
    if OUT_INT32:
        tl.store(out, low.to(tl.int32))
    else:
        tl.store(out, low.to(tl.int64))


def _select_launch_config(dtype, numel, is_1d_sequence):
    """按元素宽度与工作量选 (BLOCK_SIZE, num_warps)。

    该 kernel 的瓶颈是 sorted_sequence 上的非连续 gather（每步一次）：
    - 2 字节浮点（fp16/bf16）单次 gather 事务只搬一半字节，放大块能摊薄
      每步的固定开销，实测大 case 2048/w4 比 512/w4 快 1.6x；
    - 4/8 字节元素放大块反而更慢（gather 事务数翻倍、寄存器压力上升），
      保持 512/w4；1D 序列上 2 字节浮点用 512/w2 最快。
    以上来自 gcu400 kernel-only sweep（256/512/1024/2048 × w2/w4）。
    """
    if dtype in (torch.float16, torch.bfloat16):
        if numel >= 65536:
            return 2048, 4
        if not is_1d_sequence:
            return 1024, 4
        return 512, 2
    return _GCU_BLOCK_SIZE, _GCU_NUM_WARPS


def _normalize_right(right: bool, side: str | None) -> bool:
    if side is None:
        return bool(right)
    if side == "left":
        if right:
            raise RuntimeError(
                "torch.searchsorted(): side and right can't be set to opposites, "
                "got side of left while right was True"
            )
        return False
    if side == "right":
        return True
    raise RuntimeError(
        f"torch.searchsorted(): side can only be 'left' or 'right' but got {side}"
    )


def _check_dtype(tensor: torch.Tensor, name: str):
    if tensor.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"searchsorted is not implemented for {name} dtype {tensor.dtype}"
        )


def _check_tensor_values_shape(sorted_sequence: torch.Tensor, values: torch.Tensor):
    if sorted_sequence.dim() == 0:
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match"
        )
    if sorted_sequence.dim() == 1:
        return
    if values.dim() != sorted_sequence.dim() or (
        tuple(values.shape[:-1]) != tuple(sorted_sequence.shape[:-1])
    ):
        raise RuntimeError(
            "torch.searchsorted(): boundaries tensor should be 1 dimension or "
            "the first N-1 dimensions of boundaries tensor and input value tensor "
            "must match, but we got boundaries tensor "
            f"{list(sorted_sequence.shape)} and input value tensor {list(values.shape)}"
        )


def _check_scalar_values_shape(sorted_sequence: torch.Tensor):
    if sorted_sequence.dim() != 1:
        raise RuntimeError(
            "torch.searchsorted(): input value can be a scalar only when boundaries "
            "tensor dimension is 1, but we got boundaries tensor "
            f"dim({sorted_sequence.dim()}) and input value's dim(0) numel(1)"
        )


def _check_sorter(sorted_sequence: torch.Tensor, sorter: torch.Tensor | None):
    if sorter is None:
        return
    if tuple(sorter.shape) != tuple(sorted_sequence.shape):
        raise RuntimeError(
            "torch.searchsorted(): boundary and sorter must have the same size, "
            f"but got boundary tensor {list(sorted_sequence.shape)}"
            f"and got sorter tensor {list(sorter.shape)}"
        )
    if sorter.dtype != torch.int64:
        raise RuntimeError(
            "torch.searchsorted(): sorter must be a tensor of long dtype but got "
            f"dtype {sorter.dtype}"
        )
    if sorter.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorter and boundary tensors must be on the same device"
        )
    sequence_len = sorted_sequence.shape[-1]
    if sorter.numel() != 0 and (
        torch.any(sorter < 0).item() or torch.any(sorter >= sequence_len).item()
    ):
        raise RuntimeError("torch.searchsorted(): sorter index out of range")


def _prepare_out(
    values: torch.Tensor,
    out_int32: bool,
    out: torch.Tensor | None,
):
    out_dtype = torch.int32 if out_int32 else torch.int64
    if out is None:
        return torch.empty(values.shape, dtype=out_dtype, device=values.device)
    if out.dtype != out_dtype:
        raise RuntimeError(
            "torch.searchsorted(): output tensor's dtype is wrong, it can only be "
            "Int(int32) or Long(int64) depending on whether out_int32 flag is True"
        )
    if out.device != values.device:
        raise RuntimeError(
            "torch.searchsorted(): output tensor must be on the same device as input"
        )
    if tuple(out.shape) != tuple(values.shape):
        out.resize_(values.shape)
    return out


def _prepare_scalar_out(
    sorted_sequence: torch.Tensor,
    out_int32: bool,
    out: torch.Tensor | None,
):
    """标量入口的 out 准备：0 维输出，不需要 values 张量。"""
    out_dtype = torch.int32 if out_int32 else torch.int64
    if out is None:
        return torch.empty((), dtype=out_dtype, device=sorted_sequence.device)
    if out.dtype != out_dtype:
        raise RuntimeError(
            "torch.searchsorted(): output tensor's dtype is wrong, it can only be "
            "Int(int32) or Long(int64) depending on whether out_int32 flag is True"
        )
    if out.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): output tensor must be on the same device as input"
        )
    if tuple(out.shape) != ():
        out.resize_(())
    return out


def _searchsorted_scalar_impl(
    sorted_sequence: torch.Tensor,
    value,
    *,
    out_int32: bool,
    right: bool,
    side: str | None,
    sorter: torch.Tensor | None,
    out: torch.Tensor | None = None,
):
    """python 标量入口。

    原实现先 `torch.scalar_tensor(value, device=...)` 造一个 0 维设备张量（每次
    调用一次 H2D 拷贝），再跑向量 kernel（多 lane 空转）。改为把标量直接作为
    kernel 实参、全程标量二分，省掉 H2D 与 lane 空转；带 sorter 或 ascend
    仍走通用张量路径。
    """
    right = _normalize_right(right, side)
    _check_dtype(sorted_sequence, "sorted_sequence")
    _check_scalar_values_shape(sorted_sequence)
    _check_sorter(sorted_sequence, sorter)
    if (
        sorter is not None
        or runtime_device.vendor_name == "ascend"
        or not sorted_sequence.dtype.is_floating_point
    ):
        values = torch.tensor(value, device=sorted_sequence.device)
        return _searchsorted_impl(
            sorted_sequence,
            values,
            out_int32=out_int32,
            right=right,
            side=None,
            sorter=sorter,
            out=out,
        )

    out = _prepare_scalar_out(sorted_sequence, out_int32, out)
    if sorted_sequence.shape[-1] == 0:
        out.zero_()
        return out
    sorted_sequence_contiguous = sorted_sequence.contiguous()
    kernel_out = out if out.is_contiguous() else torch.empty_like(out)
    with torch_device_fn.device(sorted_sequence.device):
        _searchsorted_scalar_kernel[(1,)](
            sorted_sequence_contiguous,
            kernel_out,
            value,
            sorted_sequence.shape[-1],
            LOG_SEQUENCE_LEN=sorted_sequence.shape[-1].bit_length(),
            RIGHT=right,
            OUT_INT32=kernel_out.dtype == torch.int32,
            num_warps=1,
        )
    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def _searchsorted_impl(
    sorted_sequence: torch.Tensor,
    values: torch.Tensor,
    *,
    out_int32: bool,
    right: bool,
    side: str | None,
    sorter: torch.Tensor | None,
    out: torch.Tensor | None = None,
):
    right = _normalize_right(right, side)
    _check_dtype(sorted_sequence, "sorted_sequence")
    _check_dtype(values, "values")
    _check_tensor_values_shape(sorted_sequence, values)
    _check_sorter(sorted_sequence, sorter)
    if values.device != sorted_sequence.device:
        raise RuntimeError(
            "torch.searchsorted(): sorted_sequence and values must be on the same device"
        )

    out = _prepare_out(values, out_int32, out)
    if values.numel() == 0:
        return out
    if sorted_sequence.shape[-1] == 0:
        out.zero_()
        return out

    sorted_sequence_contiguous = sorted_sequence.contiguous()
    values_contiguous = values.contiguous()
    sorter_contiguous = sorter.contiguous() if sorter is not None else None
    is_ascend = runtime_device.vendor_name == "ascend"
    if sorter_contiguous is not None and is_ascend:
        sorted_sequence_contiguous = flag_gems.gather(
            sorted_sequence_contiguous, -1, sorter_contiguous
        )
        sorter_contiguous = None
    kernel_out = (
        out
        if out.is_contiguous()
        else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    )

    sequence_len = sorted_sequence.shape[-1]
    values_per_row = values.shape[-1] if sorted_sequence.dim() != 1 else values.numel()
    is_1d_sequence = sorted_sequence.dim() == 1
    scalar_values = values.numel() == 1
    if is_ascend:
        block_size = (
            _ASCEND_BLOCK_SIZE
            if sorted_sequence.dtype.is_floating_point
            else _CUDA_BLOCK_SIZE
        )
        num_warps = _GCU_NUM_WARPS
    else:
        block_size, num_warps = _select_launch_config(
            sorted_sequence.dtype, values.numel(), is_1d_sequence
        )
        if scalar_values:
            # 单元素仍走向量 kernel 时（整数 dtype 回退路径），压小块宽
            block_size, num_warps = SCALAR_BLOCK_SIZE, _GCU_NUM_WARPS
    int32_index_safe = (
        values.numel() < torch.iinfo(torch.int32).max
        and sorted_sequence.numel() < torch.iinfo(torch.int32).max
    )
    use_int32_index = (
        (is_ascend or GCU400_TENSOR_INT32_INDEX) and int32_index_safe
    ) or (
        scalar_values
        and GCU400_SCALAR_INT32_INDEX
        and sequence_len <= torch.iinfo(torch.int32).max
    )

    with torch_device_fn.device(sorted_sequence.device):
        grid = (triton.cdiv(values.numel(), block_size),)
        _searchsorted_kernel[grid](
            sorted_sequence_contiguous,
            values_contiguous,
            (
                sorter_contiguous
                if sorter_contiguous is not None
                else sorted_sequence_contiguous
            ),
            kernel_out,
            values.numel(),
            values_per_row,
            sequence_len,
            LOG_SEQUENCE_LEN=sequence_len.bit_length(),
            RIGHT=right,
            HAS_SORTER=sorter_contiguous is not None,
            IS_1D_SEQUENCE=is_1d_sequence,
            USE_INT32_INDEX=use_int32_index,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def searchsorted(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS SEARCHSORTED")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS SEARCHSORTED_OUT")
    return _searchsorted_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )


# ---------------- native 参考桥（torch_gcu topsaten searchsorted 破损失域） -----------
# 探针实证（2026-11-12，L600，TORCH_GCU_ENABLE_INT64_AND_UINT64=1，逐 case 子进程）：
#   - side="right"（right=False）→ gcu_utils.cpp:74 Check failed: 0，
#     TOPSATEN_STATUS_BAD_PARAM 硬 abort；right=True 单独可用、side="left" 可用；
#   - 空 boundaries（sorted_sequence.shape[-1] == 0）→ BAD_PARAM 硬 abort（全 dtype）；
#   - python float 标量（.Scalar / .Scalar_out 重载）→ NOT_SUPPORT（F64 标量）硬
#     abort；int 标量可用；
#   - values 含 NaN → 错值（NaN 视作小于全部→0；CPU/FlagGems 语义为排尾）。
# 修复口径 = 参考路径回退 CPU（服务须知第 9 条）：仅上述破损失域改走 CPU 原生
# 实现，正常张量域原样透传 native（bench latency_base 不受影响）；FlagGems 算子
# 路径（flag_gems.searchsorted*）零改动。use_gems 激活期间整体透传——FlagGems
# kernel 已在 dispatcher 层接管且覆盖全部破损失域。
_NAN_CHECK_NUMEL_LIMIT = 1024
_TORCH_SEARCHSORTED_NATIVE = None
_SCALAR_OUT_BRIDGE_LIBRARY = None


def _gems_runtime_active():
    try:
        import flag_gems

        return getattr(flag_gems, "current_work_registrar", None) is not None
    except Exception:
        return False


def _has_gcu_tensor(*tensors):
    return any(isinstance(t, torch.Tensor) and t.device.type == "gcu" for t in tensors)


def _searchsorted_cpu_reference(
    sorted_sequence, self, *, out_int32, right, sorter, out
):
    seq_cpu = (
        sorted_sequence.cpu()
        if isinstance(sorted_sequence, torch.Tensor)
        else sorted_sequence
    )
    values_cpu = self.cpu() if isinstance(self, torch.Tensor) else self
    sorter_cpu = sorter.cpu() if isinstance(sorter, torch.Tensor) else sorter
    if out is None:
        result = _TORCH_SEARCHSORTED_NATIVE(
            seq_cpu,
            values_cpu,
            out_int32=out_int32,
            right=right,
            sorter=sorter_cpu,
        )
        target = self if isinstance(self, torch.Tensor) else sorted_sequence
        if isinstance(target, torch.Tensor):
            result = result.to(target.device)
        return result
    out_cpu = (
        out if out.device.type == "cpu" else torch.empty(out.shape, dtype=out.dtype)
    )
    result = _TORCH_SEARCHSORTED_NATIVE(
        seq_cpu,
        values_cpu,
        out_int32=out_int32,
        right=right,
        sorter=sorter_cpu,
        out=out_cpu,
    )
    if out_cpu is not out:
        if tuple(out.shape) != tuple(result.shape):
            out.resize_(result.shape)
        out.copy_(result)
    return out


def _searchsorted_reference_bridge(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out=None,
):
    if not _gems_runtime_active() and _has_gcu_tensor(
        sorted_sequence, self, sorter, out
    ):
        # side 规范化：side 只在 python 层改写为 right（native 对 side="right" 是
        # 硬 abort，对 side/right 冲突同样是硬 abort 而非 RuntimeError）。
        if side is not None:
            right = _normalize_right(right, side)
            side = None
        need_cpu_reference = False
        if isinstance(self, (int, float)):
            # python 标量（含 int）：native 对 int16 等窄整型序列 + int 标量同样
            # 硬 abort（test_searchsorted_scalar[int16-...] 实测栈落在本函数末行），
            # 因此 int/float 标量统一走 CPU 参考。
            need_cpu_reference = True
        elif isinstance(sorted_sequence, torch.Tensor):
            if sorted_sequence.dim() > 0 and sorted_sequence.shape[-1] == 0:
                need_cpu_reference = True
            elif isinstance(self, torch.Tensor) and self.numel() == 0:
                # 空 values（如 values.shape=(2,0)）同样落在 native 破损失域：
                # test_searchsorted_tensor[empty_values-*] 直接 TOPSATEN 硬 abort。
                need_cpu_reference = True
            elif (
                isinstance(self, torch.Tensor)
                and self.dtype.is_floating_point
                and self.numel() <= _NAN_CHECK_NUMEL_LIMIT
                and bool(flag_gems.isnan(self).any())
            ):
                need_cpu_reference = True
        if need_cpu_reference:
            return _searchsorted_cpu_reference(
                sorted_sequence,
                self,
                out_int32=out_int32,
                right=right,
                sorter=sorter,
                out=out,
            )
    return _TORCH_SEARCHSORTED_NATIVE(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )


def _searchsorted_tensor_out_gcu_impl(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out=None,
):
    # 本槽位（searchsorted.Tensor_out@PrivateUse1）修前对任意 out 布局 100% 硬
    # abort（TOPSATEN_STATUS_BAD_PARAM，见 log_ss_tensor_out.log：连 contiguous
    # out 也 abort），因此参考实现不可用、tests/test_searchsorted.py 的
    # test_searchsorted_tensor_out 无法通过。修法与 Scalar_out 槽位一致：
    # use_gems 激活时委托 FlagGems kernel（保住 Triton 主路径），否则回退 CPU 参考。
    if _gems_runtime_active():
        return searchsorted_out(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
            out=out,
        )
    right = _normalize_right(right, side)
    return _searchsorted_cpu_reference(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        sorter=sorter,
        out=out,
    )


def _searchsorted_scalar_out_gcu_impl(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out=None,
):
    # 本槽位（searchsorted.Scalar_out@PrivateUse1）修前 native 对 float 标量
    # 100% 硬 abort（int 标量可用但一并走 CPU 保语义一致，零回退面）。
    if _gems_runtime_active():
        # use_gems 下本应由 FlagGems 注册本槽位（被本桥占位而跳过）；直接委托
        # vendor 实现，保住 kernel 路径。
        return searchsorted_scalar_out(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
            out=out,
        )
    right = _normalize_right(right, side)
    return _searchsorted_cpu_reference(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        sorter=sorter,
        out=out,
    )


def _install_searchsorted_reference_bridge():
    """幂等安装（后端加载器会以两个模块名各 import 一次本文件，锚定 torch 属性）。"""
    global _TORCH_SEARCHSORTED_NATIVE, _SCALAR_OUT_BRIDGE_LIBRARY
    if getattr(torch, "_flaggems_gcu_searchsorted_bridge", None) is not None:
        return
    _TORCH_SEARCHSORTED_NATIVE = torch.searchsorted
    torch.searchsorted = _searchsorted_reference_bridge
    setattr(torch, "_flaggems_gcu_searchsorted_bridge", _searchsorted_reference_bridge)
    library = torch.library.Library("aten", "IMPL")
    try:
        library.impl(
            "searchsorted.Scalar_out",
            _searchsorted_scalar_out_gcu_impl,
            "PrivateUse1",
            allow_override=True,
        )
        _SCALAR_OUT_BRIDGE_LIBRARY = library
    except Exception as err:
        logger.warning(
            "searchsorted reference bridge: Scalar_out@PrivateUse1 not "
            "registered: %s",
            err,
        )
    try:
        library.impl(
            "searchsorted.Tensor_out",
            _searchsorted_tensor_out_gcu_impl,
            "PrivateUse1",
            allow_override=True,
        )
        _SCALAR_OUT_BRIDGE_LIBRARY = library
    except Exception as err:
        logger.warning(
            "searchsorted reference bridge: Tensor_out@PrivateUse1 not "
            "registered: %s",
            err,
        )


_install_searchsorted_reference_bridge()


def searchsorted_scalar(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    logger.debug("GEMS SEARCHSORTED_SCALAR")
    return _searchsorted_scalar_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )


def searchsorted_scalar_out(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
    out,
):
    logger.debug("GEMS SEARCHSORTED_SCALAR_OUT")
    return _searchsorted_scalar_impl(
        sorted_sequence,
        self,
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
        out=out,
    )
