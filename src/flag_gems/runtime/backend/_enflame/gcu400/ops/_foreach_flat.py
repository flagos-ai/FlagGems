# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 ``foreach`` 系列的一维 flat kernel 内部工具。

公共 ``foreach`` 执行器把 tensor 列表编码成 metadata table，并用二维
``grid=(block, tensor)`` 启动；GCU400 的 ``grid.y`` 硬件上限为 255，
大 shape 下会直接 ``OutOfResources``。这里提供两类替代路径：

* 逐 tensor 的一维固定 grid + 核内跨步 kernel（朴素迁移基线）；
* 把整个列表的地址/元素数打包进 int64 metadata，用一维 grid 一次 launch
  完成全部 tensor，``grid.x`` 按 ``(tensor, part)`` 线性展开，避免
  16 次独立 launch 的固定开销。

只有「浮点、连续、同 dtype、标准广播形态」的列表走快路径；其余情形
由调用方回退到公共执行器，保证 dtype policy / 异常语义完全不变。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

# Triton 3.6 的 tl tensor 没有 __pow__；复用公共实现同款 GCU libdevice powf。
_pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)

FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# 编译期分派的二元运算编码，避免为每个算子复制一份 kernel。
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3
OP_MIN = 4
OP_MAX = 5
OP_POW = 6

# 一元运算编码。
UNARY_EXP = 0
UNARY_CEIL = 1

# 第二操作数的形态编码。
MODE_LIST = 0
MODE_SCALAR = 1
MODE_SCALAR_LIST = 2
MODE_TENSOR = 3

# ``run_binary`` / ``run_unary`` 用它表示「本次输入不适合快路径」。
FALLBACK = object()

# host dtype → Triton dtype，用于把 metadata 里的裸地址还原成带类型的指针。
_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@libentry()
@triton.jit(do_not_specialize=["N"])
def binary_flat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    OP: tl.constexpr,
    HAS_B_TENSOR: tl.constexpr,
    B_BROADCAST: tl.constexpr,
    B_SCALAR: tl.constexpr,
    ALPHA: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """单个 tensor 的一维跨步计算；标量/广播操作数在编译期分派。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask)
        if HAS_B_TENSOR:
            if B_BROADCAST:
                # 0-d / 单元素操作数只读一次，避免整块重复加载。
                b = tl.load(b_ptr)
            else:
                b = tl.load(b_ptr + offs, mask=mask)
            if FP32_PROMOTE:
                b = b.to(tl.float32)
        elif FP32_PROMOTE:
            b = tl.full((), B_SCALAR, tl.float32)
        else:
            b = B_SCALAR
        if FP32_PROMOTE:
            # 与公共执行器一致：低精度先提升 fp32 计算，store 时隐式降回。
            a = a.to(tl.float32)
        if ALPHA != 1:
            b = b * ALPHA
        if OP == 0:
            result = a + b
        elif OP == 1:
            result = a - b
        elif OP == 2:
            result = a * b
        elif OP == 3:
            result = a / b
        elif OP == 4:
            result = tl.minimum(a, b)
        elif OP == 5:
            result = tl.maximum(a, b)
        else:
            # powf 需要 fp32 操作数，与公共 _pow_fn 的提升保持一致。
            result = _pow(a.to(tl.float32), b.to(tl.float32))
        tl.store(out_ptr + offs, result, mask=mask)


@libentry()
@triton.jit
def fused_binary_list_kernel(
    meta_ptr,
    NT: tl.constexpr,
    OP: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``.List``：一次 launch 处理整个配对 tensor 列表。

    ``meta_ptr`` 依次存放输入指针、输出指针、第二操作数指针、元素数
    （各 NT 项，int64）。grid.x 按 ``(tensor, part)`` 线性展开。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    b_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(A_DT))

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        if FP32_PROMOTE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        if OP == 0:
            result = a + b
        elif OP == 1:
            result = a - b
        elif OP == 2:
            result = a * b
        elif OP == 3:
            result = a / b
        elif OP == 4:
            result = tl.minimum(a, b)
        elif OP == 5:
            result = tl.maximum(a, b)
        else:
            # powf 需要 fp32 操作数，与公共 _pow_fn 的提升保持一致。
            result = _pow(a.to(tl.float32), b.to(tl.float32))
        tl.store(out_ptr + offs, result, mask=mask)


@libentry()
@triton.jit
def fused_binary_scalar_kernel(
    meta_ptr,
    scalar_ptr,
    NT: tl.constexpr,
    OP: tl.constexpr,
    A_DT: tl.constexpr,
    HAS_B_TENSOR: tl.constexpr,
    B_DT: tl.constexpr,
    B_SCALAR: tl.constexpr,
    SCALAR_PER_TENSOR: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``.Scalar`` / ``.ScalarList`` / ``.Tensor``：一次 launch 处理整个列表。

    ``HAS_B_TENSOR`` 对应 0-d tensor 广播；``SCALAR_PER_TENSOR`` 对应
    ScalarList 的逐 tensor 标量数组；其余情况用编译期常量 ``B_SCALAR``。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    b_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(B_DT))

    if SCALAR_PER_TENSOR:
        scalar = tl.load(scalar_ptr + tensor_id)
    else:
        scalar = tl.full((), B_SCALAR, A_DT)

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        a = tl.load(a_ptr + offs, mask=mask)
        if HAS_B_TENSOR:
            b = tl.load(b_ptr)
        else:
            b = scalar
        if FP32_PROMOTE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        if OP == 0:
            result = a + b
        elif OP == 1:
            result = a - b
        elif OP == 2:
            result = a * b
        elif OP == 3:
            result = a / b
        elif OP == 4:
            result = tl.minimum(a, b)
        elif OP == 5:
            result = tl.maximum(a, b)
        else:
            # powf 需要 fp32 操作数，与公共 _pow_fn 的提升保持一致。
            result = _pow(a.to(tl.float32), b.to(tl.float32))
        tl.store(out_ptr + offs, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def unary_flat_kernel(
    data_ptr,
    N,
    OP: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """单个 tensor 的原地一元运算，一维跨步写回。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(data_ptr + offs, mask=mask)
        # GCU 后端的 exp/ceil 只接受 fp32/fp64，低精度必须先提升；
        # store 时会隐式降回输出 tensor 的 dtype。
        value = value.to(tl.float32)
        if OP == 0:
            result = tl.exp(value)
        else:
            result = tl.ceil(value)
        tl.store(data_ptr + offs, result, mask=mask)


@libentry()
@triton.jit
def fused_unary_kernel(
    meta_ptr,
    NT: tl.constexpr,
    OP: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """一次 launch 对整个 tensor 列表做原地一元运算。"""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    data_raw = tl.load(meta_ptr + tensor_id)
    num_elems = tl.load(meta_ptr + NT + tensor_id)
    data_ptr = data_raw.to(tl.pointer_type(A_DT))

    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(part_id, tl.cdiv(num_elems, BLOCK), num_parts):
        offs = block_id * BLOCK + arange
        mask = offs < num_elems
        value = tl.load(data_ptr + offs, mask=mask)
        # GCU 后端的 exp/ceil 只接受 fp32/fp64，低精度必须先提升；
        # store 时会隐式降回输出 tensor 的 dtype。
        value = value.to(tl.float32)
        if OP == 0:
            result = tl.exp(value)
        else:
            result = tl.ceil(value)
        tl.store(data_ptr + offs, result, mask=mask)


def can_use_flat(self, operand, mode):
    """判断列表是否属于快路径可安全处理的形态。"""
    if not self:
        return False
    dtype = self[0].dtype
    if dtype not in FLOAT_DTYPES or dtype.is_complex:
        return False
    for tensor in self:
        if tensor.dtype != dtype or not tensor.is_contiguous() or tensor.numel() == 0:
            return False
    if mode is None:
        # 一元运算没有第二操作数，列表自身检查通过即可。
        return True
    if mode == MODE_LIST:
        if not isinstance(operand, (list, tuple)) or len(operand) != len(self):
            return False
        for index, other in enumerate(operand):
            if not isinstance(other, torch.Tensor) or other.dtype != dtype:
                return False
            if not other.is_contiguous():
                return False
            if other.numel() != self[index].numel() and other.numel() != 1:
                return False
    elif mode == MODE_SCALAR_LIST:
        if not isinstance(operand, (list, tuple)) or len(operand) != len(self):
            return False
        if not all(isinstance(value, (int, float)) for value in operand):
            return False
    elif mode == MODE_TENSOR:
        if not isinstance(operand, torch.Tensor) or operand.numel() != 1:
            return False
    elif not isinstance(operand, (int, float)):
        return False
    return True


def _operand_for(tensor, operand, mode, index):
    """把四种操作数形态归一成 (指针实参, 是否张量, 是否广播, 标量值)。"""
    if mode == MODE_LIST:
        other = operand[index]
        return other, 1, int(other.numel() == 1), 0.0
    if mode == MODE_TENSOR:
        return operand, 1, 1, 0.0
    if mode == MODE_SCALAR_LIST:
        return tensor, 0, 0, float(operand[index])
    return tensor, 0, 0, float(operand)


def _fused_grid(self, fixed_grid, block_size):
    """小列表只需 NT 个 program；大 shape 让每个 tensor 分到多个 program。"""
    num_tensors = len(self)
    blocks_per_tensor = triton.cdiv(max(t.numel() for t in self), block_size)
    return max(1, min(fixed_grid, num_tensors * blocks_per_tensor))


def _fused_meta(self, outputs, inplace, b_pointers):
    """构造四段 int64 metadata：输入/输出/第二操作数地址 + 各元素数。"""
    out_tensors = self if inplace else outputs
    pointers = (
        [tensor.data_ptr() for tensor in self]
        + [tensor.data_ptr() for tensor in out_tensors]
        + b_pointers
        + [tensor.numel() for tensor in self]
    )
    return torch.tensor(pointers, dtype=torch.int64, device=self[0].device)


def _run_fused(
    self,
    operand,
    op_code,
    mode,
    outputs,
    inplace,
    fixed_grid,
    block_size,
    num_warps,
    num_stages,
    use_device_context,
    fp32_promote,
):
    """按操作数形态选择融合 kernel，一次 launch 完成整个列表。"""
    num_tensors = len(self)
    dtype = self[0].dtype
    triton_dtype = _TL_DTYPES[dtype]
    grid_size = _fused_grid(self, fixed_grid, block_size)

    if mode == MODE_LIST:
        b_pointers = [other.data_ptr() for other in operand]
        scalar_tensor = self[0]  # 占位，不会在该 kernel 中读取
        kernel = fused_binary_list_kernel
        extra = ()
    elif mode == MODE_TENSOR:
        b_pointers = [operand.data_ptr()] * num_tensors
        scalar_tensor = self[0]
        kernel = fused_binary_scalar_kernel
        # 0-d tensor 常为 fp32，必须按它自己的 dtype 读取，不能沿用列表 dtype。
        b_dtype = _TL_DTYPES.get(operand.dtype, triton_dtype)
        extra = (1, b_dtype, 0.0, 0)
    elif mode == MODE_SCALAR_LIST:
        b_pointers = [0] * num_tensors
        scalar_tensor = torch.tensor(
            operand, dtype=torch.float32, device=self[0].device
        )
        kernel = fused_binary_scalar_kernel
        extra = (0, triton_dtype, 0.0, 1)
    else:
        b_pointers = [0] * num_tensors
        scalar_tensor = self[0]
        kernel = fused_binary_scalar_kernel
        extra = (0, triton_dtype, float(operand), 0)

    meta = _fused_meta(self, outputs, inplace, b_pointers)
    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        if mode == MODE_LIST:
            kernel[(grid_size,)](
                meta,
                num_tensors,
                op_code,
                triton_dtype,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            kernel[(grid_size,)](
                meta,
                scalar_tensor,
                num_tensors,
                op_code,
                triton_dtype,
                *extra,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return None if inplace else outputs


def run_binary(
    self,
    operand,
    op_code,
    mode,
    inplace,
    alpha=1.0,
    *,
    fixed_grid=48,
    block_size=8192,
    num_warps=4,
    num_stages=1,
    use_device_context=1,
    fp32_promote=1,
    fused=1,
):
    """执行二元 foreach；不适合快路径时返回 ``FALLBACK`` 哨兵。"""
    if alpha != 1 or not can_use_flat(self, operand, mode):
        return FALLBACK

    outputs = None if inplace else [torch.empty_like(tensor) for tensor in self]
    # ``.List`` 的第二操作数若需广播，不能与输入共用同一平坦索引。
    fused_ready = fused and (
        mode != MODE_LIST
        or all(other.numel() == tensor.numel() for tensor, other in zip(self, operand))
    )
    if fused_ready:
        return _run_fused(
            self,
            operand,
            op_code,
            mode,
            outputs,
            inplace,
            fixed_grid,
            block_size,
            num_warps,
            num_stages,
            use_device_context,
            fp32_promote,
        )

    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        for index, tensor in enumerate(self):
            num_elems = tensor.numel()
            out = tensor if inplace else outputs[index]
            b_arg, has_b_tensor, b_broadcast, b_scalar = _operand_for(
                tensor, operand, mode, index
            )
            grid = (min(triton.cdiv(num_elems, block_size), fixed_grid),)
            binary_flat_kernel[grid](
                tensor,
                b_arg,
                out,
                num_elems,
                op_code,
                has_b_tensor,
                b_broadcast,
                b_scalar,
                alpha,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return None if inplace else outputs


def _run_fused_unary(
    self,
    op_code,
    inplace,
    fixed_grid,
    block_size,
    num_warps,
    num_stages,
    use_device_context,
    fp32_promote,
):
    """一次 launch 对整个列表做一元运算。"""
    num_tensors = len(self)
    out_tensors = self if inplace else [torch.empty_like(t) for t in self]
    if not inplace:
        for source, target in zip(self, out_tensors):
            target.copy_(source)
    pointers = [t.data_ptr() for t in out_tensors] + [t.numel() for t in self]
    meta = torch.tensor(pointers, dtype=torch.int64, device=self[0].device)
    grid_size = _fused_grid(self, fixed_grid, block_size)
    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        fused_unary_kernel[(grid_size,)](
            meta,
            num_tensors,
            op_code,
            _TL_DTYPES[self[0].dtype],
            fp32_promote,
            block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return None if inplace else out_tensors


def run_unary(
    self,
    op_code,
    inplace,
    *,
    fixed_grid=48,
    block_size=8192,
    num_warps=4,
    num_stages=1,
    use_device_context=1,
    fp32_promote=1,
    fused=1,
):
    """执行一元 foreach；不适合快路径时返回 ``FALLBACK`` 哨兵。"""
    if not can_use_flat(self, None, None):
        return FALLBACK
    if fused:
        return _run_fused_unary(
            self,
            op_code,
            inplace,
            fixed_grid,
            block_size,
            num_warps,
            num_stages,
            use_device_context,
            fp32_promote,
        )

    outputs = None if inplace else [torch.empty_like(tensor) for tensor in self]
    device_context = (
        torch_device_fn.device(self[0].device) if use_device_context else nullcontext()
    )
    with device_context:
        for index, tensor in enumerate(self):
            num_elems = tensor.numel()
            out = tensor if inplace else outputs[index]
            if not inplace:
                # 非原地形式必须先复制输入，再对新 tensor 做原地计算。
                out.copy_(tensor)
            grid = (min(triton.cdiv(num_elems, block_size), fixed_grid),)
            unary_flat_kernel[grid](
                out,
                num_elems,
                op_code,
                fp32_promote,
                block_size,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return None if inplace else outputs
