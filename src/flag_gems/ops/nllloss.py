import logging
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    assert tgt >= 0 and tgt < C, "Invalid target value"
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * wgt_tgt * -1

    # none
    if reduction == 0:
        tl.store(out_ptr + offsets_n, out, mask=mask_n)
    # mean
    elif reduction == 1:
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")  # output
        tl.atomic_add(out_ptr + 1, total_wgt, sem="relaxed")  # weight
        tl.atomic_add(out_ptr + 2, 1, sem="release")  # counter
        counter = tl.load(out_ptr + 2)
        if counter == tl.num_programs(0):
            total_out = tl.load(out_ptr)
            total_wgt = tl.load(out_ptr + 1)
            tl.store(out_ptr + 3, total_out / total_wgt)
    # sum
    else:
        total_out = tl.sum(out)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)
    if reduction == 1:
        total_w = tl.load(total_weight).to(tl.float32)
    else:
        total_w = 1

    inp_grad = tl.where(ignore_mask, -1 * out_grad * wgt_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
    inp_ptr, tgt_ptr, wgt_ptr, out_ptr,
    ignore_index,
    N, C, H, W,
    # 新增: Strides 参数
    stride_inp_n, stride_inp_c, stride_inp_h, stride_inp_w,
    stride_tgt_n, stride_tgt_h, stride_tgt_w,
    reduction: tl.constexpr = 1,
    BLOCK_SIZE: tl.constexpr = 128,
):
    # 处理 N*H*W 个元素
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    n_elements = N * H * W
    mask = offsets < n_elements

    # --- 坐标反解: Linear Index -> (n, h, w) ---
    _w = offsets % W
    _h = (offsets // W) % H
    _n = offsets // (W * H)

    # --- Target 读取 (使用 Stride) ---
    tgt_offset = _n * stride_tgt_n + _h * stride_tgt_h + _w * stride_tgt_w
    tgt = tl.load(tgt_ptr + tgt_offset, mask=mask, other=0)
    
    ignore_mask = (tgt != ignore_index) & mask

    # --- Weight 读取 ---
    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    # --- Input 读取 (使用 Stride) ---
    # Input Offset = n*s_n + class*s_c + h*s_h + w*s_w
    inp_offset = _n * stride_inp_n + tgt * stride_inp_c + _h * stride_inp_h + _w * stride_inp_w
    inp_tgt = tl.load(inp_ptr + inp_offset, mask=ignore_mask, other=0).to(tl.float32)
    
    out = inp_tgt * wgt_tgt * -1

    # --- 输出 ---
    if reduction == 0:
        # Output 是新创建的连续 Tensor，直接用 linear offset
        tl.store(out_ptr + offsets, out, mask=mask)
    elif reduction == 1: # Mean
        total_out = tl.sum(out)
        total_wgt = tl.sum(wgt_tgt)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")
        tl.atomic_add(out_ptr + 1, total_wgt, sem="relaxed")
        tl.atomic_add(out_ptr + 2, 1, sem="release")
        
        counter = tl.load(out_ptr + 2)
        if counter == tl.num_programs(0):
            total_out = tl.load(out_ptr)
            total_wgt = tl.load(out_ptr + 1)
            if total_wgt == 0:
                tl.store(out_ptr + 3, 0.0)
            else:
                tl.store(out_ptr + 3, total_out / total_wgt)
    else: # Sum
        total_out = tl.sum(out)
        tl.atomic_add(out_ptr, total_out, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr, tgt_ptr, wgt_ptr, inp_grad_ptr,
    ignore_index, total_weight,
    N, C, H, W,
    # 新增: 各种 Stride 参数
    stride_grad_inp_n, stride_grad_inp_c, stride_grad_inp_h, stride_grad_inp_w,
    stride_tgt_n, stride_tgt_h, stride_tgt_w,
    stride_grad_out_n, stride_grad_out_h, stride_grad_out_w,
    reduction: tl.constexpr = 1,
    BLOCK_SIZE: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * H * W

    # 坐标反解
    _w = offsets % W
    _h = (offsets // W) % H
    _n = offsets // (W * H)

    # 读取 Target
    tgt_offset = _n * stride_tgt_n + _h * stride_tgt_h + _w * stride_tgt_w
    tgt = tl.load(tgt_ptr + tgt_offset, mask=mask, other=0)
    
    ignore_mask = (tgt != ignore_index) & mask

    # 读取 Weight
    if wgt_ptr is None:
        wgt_tgt = ignore_mask.to(tl.float32)
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    # 读取 Grad Output
    if reduction == 0:
        grad_out_off = _n * stride_grad_out_n + _h * stride_grad_out_h + _w * stride_grad_out_w
        out_grad = tl.load(out_grad_ptr + grad_out_off, mask=mask, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    total_w = tl.load(total_weight).to(tl.float32) if reduction == 1 else 1.0

    grad_val = -1 * out_grad * wgt_tgt
    if reduction == 1:
        grad_val = grad_val / total_w

    # 写入 Input Grad (利用 Stride)
    inp_grad_off = _n * stride_grad_inp_n + tgt * stride_grad_inp_c + _h * stride_grad_inp_h + _w * stride_grad_inp_w
    
    # 仅当 Mask 有效且该位置是 Target Class 时写入
    store_mask = mask & (tgt != ignore_index)
    tl.store(inp_grad_ptr + inp_grad_off, grad_val, mask=store_mask)


# Negative Log Likelihood Loss (NLLLoss)
#
# This loss function is used for training classification problems with C classes.
#
# Parameters:
# - input (Tensor):
#   - Expected to contain log-probabilities for each class.
#   - Shape can be either:
#     - (minibatch, C) for standard classification tasks.
#     - (minibatch, C, d1, d2, ..., dK) for K-dimensional inputs (e.g., per-pixel loss for 2D images).
#
# - target (Tensor):
#   - Should contain class indices in the range [0, C-1].
#   - If ignore_index is specified, this index can be outside the class range
#       and will be ignored in the loss computation.
#
# - weight (1D Tensor, optional):
#   - Assigns weight to each class, useful for unbalanced datasets.
#
# Reduction modes:
# - 'none': returns per-sample loss (shape: (N,)).
# - 'mean' (default): computes the mean of the weighted losses.
# - 'sum': computes the sum of the weighted losses.
#
# Mathematical description:
# - Unreduced loss:
#   l_n = -w_y_n * x_n, where w_c = weight[c] * 1{c != ignore_index}.
# - Reduced loss (depending on the specified reduction mode):
#   - mean: ℓ(x, y) = (1/N) * Σ(w_y_n * l_n)
#   - sum: ℓ(x, y) = Σ(l_n)


# 1d & 2d tensor
def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS NLL Loss FWD")
    assert self.ndim <= 2, "Invalid input ndim"
    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 0:
        out = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = torch.zeros(
            [
                4,
            ],
            dtype=torch.float32,
            device=self.device,
        )
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_forward_kernel[grid](
            self,
            target,
            weight,
            out,
            ignore_index,
            N,
            C,
            reduction,
        )

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 0:
        output = out
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = out.to(self.dtype)
        output = out[3]
        total_weight = out[1]
    else:
        output = out.to(self.dtype)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    return output, total_weight


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logger.debug("GEMS NLL Loss BWD")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch_device_fn.device(self.device):
        nll_loss_backward_kernel[grid](
            grad_output,
            target,
            weight,
            grad_input,
            ignore_index,
            total_weight,
            N,
            C,
            reduction,
        )

    return grad_input


# 3d+ tensor
def nll_loss2d_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logger.debug("GEMS NLL Loss2d FWD")
    assert self.ndim == 4
    N, C, H, W = self.shape
    assert list(target.shape) == [N, H, W]

    # 不再强制 self.contiguous() 和 target.contiguous()
    # Weight 通常很小，保持 contiguous 以简化逻辑
    weight = None if weight is None else weight.contiguous()

    if reduction == 0:
        out = torch.empty((N, H, W), dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = torch.zeros([4], dtype=torch.float32, device=self.device)
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)

    # 获取 Strides
    s_n, s_c, s_h, s_w = self.stride()
    t_s_n, t_s_h, t_s_w = target.stride()

    grid = lambda meta: (triton.cdiv(N * H * W, meta["BLOCK_SIZE"]),)
    
    with torch_device_fn.device(self.device):
        nll_loss2d_forward_kernel[grid](
            self, target, weight, out,
            ignore_index,
            N, C, H, W,
            # 传入 Strides
            s_n, s_c, s_h, s_w,
            t_s_n, t_s_h, t_s_w,
            reduction
        )

    if reduction == 0:
        return out, torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction == 1:
        out = out.to(self.dtype)
        return out[3], out[1]
    else:
        return out.to(self.dtype), torch.empty([], dtype=self.dtype, device=self.device)


def nll_loss2d_backward(grad_output, self, target, weight=None, reduction=1, ignore_index=-100, total_weight=None):
    logger.debug("GEMS NLL Loss2d BWD")
    N, C, H, W = self.shape

    # Input Grad 虽然是新创建的，但也获取 Stride 以保持通用性
    grad_input = torch.zeros_like(self)
    gi_s_n, gi_s_c, gi_s_h, gi_s_w = grad_input.stride()
    
    # Target Strides
    t_s_n, t_s_h, t_s_w = target.stride()
    
    # Grad Output Strides
    if reduction == 0:
        go_s_n, go_s_h, go_s_w = grad_output.stride()
    else:
        go_s_n, go_s_h, go_s_w = 0, 0, 0
        
    weight = None if weight is None else weight.contiguous()

    grid = lambda meta: (triton.cdiv(N * H * W, meta["BLOCK_SIZE"]),)
    
    with torch_device_fn.device(self.device):
        nll_loss2d_backward_kernel[grid](
            grad_output, target, weight, grad_input,
            ignore_index, total_weight,
            N, C, H, W,
            # 传入所有 Strides
            gi_s_n, gi_s_c, gi_s_h, gi_s_w,
            t_s_n, t_s_h, t_s_w,
            go_s_n, go_s_h, go_s_w,
            reduction
        )

    return grad_input