import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from .sum import sum


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index"])
def celoss_indice_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index)

    w_ptrs = w_ptr + tgt
    w_tgt = tl.load(w_ptrs, mask=tgt_mask, other=0).to(tl.float32)
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d
    tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask and ignore_mask)

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[None, :])
    final_sum = tl.log(tl.sum(tmp_sum, axis=0))

    inp_tgt_ptrs = inp_ptr + pid_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=-float("inf")).to(tl.float32)

    out = (final_sum + final_max - inp_tgt) * w_tgt
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask and ignore_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["label_smoothing"])
def celoss_probability_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    label_smoothing,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.log(tl.sum(tmp_sum, axis=0))[None, :]

    _sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        tgt_ptrs = tgt_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        w_ptrs = w_ptr + offset_c
        w_mask = offset_c < C
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)
        tgt = tl.load(tgt_ptrs, mask, other=1).to(tl.float32)
        tgt = tgt * (1.0 - label_smoothing) + label_smoothing / C
        w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)[:, None]
        log = final_sum + final_max - inp
        _sum += w * log * tgt

    out = tl.sum(_sum, axis=0)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=offset_d < D)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing"])
def celoss_indice_smooth_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    label_smoothing,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index)

    w_tgt = tl.load(w_ptr + tgt, mask=tgt_mask, other=0).to(tl.float32)
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d
    tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask and ignore_mask)

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.log(tl.sum(tmp_sum, axis=0))[None, :]

    _sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        offset = offset_c[:, None] * D + offset_d[None, :]
        inp_ptrs = inp_ptr + pid_n * C * D + offset
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        w_ptrs = w_ptr + offset_c
        w = tl.load(w_ptrs, offset_c < C, other=0).to(tl.float32)

        smooth = tl.full([BLOCK_C, BLOCK_D], label_smoothing / C, dtype=tl.float32)
        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            smooth,
        )

        log = final_sum + final_max - inp
        _sum += log * smooth * w[:, None]

    out = tl.sum(_sum, axis=0)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask and ignore_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "mean_num"])
def celoss_indice_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    mean_num,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)[None, :]
    w_ptrs = w_ptr + tgt
    w_tgt = tl.load(w_ptrs, mask=tgt_mask, other=0).to(tl.float32)[None, :]

    ignore_mask = (tgt != ignore_index)[None, :]

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=0)[None, :]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        minus_one = offset_c[:, None] == tgt[None, :]
        inp_grad = (
            (tl.exp(inp - final_max) / final_sum - minus_one)
            * w_tgt
            * out_grad
            * mean_num
        )
        inp_grad_ptrs = (
            inp_grad_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        )
        tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask and ignore_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["label_smoothing", "mean_num"])
def celoss_probability_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    label_smoothing,
    mean_num,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=offset_d < D, other=0).to(tl.float32)[
        None, :
    ]

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    w_tgt_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp = tl.load(inp_ptrs, mask, other=-float("inf")).to(tl.float32)

        tgt_ptrs = tgt_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_ptrs = w_ptr + offset_c
        w_mask = offset_c < C
        w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)[:, None]

        w_tgt_sum += tgt * w

        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=0)[None, :]
    w_tgt_sum = tl.sum(w_tgt_sum, axis=0)[None, :]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        offset = pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_ptrs = inp_ptr + offset
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        tgt_ptrs = tgt_ptr + offset
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_ptrs = w_ptr + offset_c
        w_mask = offset_c < C
        w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)[:, None]

        grad = w_tgt_sum / final_sum * tl.exp(inp - final_max) - w * tgt
        inp_grad = grad * out_grad * mean_num

        inp_grad_ptrs = inp_grad_ptr + offset
        tl.store(inp_grad_ptrs, inp_grad, mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing", "mean_num"])
def celoss_indice_smooth_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    label_smoothing,
    mean_num,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)[None, :]

    ignore_mask = (tgt != ignore_index)[None, :]

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    w_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)

        w_ptrs = w_ptr + offset_c
        w_mask = offset_c < C
        w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        smooth = tl.full([BLOCK_C, BLOCK_D], label_smoothing / C, dtype=tl.float32)
        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            smooth,
        )

        w_sum += smooth * w[:, None]

        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=0)[None, :]
    w_sum = tl.sum(w_sum, axis=0)[None, :]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)

        w_ptrs = w_ptr + offset_c
        w_mask = offset_c < C
        w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        smooth = tl.full([BLOCK_C, BLOCK_D], label_smoothing / C, dtype=tl.float32)
        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            smooth,
        )

        grad = w_sum / final_sum * tl.exp(inp - final_max) - smooth * w[:, None]
        inp_grad = grad * out_grad * mean_num
        inp_grad_ptrs = (
            inp_grad_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        )
        tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask and ignore_mask)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logging.debug("GEMS CrossEntropyLoss")
        # label_smoothing not supported

        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        if weight is None:
            weight = torch.ones(
                [
                    C,
                ],
                dtype=inp.dtype,
                device=inp.device,
            )

        inp = inp.contiguous()
        tgt = target.contiguous()
        weight = weight.contiguous()
        out = torch.zeros(shape, dtype=torch.float32, device=inp.device)
        grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))

        if tgt.ndim == dim:
            # target probabilities
            with torch.cuda.device(inp.device):
                celoss_probability_kernel[grid](
                    inp, tgt, weight, out, label_smoothing, N, C, D
                )
        elif label_smoothing == 0:
            # target indices
            w_tgt = torch.zeros(shape, dtype=torch.float32, device=inp.device)
            with torch.cuda.device(inp.device):
                celoss_indice_kernel[grid](
                    inp, tgt, weight, out, w_tgt, ignore_index, N, C, D
                )
        else:
            w_tgt = torch.zeros(shape, dtype=torch.float32, device=inp.device)
            with torch.cuda.device(inp.device):
                celoss_indice_smooth_kernel[grid](
                    inp, tgt, weight, out, w_tgt, ignore_index, label_smoothing, N, C, D
                )
        ctx.save_for_backward(inp, tgt, weight)
        ctx.N = N
        ctx.C = C
        ctx.D = D
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.mean_num = 1
        ctx.shape = shape

        if reduction == 0:  # NONE
            return out.to(inp.dtype)
        elif reduction == 1:  # MEAN
            if tgt.ndim == dim:
                ctx.mean_num = 1 / (N * D)
            else:
                ctx.mean_num = 1 / sum(w_tgt).item()
            return (sum(out) * ctx.mean_num).to(inp.dtype)
        else:  # SUM
            return sum(out).to(inp.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS CrossEntropyLoss VJP")

        inp, tgt, weight = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        mean_num = ctx.mean_num
        shape = ctx.shape

        out_grad = out_grad.broadcast_to(shape).contiguous()

        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
        if tgt.ndim == inp.ndim:
            celoss_probability_bwd[grid](
                out_grad, inp, tgt, weight, inp_grad, label_smoothing, mean_num, N, C, D
            )
        elif label_smoothing == 0:
            celoss_indice_bwd[grid](
                out_grad, inp, tgt, weight, inp_grad, ignore_index, mean_num, N, C, D
            )
        else:
            celoss_indice_smooth_bwd[grid](
                out_grad,
                inp,
                tgt,
                weight,
                inp_grad,
                ignore_index,
                label_smoothing,
                mean_num,
                N,
                C,
                D,
            )
        return inp_grad, None, None, None, None, None


def cross_entropy_loss(
    inp, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        inp, target, weight, reduction, ignore_index, label_smoothing
    )
