import logging
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def heur_block_c(args):
    bc = triton.next_power_of_2(triton.cdiv(args["C"], 12))
    return bc if bc > 64 else 64
    # return triton.cdiv(args["C"], 12)


def heur_block_d(args):
    # return args["D"]
    return triton.cdiv(args["D"], 12)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("cross_entropy_loss"),
#     key=["C", "D"],
# )
@triton.heuristics(
    values={
        "BLOCK_C": heur_block_c,
        "BLOCK_D": heur_block_d,
    },
)
@triton.jit(do_not_specialize=["ignore_index"])
def celoss_indices_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index) and tgt_mask

    tmp_max = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp_mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=1)
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[:, None])
    final_sum = tl.log(tl.sum(tmp_sum, axis=1))

    inp_tgt_ptrs = inp_ptr + pid_n * C * D + tgt + offset_d * C
    inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=-float("inf")).to(tl.float32)

    out = final_sum + final_max - inp_tgt
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d

    if w_ptr is None:
        w_tgt = ignore_mask
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask)
    out *= w_tgt
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("cross_entropy_loss"),
#     key=["C", "D"],
# )
@triton.heuristics(
    values={
        "BLOCK_C": heur_block_c,
        "BLOCK_D": heur_block_d,
    },
)
@triton.jit(do_not_specialize=["label_smoothing"])
def celoss_probability_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    label_smoothing,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tmp_max = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp_mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=1)
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[:, None])
    final_sum = tl.log(tl.sum(tmp_sum, axis=1))

    _sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        tgt_ptrs = tgt_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1.0 - label_smoothing) + label_smoothing / C
        log = final_sum[:, None] + final_max[:, None] - inp
        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w = tl.load(w_ptr + offset_c, mask=w_mask, other=0).to(tl.float32)
        _sum += log * tgt * w[None, :]

    out = tl.sum(_sum, axis=1)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=offset_d < D)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("cross_entropy_loss"),
#     key=["C", "D"],
# )
@triton.heuristics(
    values={
        "BLOCK_C": heur_block_c,
        "BLOCK_D": heur_block_d,
    },
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing"])
def celoss_indices_smooth_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    label_smoothing,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index) and tgt_mask

    if w_ptr is None:
        w_tgt = ignore_mask
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0)
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d
    tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask)

    tmp_max = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=1)[:, None]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.log(tl.sum(tmp_sum, axis=1))[:, None]
    final_sum_max = final_sum + final_max

    _sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w = tl.load(w_ptr + offset_c, w_mask, other=0).to(tl.float32)

        smooth = tl.where(
            offset_c[None, :] == tgt[:, None],
            1 - label_smoothing + label_smoothing / C,
            label_smoothing / C,
        ).to(tl.float32)

        log = final_sum_max - inp
        _sum += log * smooth * w[None, :]

    out = tl.sum(_sum, axis=1)
    out = tl.where(ignore_mask, out, 0)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("cross_entropy_loss"),
#     key=["C", "D"],
# )
@triton.heuristics(
    values={
        "BLOCK_C": heur_block_c,
        "BLOCK_D": heur_block_d,
    },
)
@triton.jit(do_not_specialize=["ignore_index", "mean_num"])
def celoss_indices_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    mean_num,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)

    ignore_mask = not (tgt == ignore_index) and tgt_mask

    if w_ptr is None:
        w_tgt = ignore_mask
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    tmp_max = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp_mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=1)
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[:, None])
    final_sum = tl.sum(tmp_sum, axis=1)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp_mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        minus_one = (offset_c[None, :] == tgt[:, None]).to(tl.float32)
        inp_grad = (
            (tl.exp(inp - final_max[:, None]) / final_sum[:, None] - minus_one)
            * w_tgt[:, None]
            * out_grad[:, None]
            * mean_num
        )
        inp_grad_ptrs = (
            inp_grad_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        )
        tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask and ignore_mask[:, None])


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("cross_entropy_loss"),
#     key=["C", "D"],
# )
@triton.heuristics(
    values={
        "BLOCK_C": heur_block_c,
        "BLOCK_D": heur_block_d,
    },
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
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=offset_d < D, other=0).to(tl.float32)

    tmp_max = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    w_tgt_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp = tl.load(inp_ptrs, mask, other=-float("inf")).to(tl.float32)

        tgt_ptrs = tgt_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        w_tgt_sum += tgt * w[None, :]

        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=1)[:, None]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=1)[:, None]
    w_tgt_sum = tl.sum(w_tgt_sum, axis=1)[:, None]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        offset = pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp_ptrs = inp_ptr + offset
        mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        tgt_ptrs = tgt_ptr + offset
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        grad = w_tgt_sum / final_sum * tl.exp(inp - final_max) - tgt * w[None, :]
        inp_grad = grad * out_grad[:, None] * mean_num

        inp_grad_ptrs = inp_grad_ptr + offset
        tl.store(inp_grad_ptrs, inp_grad, mask)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("cross_entropy_loss"),
#     key=["C", "D"],
# )
@triton.heuristics(
    values={
        "BLOCK_C": heur_block_c,
        "BLOCK_D": heur_block_d,
    },
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing", "mean_num"])
def celoss_indices_smooth_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    label_smoothing,
    mean_num,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)

    ignore_mask = (tgt != ignore_index)[:, None]

    tmp_max = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)
    w_sum = tl.zeros([BLOCK_D, BLOCK_C], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp_mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        smooth = tl.full([BLOCK_D, BLOCK_C], label_smoothing / C, dtype=tl.float32)
        smooth = tl.where(
            offset_c[None, :] == tgt[:, None],
            1 - label_smoothing + label_smoothing / C,
            smooth,
        )

        w_sum += smooth * w[None, :]

        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=1)[:, None]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=1)[:, None]
    w_sum = tl.sum(w_sum, axis=1)[:, None]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        inp_mask = offset_c[None, :] < C and offset_d[:, None] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        smooth = tl.where(
            offset_c[None, :] == tgt[:, None],
            1 - label_smoothing + label_smoothing / C,
            label_smoothing / C,
        )

        grad = w_sum / final_sum * tl.exp(inp - final_max) - smooth * w[None, :]
        inp_grad = grad * out_grad[:, None] * mean_num
        inp_grad_ptrs = (
            inp_grad_ptr + pid_n * C * D + offset_d[:, None] * C + offset_c[None, :]
        )
        tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask and ignore_mask)


@libentry()
@triton.jit
def sum_and_scale(
    inp_ptr,
    out_ptr,
    N,
    scalebyw: tl.constexpr,
    BLOCK_N: tl.constexpr = 128,
    scale=1.0,
    mean_num=None,
):
    mid_sum = tl.zeros(
        [
            BLOCK_N,
        ],
        dtype=tl.float32,
    )
    if scalebyw:
        mid_wgt = tl.zeros(
            [
                BLOCK_N,
            ],
            dtype=tl.float32,
        )
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
            wgt_ptrs = scale + offset
            wgt_vals = tl.load(wgt_ptrs, mask=mask, other=0.0)
            mid_wgt += wgt_vals
        out_val = tl.sum(mid_sum)
        scale_val = tl.sum(mid_wgt)
        tl.store(mean_num, scale_val)
    else:
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
        out_val = tl.sum(mid_sum)
        scale_val = scale
    out_val /= scale_val
    tl.store(out_ptr, out_val)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logger.debug("GEMS CrossEntropyLoss")

        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]

        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        grad = inp.requires_grad
        if dim == 3:
            inp = inp.transpose(1, -1)
            D_new = inp.shape[1]
        else:
            D_new = D

        inp = inp.contiguous()
        if dim == 3:
            target = target.transpose(1, -1).contiguous()
        tgt = target.contiguous()
        weight = weight.contiguous() if weight is not None else None
        out = torch.empty(shape, dtype=torch.float32, device=inp.device)
        grid = lambda meta: (triton.cdiv(D_new, meta["BLOCK_D"]), N)

        if tgt.ndim == dim:
            # target probabilities
            with torch_device_fn.device(inp.device):
                if shape != [1]:
                    os.environ["TRITONXPU_OTHER_SIM"] = "1"
                    os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
                    os.environ["TRITONXPU_CLOSE_OPTIMIZE"] = "1"
                celoss_probability_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    label_smoothing,
                    C,
                    D,
                )
                if shape != [1]:
                    if "TRITONXPU_OTHER_SIM" in os.environ:
                        del os.environ["TRITONXPU_OTHER_SIM"]
                    if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                        del os.environ["TRITONXPU_STORE_MASK_SIM"]
                    if "TRITONXPU_CLOSE_OPTIMIZE" in os.environ:
                        del os.environ["TRITONXPU_CLOSE_OPTIMIZE"]
        elif label_smoothing == 0:
            # target indices
            w_tgt = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                celoss_indices_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    w_tgt,
                    ignore_index,
                    C,
                    D,
                )
                if dim > 1:
                    out = out.view(shape[:axis] + shape[axis + 1 :])
        else:
            w_tgt = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                os.environ["TRITONXPU_OTHER_SIM"] = "1"
                os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
                os.environ["TRITONXPU_CLOSE_OPTIMIZE"] = "1"
                celoss_indices_smooth_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    w_tgt,
                    ignore_index,
                    label_smoothing,
                    C,
                    D,
                )
                if "TRITONXPU_OTHER_SIM" in os.environ:
                    del os.environ["TRITONXPU_OTHER_SIM"]
                if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                    del os.environ["TRITONXPU_STORE_MASK_SIM"]
                if "TRITONXPU_CLOSE_OPTIMIZE" in os.environ:
                    del os.environ["TRITONXPU_CLOSE_OPTIMIZE"]
        if reduction == "mean":  # MEAN
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            if tgt.ndim == dim:
                sum_and_scale[(1,)](out, out_reduce, N * D, False, scale=N * D)
            else:
                wgt_sum = torch.empty([], dtype=torch.float32, device=inp.device)
                sum_and_scale[(1,)](
                    out, out_reduce, N * D, True, scale=w_tgt, mean_num=wgt_sum
                )
            out = out_reduce
        elif reduction == "sum":  # SUM
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            sum_and_scale[(1,)](out, out_reduce, N * D, False)
            out = out_reduce

        if grad:
            ctx.save_for_backward(inp, tgt, weight)
            ctx.N = N
            ctx.C = C
            ctx.D = D
            ctx.ignore_index = ignore_index
            ctx.label_smoothing = label_smoothing
            ctx.shape = shape
            ctx.mean_num = 1
            if reduction == "mean":
                ctx.mean_num = N * D if tgt.ndim == dim else wgt_sum

        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS CrossEntropyLoss VJP")

        inp, tgt, weight = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        mean_num = (
            1 / ctx.mean_num.item()
            if isinstance(ctx.mean_num, torch.Tensor)
            else 1 / ctx.mean_num
        )

        shape = ctx.shape
        out_grad = out_grad.broadcast_to(shape).contiguous()
        dim = inp.ndim
        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)

        if tgt.ndim == inp.ndim:
            if shape != [1]:
                os.environ["TRITONXPU_OTHER_SIM"] = "1"
                os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
                os.environ["TRITONXPU_CLOSE_OPTIMIZE"] = "1"
            celoss_probability_bwd[grid](
                out_grad, inp, tgt, weight, inp_grad, label_smoothing, mean_num, C, D
            )
            if shape != [1]:
                if "TRITONXPU_OTHER_SIM" in os.environ:
                    del os.environ["TRITONXPU_OTHER_SIM"]
                if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                    del os.environ["TRITONXPU_STORE_MASK_SIM"]
                if "TRITONXPU_CLOSE_OPTIMIZE" in os.environ:
                    del os.environ["TRITONXPU_CLOSE_OPTIMIZE"]
        elif label_smoothing == 0:
            celoss_indices_bwd[grid](
                out_grad, inp, tgt, weight, inp_grad, ignore_index, mean_num, C, D
            )
        else:
            os.environ["TRITONXPU_OTHER_SIM"] = "1"
            os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"
            os.environ["TRITONXPU_CLOSE_OPTIMIZE"] = "1"
            celoss_indices_smooth_bwd[grid](
                out_grad,
                inp,
                tgt,
                weight,
                inp_grad,
                ignore_index,
                label_smoothing,
                mean_num,
                C,
                D,
            )
            if "TRITONXPU_OTHER_SIM" in os.environ:
                del os.environ["TRITONXPU_OTHER_SIM"]
            if "TRITONXPU_STORE_MASK_SIM" in os.environ:
                del os.environ["TRITONXPU_STORE_MASK_SIM"]
            if "TRITONXPU_CLOSE_OPTIMIZE" in os.environ:
                del os.environ["TRITONXPU_CLOSE_OPTIMIZE"]
        if dim == 3:
            inp_grad = inp_grad.transpose(1, -1).contiguous()
        return inp_grad, None, None, None, None, None


def cross_entropy_loss(
    inp, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        inp, target, weight, reduction, ignore_index, label_smoothing
    )
