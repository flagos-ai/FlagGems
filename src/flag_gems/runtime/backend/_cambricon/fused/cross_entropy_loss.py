import logging

import torch
import triton
import triton.language as tl
from torch.nn import _reduction as _Reduction

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..ops import sum
from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 2**n}, num_warps=1, num_stages=3)
        for n in range(10, 17, 2)
    ],
    key=["C"],
)
@triton.jit
def softmax_forward_kernel(
    inp_ptr,
    final_max_ptr,
    final_sum_ptr,
    N,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    job_id = tl.program_id(0)
    job_num = tl.num_programs(0)

    batch_per_job = N // job_num
    job_remain_batch = N - batch_per_job * job_num
    batch_per_job += 1
    batch_begin = job_id * batch_per_job
    if job_id >= job_remain_batch:
        batch_per_job -= 1
        batch_begin = job_id * batch_per_job + job_remain_batch
    batch_end = batch_begin + batch_per_job

    for batch_idx in range(batch_begin, batch_end):
        pid_n = batch_idx

        if C <= BLOCK_C:
            offset_d = tl.arange(0, D)
            offset_c = tl.arange(0, C)

            inp_ptrs = (
                inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            inp = tl.load(inp_ptrs).to(tl.float32)
            final_max = tl.max(inp, axis=0)
            final_sum = tl.sum(tl.exp(inp - final_max[None, :]), axis=0)

            final_max_ptrs = final_max_ptr + pid_n * D + offset_d
            final_sum_ptrs = final_sum_ptr + pid_n * D + offset_d

            tl.store(final_max_ptrs, final_max)
            tl.store(final_sum_ptrs, final_sum)
        else:
            tmp_max = tl.zeros([BLOCK_C, D], dtype=tl.float32)
            tmp_sum = tl.zeros([BLOCK_C, D], dtype=tl.float32)
            offset_d = tl.arange(0, D)

            for off in range(0, C, BLOCK_C):
                offset_c = off + tl.arange(0, BLOCK_C)
                inp_ptrs = (
                    inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
                )
                inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
                inp = tl.load(inp_ptrs, mask=inp_mask, other=-float("inf")).to(
                    tl.float32
                )
                cur_max = tl.maximum(tmp_max, inp)
                cur_exp = tl.exp(inp - cur_max)
                tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
                tmp_max = cur_max

            final_max = tl.max(tmp_max, axis=0)
            tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[None, :])
            final_sum = tl.sum(tmp_sum, axis=0)

            final_max_ptrs = final_max_ptr + pid_n * D + offset_d
            final_sum_ptrs = final_sum_ptr + pid_n * D + offset_d

            tl.store(final_max_ptrs, final_max)
            tl.store(final_sum_ptrs, final_sum)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"C_TILE_NUM": num}, num_warps=1, num_stages=s)
        for num in [4, 8, 16, 48]
        for s in [0, 3]
    ],
    key=["C"],
    restore_value=["final_max_ptr"],
)
@triton.jit
def max_kernel(
    inp_ptr,
    final_max_ptr,
    N,
    C: tl.constexpr,
    D: tl.constexpr,
    C_TILE_NUM: tl.constexpr,
):
    job_id = tl.program_id(0)
    job_num = tl.num_programs(0)

    batch_per_job = N // job_num
    job_remain_batch = N - batch_per_job * job_num
    batch_per_job += 1
    batch_begin = job_id * batch_per_job
    if job_id >= job_remain_batch:
        batch_per_job -= 1
        batch_begin = job_id * batch_per_job + job_remain_batch
    batch_end = batch_begin + batch_per_job

    core_id = tl.program_id(1)
    offset_d = tl.arange(0, D)
    BLOCK_C: tl.constexpr = (C + C_TILE_NUM - 1) // C_TILE_NUM

    for batch_idx in range(batch_begin, batch_end):
        pid_n = batch_idx
        offset_c = core_id * BLOCK_C + tl.arange(0, BLOCK_C)

        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C
        inp = tl.load(inp_ptrs, mask=inp_mask, other=-float("inf")).to(tl.float32)

        final_max = tl.max(inp, axis=0)
        final_max_ptrs = final_max_ptr + pid_n * D + offset_d
        tl.atomic_max(final_max_ptrs, final_max)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"C_TILE_NUM": num}, num_warps=1, num_stages=s)
        for num in [4, 8, 16, 48]
        for s in [0, 3]
    ],
    key=["C"],
    reset_to_zero=["final_sum_ptr"],
)
@triton.jit
def softmax_forward_with_max_kernel(
    inp_ptr,
    final_max_ptr,
    final_sum_ptr,
    N,
    C: tl.constexpr,
    D: tl.constexpr,
    C_TILE_NUM: tl.constexpr,
):
    job_id = tl.program_id(0)
    job_num = tl.num_programs(0)

    batch_per_job = N // job_num
    job_remain_batch = N - batch_per_job * job_num
    batch_per_job += 1
    batch_begin = job_id * batch_per_job
    if job_id >= job_remain_batch:
        batch_per_job -= 1
        batch_begin = job_id * batch_per_job + job_remain_batch
    batch_end = batch_begin + batch_per_job

    core_id = tl.program_id(1)
    offset_d = tl.arange(0, D)
    BLOCK_C: tl.constexpr = (C + C_TILE_NUM - 1) // C_TILE_NUM

    for batch_idx in range(batch_begin, batch_end):
        pid_n = batch_idx
        offset_c = core_id * BLOCK_C + tl.arange(0, BLOCK_C)

        final_max_ptrs = final_max_ptr + pid_n * D + offset_d
        final_sum_ptrs = final_sum_ptr + pid_n * D + offset_d
        final_max = tl.load(final_max_ptrs)

        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C
        inp = tl.load(inp_ptrs, mask=inp_mask, other=-float("inf")).to(tl.float32)

        final_sum = tl.sum(tl.exp(inp - final_max[None, :]), axis=0)
        tl.atomic_add(final_sum_ptrs, final_sum)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 2**n}, num_warps=4, num_stages=0)
        for n in range(4, 11, 2)
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["ignore_index"])
def nllloss_without_weight_kernel(
    inp_ptr,
    tgt_ptr,
    final_max_ptr,
    final_sum_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    core_id = tl.program_id(0)
    offset_n = core_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_d = tl.arange(0, D)

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt_mask = offset_n < N
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index)

    final_max_ptrs = final_max_ptr + offset_n * D + offset_d
    final_sum_ptrs = final_sum_ptr + offset_n * D + offset_d
    final_max = tl.load(final_max_ptrs, mask=tgt_mask, other=0)
    final_sum = tl.load(final_sum_ptrs, mask=tgt_mask, other=1)

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=-float("inf")).to(tl.float32)

    loge2 = 0.693147
    out = tl.log2(final_sum) * loge2 + final_max - inp_tgt

    out_ptrs = out_ptr + offset_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask and ignore_mask)


@libentry()
@triton.heuristics(
    values={
        "num_warps": lambda args: 1,
        "num_stages": lambda args: 0,
    },
)
@triton.jit(do_not_specialize=["ignore_index"])
def nllloss_with_weight_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    w_tgt_ptr,
    final_max_ptr,
    final_sum_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offset_d = tl.arange(0, D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt = tl.load(tgt_ptrs)

    ignore_mask = not (tgt == ignore_index)

    if w_ptr is None:
        w_tgt = ignore_mask
    else:
        w_ptrs = w_ptr + tgt
        w_tgt = tl.load(w_ptrs).to(tl.float32)
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d
    tl.store(w_tgt_ptrs, w_tgt, mask=ignore_mask)

    final_max_ptrs = final_max_ptr + pid_n * D + offset_d
    final_sum_ptrs = final_sum_ptr + pid_n * D + offset_d
    final_max = tl.load(final_max_ptrs)
    final_sum = tl.load(final_sum_ptrs)

    inp_tgt_ptrs = inp_ptr + pid_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs).to(tl.float32)

    loge2 = 0.693147
    out = (tl.log2(final_sum) * loge2 + final_max - inp_tgt) * w_tgt

    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=ignore_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
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
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
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
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1.0 - label_smoothing) + label_smoothing / C
        log = final_sum + final_max - inp
        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w = tl.load(w_ptr + offset_c, mask=w_mask, other=0).to(tl.float32)
        _sum += log * tgt * w[:, None]

    out = tl.sum(_sum, axis=0)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=offset_d < D)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
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
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
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
    final_sum_max = final_sum + final_max

    _sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w = tl.load(w_ptr + offset_c, w_mask, other=0).to(tl.float32)

        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            label_smoothing / C,
        ).to(tl.float32)

        log = final_sum_max - inp
        _sum += log * smooth * w[:, None]

    out = tl.sum(_sum, axis=0)
    out = tl.where(ignore_mask, out, 0)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask)


@triton.jit
def single_celoss_indice_bwd(
    pid_n,
    offset_c,
    offset_d,
    final_max,
    final_sum,
    tgt,
    w_tgt,
    out_grad,
    mean_num,
    inp_ptr,
    inp_grad_ptr,
    ignore_mask,
    C,
    D,
):
    inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
    inp_mask = offset_c[:, None] < C
    inp = tl.load(inp_ptrs, mask=inp_mask, other=-float("inf")).to(tl.float32)

    minus_one = offset_c[:, None] == tgt[None, :]
    inp_grad = (
        (tl.exp(inp - final_max[None, :]) / final_sum[None, :] - minus_one)
        * w_tgt
        * out_grad
        * mean_num
    )
    inp_grad_ptrs = (
        inp_grad_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
    )
    tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask and ignore_mask)


def config_prune(configs, named_args, **kwargs):
    pruned_configs = []

    for config in configs:
        kw = config.kwargs
        mode, num, BLOCK_C = (kw["TILE_MODE"], kw["C_TILE_NUM"], kw["BLOCK_C"])
        if (mode == 0 and num == 1) or (mode == 1 and num >= 4 and BLOCK_C <= 1024):
            pruned_configs.append(config)
    return pruned_configs


@libentry()
@triton.autotune(
    configs=[
        triton.Config(
            {
                "TILE_MODE": mode,
                "C_TILE_NUM": num,
                "BLOCK_C": 2**n,
            },
            num_warps=1,
            num_stages=s,
        )
        for mode in [0, 1]
        for num in [1, 4, 8, 16, 48]
        for n in range(10, 17, 2)
        for s in [0, 3]
    ],
    key=["C"],
    prune_configs_by={
        "early_config_prune": config_prune,
    },
)
@triton.jit(do_not_specialize=["ignore_index", "mean_num"])
def celoss_indice_bwd_with_saved_sum_kernel(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    final_max_ptr,
    final_sum_ptr,
    ignore_index,
    mean_num,
    N,
    C: tl.constexpr,
    D: tl.constexpr,
    is_has_weight: tl.constexpr,
    is_has_ignore_index: tl.constexpr,
    is_tgt_in_i32: tl.constexpr,
    TILE_MODE: tl.constexpr,
    C_TILE_NUM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    job_id = tl.program_id(0)
    job_num = tl.num_programs(0)

    batch_per_job = N // job_num
    job_remain_batch = N - batch_per_job * job_num
    batch_per_job += 1
    batch_begin = job_id * batch_per_job
    if job_id >= job_remain_batch:
        batch_per_job -= 1
        batch_begin = job_id * batch_per_job + job_remain_batch
    batch_end = batch_begin + batch_per_job

    for batch_idx in range(batch_begin, batch_end):
        pid_n = batch_idx
        offset_d = tl.arange(0, D)

        tgt_ptrs = tgt_ptr + pid_n * D + offset_d
        if is_tgt_in_i32:
            tgt = tl.load(tgt_ptrs).to(tl.int32)
        else:
            tgt = tl.load(tgt_ptrs)

        out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs).to(tl.float32)[None, :]

        if is_has_weight:
            w_ptrs = w_ptr + tgt
            w_tgt = tl.load(w_ptrs).to(tl.float32)[None, :]
        else:
            w_tgt = 1

        if is_has_ignore_index:
            ignore_mask = (tgt != ignore_index)[None, :]
        else:
            ignore_mask = True

        final_max_ptrs = final_max_ptr + pid_n * D + offset_d
        final_sum_ptrs = final_sum_ptr + pid_n * D + offset_d
        final_max = tl.load(final_max_ptrs)
        final_sum = tl.load(final_sum_ptrs)

        if TILE_MODE == 0:
            if C <= BLOCK_C:
                offset_c = tl.arange(0, C)
                single_celoss_indice_bwd(
                    pid_n,
                    offset_c,
                    offset_d,
                    final_max,
                    final_sum,
                    tgt,
                    w_tgt,
                    out_grad,
                    mean_num,
                    inp_ptr,
                    inp_grad_ptr,
                    ignore_mask,
                    C,
                    D,
                )
            else:
                for off in range(0, C, BLOCK_C):
                    offset_c = off + tl.arange(0, BLOCK_C)
                    single_celoss_indice_bwd(
                        pid_n,
                        offset_c,
                        offset_d,
                        final_max,
                        final_sum,
                        tgt,
                        w_tgt,
                        out_grad,
                        mean_num,
                        inp_ptr,
                        inp_grad_ptr,
                        ignore_mask,
                        C,
                        D,
                    )
        else:
            core_id = tl.program_id(1)
            C_TILE_SIZE: tl.constexpr = (C + C_TILE_NUM - 1) // C_TILE_NUM
            offset_c = core_id * C_TILE_SIZE + tl.arange(0, C_TILE_SIZE)

            single_celoss_indice_bwd(
                pid_n,
                offset_c,
                offset_d,
                final_max,
                final_sum,
                tgt,
                w_tgt,
                out_grad,
                mean_num,
                inp_ptr,
                inp_grad_ptr,
                ignore_mask,
                C,
                D,
            )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
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
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
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

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        w_tgt_sum += tgt * w[:, None]

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

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        grad = w_tgt_sum / final_sum * tl.exp(inp - final_max) - tgt * w[:, None]
        inp_grad = grad * out_grad * mean_num

        inp_grad_ptrs = inp_grad_ptr + offset
        tl.store(inp_grad_ptrs, inp_grad, mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
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
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
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

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
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

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            label_smoothing / C,
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
        logger.debug("GEMS_CAMBRICON CrossEntropyLoss")

        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        inp = inp.contiguous()
        tgt = target.contiguous()

        ctx.N = N
        ctx.C = C
        ctx.D = D
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.shape = shape

        final_max = None
        final_sum = None

        mean_num = 1
        if reduction == 1 and tgt.ndim == dim:
            mean_num = 1 / (N * D)
        out = torch.empty(shape, dtype=torch.float32, device=inp.device)

        def get_result(inp, tgt, out, reduction, mean_num):
            if reduction == 0:  # NONE
                return out.to(inp.dtype)
            elif reduction == 1:  # MEAN
                return (sum(out) * mean_num).to(inp.dtype)
            else:  # SUM
                return sum(out).to(inp.dtype)

        if weight is None and tgt.ndim != dim and label_smoothing == 0:
            final_max = torch.full(
                shape,
                torch.finfo(torch.float32).min,
                dtype=torch.float32,
                device=inp.device,
            )
            final_sum = torch.zeros(shape, dtype=torch.float32, device=inp.device)
            with torch.mlu.device(inp.device):
                if C <= (32 * 1000) or C > (2048 * 1000):
                    softmax_forward_kernel[(TOTAL_CORE_NUM,)](
                        inp, final_max, final_sum, N, C, D
                    )
                else:
                    grid = lambda meta: (
                        triton.cdiv(TOTAL_CORE_NUM, meta["C_TILE_NUM"]),
                        meta["C_TILE_NUM"],
                    )
                    max_kernel[grid](inp, final_max, N, C, D)
                    softmax_forward_with_max_kernel[grid](
                        inp, final_max, final_sum, N, C, D
                    )

                grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
                nllloss_without_weight_kernel[grid](
                    inp, tgt, final_max, final_sum, out, ignore_index, N, C, D
                )
            if reduction == 1:
                if ignore_index < 0 or ignore_index >= C:
                    mean_num = 1 / C
                else:
                    mean_num = 1 / (C - 1)
            ctx.mean_num = mean_num

            ctx.save_for_backward(inp, tgt, weight, final_max, final_sum)
            return get_result(inp, tgt, out, reduction, mean_num)

        weight = weight.contiguous() if weight is not None else None
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)

        if tgt.ndim == dim:
            # target probabilities
            with torch_device_fn.device(inp.device):
                celoss_probability_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    label_smoothing,
                    C,
                    D,
                )
        elif label_smoothing == 0:
            # target indices
            w_tgt = torch.zeros(shape, dtype=torch.float32, device=inp.device)
            final_max = torch.empty(shape, dtype=torch.float32, device=inp.device)
            final_sum = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                softmax_forward_kernel[(TOTAL_CORE_NUM,)](
                    inp, final_max, final_sum, N, C, D
                )
                nllloss_with_weight_kernel[(N,)](
                    inp,
                    tgt,
                    weight,
                    w_tgt,
                    final_max,
                    final_sum,
                    out,
                    ignore_index,
                    N,
                    C,
                    D,
                )
        else:
            w_tgt = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
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
        ctx.save_for_backward(inp, tgt, weight, final_max, final_sum)
        ctx.mean_num = 1

        if reduction == 1 and tgt.ndim != dim:
            mean_num = 1 / sum(w_tgt).item()
        ctx.mean_num = mean_num
        return get_result(inp, tgt, out, reduction, mean_num)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_CAMBRICON CrossEntropyLoss VJP")

        inp, tgt, weight, final_max, final_sum = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        mean_num = ctx.mean_num
        shape = ctx.shape

        out_grad = out_grad.broadcast_to(shape).contiguous()

        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)
        if tgt.ndim == inp.ndim:
            celoss_probability_bwd[grid](
                out_grad, inp, tgt, weight, inp_grad, label_smoothing, mean_num, C, D
            )
        elif label_smoothing == 0:
            if final_sum is not None:
                is_has_weight = weight is not None
                is_has_ignore_index = ignore_index >= 0 and ignore_index < C
                is_tgt_in_i32 = C < (1 << 31)
                grid = lambda meta: (
                    triton.cdiv(TOTAL_CORE_NUM, meta["C_TILE_NUM"]),
                    meta["C_TILE_NUM"],
                )
                celoss_indice_bwd_with_saved_sum_kernel[grid](
                    out_grad,
                    inp,
                    tgt,
                    weight,
                    inp_grad,
                    final_max,
                    final_sum,
                    ignore_index,
                    mean_num,
                    N,
                    C,
                    D,
                    is_has_weight,
                    is_has_ignore_index,
                    is_tgt_in_i32,
                )
        else:
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
        return inp_grad, None, None, None, None, None


def cross_entropy_loss(
    inp, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        inp,
        target,
        weight,
        _Reduction.get_enum(reduction),
        ignore_index,
        label_smoothing,
    )
