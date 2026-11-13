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
from torch.nn import _reduction as _Reduction

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ...gcu400.utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


# ext.program_id to tl.program_id for gcu400 temply
# change pid_d and pid_n in grid(celoss_probability_kernel, celoss_probability_bwd)@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
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
    N,
    GRID_N,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    for n_idx in range(pid_n, N, GRID_N):
        tgt_ptrs = tgt_ptr + n_idx * D + offset_d
        tgt_mask = offset_d < D
        tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

        ignore_mask = (not (tgt == ignore_index)) & tgt_mask

        tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

        for off in range(0, C, BLOCK_C):
            offset_c = off + tl.arange(0, BLOCK_C)
            inp_ptrs = (
                inp_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
            inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
            cur_max = tl.maximum(tmp_max, inp)
            cur_exp = tl.exp(inp - cur_max)
            tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
            tmp_max = cur_max
        final_max = tl.max(tmp_max, axis=0)
        tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[None, :])
        final_sum = tl.log(tl.sum(tmp_sum, axis=0))

        inp_tgt_ptrs = inp_ptr + n_idx * C * D + tgt * D + offset_d
        inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=-float("inf")).to(
            tl.float32
        )

        out = final_sum + final_max - inp_tgt
        w_tgt_ptrs = w_tgt_ptr + n_idx * D + offset_d

        if w_ptr is None:
            w_tgt = ignore_mask
        else:
            w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

        tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask)
        out *= w_tgt
        out_ptrs = out_ptr + n_idx * D + offset_d
        tl.store(out_ptrs, out, mask=tgt_mask)


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
    pid_d = tl.program_id(1)
    pid_n = tl.program_id(0)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
        mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
    N,
    GRID_N,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    for n_idx in range(pid_n, N, GRID_N):
        tgt_ptrs = tgt_ptr + n_idx * D + offset_d
        tgt_mask = offset_d < D
        tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

        ignore_mask = (not (tgt == ignore_index)) & tgt_mask

        if w_ptr is None:
            w_tgt = ignore_mask
        else:
            w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0)
        w_tgt_ptrs = w_tgt_ptr + n_idx * D + offset_d
        tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask)

        tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

        for off in range(0, C, BLOCK_C):
            offset_c = off + tl.arange(0, BLOCK_C)
            inp_ptrs = (
                inp_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
            inp_ptrs = (
                inp_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
        out_ptrs = out_ptr + n_idx * D + offset_d
        tl.store(out_ptrs, out, mask=tgt_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
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
    N,
    GRID_N,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    for n_idx in range(pid_n, N, GRID_N):
        tgt_ptrs = tgt_ptr + n_idx * D + offset_d
        tgt_mask = offset_d < D
        tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
        out_grad_ptrs = out_grad_ptr + n_idx * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)[
            None, :
        ]

        if w_ptr is None:
            w_tgt = tgt_mask
        else:
            w_ptrs = w_ptr + tgt
            w_tgt = tl.load(w_ptrs, mask=tgt_mask, other=0).to(tl.float32)[None, :]

        ignore_mask = (tgt != ignore_index)[None, :]

        tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

        for off in range(0, C, BLOCK_C):
            offset_c = off + tl.arange(0, BLOCK_C)
            inp_ptrs = (
                inp_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
            inp_ptrs = (
                inp_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
            inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
            minus_one = offset_c[:, None] == tgt[None, :]
            inp_grad = (
                (tl.exp(inp - final_max) / final_sum - minus_one)
                * w_tgt
                * out_grad
                * mean_num
            )
            inp_grad_ptrs = (
                inp_grad_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask & ignore_mask)


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
    pid_d = tl.program_id(1)
    pid_n = tl.program_id(0)
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
        mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
        mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
    N,
    GRID_N,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    for n_idx in range(pid_n, N, GRID_N):
        tgt_ptrs = tgt_ptr + n_idx * D + offset_d
        tgt_mask = offset_d < D
        tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)
        out_grad_ptrs = out_grad_ptr + n_idx * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=tgt_mask, other=0).to(tl.float32)[
            None, :
        ]

        ignore_mask = (tgt != ignore_index)[None, :]

        tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
        w_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

        for off in range(0, C, BLOCK_C):
            offset_c = off + tl.arange(0, BLOCK_C)
            inp_ptrs = (
                inp_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
            inp_ptrs = (
                inp_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            inp_mask = (offset_c[:, None] < C) & (offset_d[None, :] < D)
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
                inp_grad_ptr + n_idx * C * D + offset_c[:, None] * D + offset_d[None, :]
            )
            tl.store(inp_grad_ptrs, inp_grad, mask=inp_mask & ignore_mask)


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


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def celoss_indices_fused_kernel(
    inp_ptr,
    tgt_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    C,
    N,
    GRID_N,
    BLOCK_C: tl.constexpr,
):
    """Single-pass per-row logsumexp for target-indices cross entropy.

    The generic kernel walks the C dimension in BLOCK_C chunks with a serial
    online-softmax dependency chain.  When the whole row fits one tile we can
    compute max / exp / sum directly, removing both the loop and the repeated
    exp(tmp_max - cur_max) rescaling.  Measured on gcu400 L600: 4096x4096
    843us -> 197us, 1024x1024 132us -> 47us.
    """
    pid_n = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    for n_idx in range(pid_n, N, GRID_N):
        x = tl.load(inp_ptr + n_idx * C + cols, mask=cols < C, other=-float("inf")).to(
            tl.float32
        )
        row_max = tl.max(x, 0)
        log_sum_exp = tl.log(tl.sum(tl.exp(x - row_max), 0)) + row_max
        tgt = tl.load(tgt_ptr + n_idx).to(tl.int32)
        ignore_mask = tgt != ignore_index
        inp_tgt = tl.load(inp_ptr + n_idx * C + tgt, mask=ignore_mask, other=0.0).to(
            tl.float32
        )
        w = tl.where(ignore_mask, 1.0, 0.0)
        tl.store(out_ptr + n_idx, (log_sum_exp - inp_tgt) * w)
        tl.store(w_tgt_ptr + n_idx, w)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logger.debug("GEMS_ENFLAME CROSS_ENTROPY_LOSS")

        if target.dtype == torch.int64:
            target = target.to(torch.int32)
        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        inp = inp.contiguous()
        tgt = target.contiguous()
        weight = weight.contiguous() if weight is not None else None
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)
        wgt_sum = None

        if tgt.ndim == dim:
            # target probabilities
            out = torch.empty(shape, dtype=torch.float32, device=inp.device)
            grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
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
            # target indices: out / w_tgt / wgt_sum share one fp32 allocation
            n_per = N * D
            buf = torch.empty(n_per * 2 + 1, dtype=torch.float32, device=inp.device)
            out = buf[:n_per].view(shape)
            w_tgt = buf[n_per : 2 * n_per].view(shape)
            wgt_sum = buf[2 * n_per : 2 * n_per + 1].view(())
            GRID_N = min(N, MAX_GRID_DIM)
            if weight is None and D == 1 and C <= 65536:
                # Whole row fits one tile: use the dependency-free single pass.
                grid_fused = (GRID_N,)
                with torch_device_fn.device(inp.device):
                    celoss_indices_fused_kernel[grid_fused](
                        inp,
                        tgt,
                        out,
                        w_tgt,
                        ignore_index,
                        C,
                        N,
                        GRID_N,
                        BLOCK_C=triton.next_power_of_2(C),
                        num_warps=4,
                    )
            else:
                grid_indices = lambda meta: (
                    triton.cdiv(D, meta["BLOCK_D"]),
                    GRID_N,
                )
                with torch_device_fn.device(inp.device):
                    celoss_indices_kernel[grid_indices](
                        inp,
                        tgt,
                        weight,
                        out,
                        w_tgt,
                        ignore_index,
                        C,
                        D,
                        N,
                        GRID_N,
                    )
        else:
            n_per = N * D
            buf = torch.empty(n_per * 2 + 1, dtype=torch.float32, device=inp.device)
            out = buf[:n_per].view(shape)
            w_tgt = buf[n_per : 2 * n_per].view(shape)
            wgt_sum = buf[2 * n_per : 2 * n_per + 1].view(())
            GRID_N = min(N, MAX_GRID_DIM)
            grid_smooth = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), GRID_N)
            with torch_device_fn.device(inp.device):
                celoss_indices_smooth_kernel[grid_smooth](
                    inp,
                    tgt,
                    weight,
                    out,
                    w_tgt,
                    ignore_index,
                    label_smoothing,
                    C,
                    D,
                    N,
                    GRID_N,
                )

        if reduction == 1:  # MEAN
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            if tgt.ndim == dim:
                sum_and_scale[(1,)](out, out_reduce, N * D, False, scale=N * D)
            else:
                sum_and_scale[(1,)](
                    out, out_reduce, N * D, True, scale=w_tgt, mean_num=wgt_sum
                )
            out = out_reduce
        elif reduction == 2:  # SUM
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            sum_and_scale[(1,)](out, out_reduce, N * D, False)
            out = out_reduce

        if inp.requires_grad:
            ctx.save_for_backward(inp, tgt, weight)
            ctx.N = N
            ctx.C = C
            ctx.D = D
            ctx.ignore_index = ignore_index
            ctx.label_smoothing = label_smoothing
            ctx.shape = shape
            ctx.mean_num = 1
            if reduction == 1:
                ctx.mean_num = N * D if tgt.ndim == dim else wgt_sum

        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_ENFLAME CROSS_ENTROPY_LOSS_VJP")

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

        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)
        if tgt.ndim == inp.ndim:
            grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
            celoss_probability_bwd[grid](
                out_grad, inp, tgt, weight, inp_grad, label_smoothing, mean_num, C, D
            )
        elif label_smoothing == 0:
            GRID_N = min(N, MAX_GRID_DIM)
            grid_indices = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), GRID_N)
            celoss_indices_bwd[grid_indices](
                out_grad,
                inp,
                tgt,
                weight,
                inp_grad,
                ignore_index,
                mean_num,
                C,
                D,
                N,
                GRID_N,
            )
        else:
            GRID_N = min(N, MAX_GRID_DIM)
            grid_smooth = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), GRID_N)
            celoss_indices_smooth_bwd[grid_smooth](
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
                N,
                GRID_N,
            )
        return inp_grad, None, None, None, None, None


def _cross_entropy_loss_fast(inp, target, reduction_enum, ignore_index):
    N, C = inp.shape
    GRID_N = min(N, MAX_GRID_DIM)
    buf = torch.empty(N * 2 + 1, dtype=torch.float32, device=inp.device)
    out = buf[:N]
    w_tgt = buf[N : 2 * N]
    wgt_sum = buf[2 * N : 2 * N + 1].view(())
    out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
    tgt = target.to(torch.int32) if target.dtype == torch.int64 else target
    with torch_device_fn.device(inp.device):
        celoss_indices_fused_kernel[(GRID_N,)](
            inp,
            tgt,
            out,
            w_tgt,
            ignore_index,
            C,
            N,
            GRID_N,
            BLOCK_C=triton.next_power_of_2(C),
            num_warps=4,
        )
    if reduction_enum == 1:
        sum_and_scale[(1,)](out, out_reduce, N, True, scale=w_tgt, mean_num=wgt_sum)
    else:
        sum_and_scale[(1,)](out, out_reduce, N, False)
    return out_reduce


def cross_entropy_loss(
    inp, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0
):
    reduction_enum = _Reduction.get_enum(reduction)
    # Fast path for the common inference case (2D logits, class indices, mean
    # or sum reduction, no weight/smoothing).  Skips autograd.Function and the
    # generic forward dispatch; one allocation + two kernels instead of four
    # allocations + scattered host work (about 50us of fixed overhead).
    if (
        not inp.requires_grad
        and not (isinstance(weight, torch.Tensor) and weight.requires_grad)
        and weight is None
        and label_smoothing == 0
        and reduction_enum in (1, 2)
        and inp.ndim == 2
        and inp.is_contiguous()
        and target.ndim == 1
        and target.shape[0] == inp.shape[0]
        and 1 <= inp.shape[1] <= 65536
    ):
        return _cross_entropy_loss_fast(inp, target, reduction_enum, ignore_index)
    # Inference path: skip torch.autograd.Function.apply (about 30us of graph
    # construction on small shapes).  forward() only touches ctx when the
    # input requires grad, which is false on this path.
    if not inp.requires_grad and not (
        isinstance(weight, torch.Tensor) and weight.requires_grad
    ):
        return CrossEntropyLoss.forward(
            None, inp, target, weight, reduction_enum, ignore_index, label_smoothing
        )
    return CrossEntropyLoss.apply(
        inp,
        target,
        weight,
        reduction_enum,
        ignore_index,
        label_smoothing,
    )
