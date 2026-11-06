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
import importlib
import logging
import sys

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_GENERIC_MODULE_NAME = "flag_gems.ops._pad_circular"

_MAX_BLOCK = 4096


@triton.jit
def _pad_circular_run_1d_kernel(
    x_ptr,
    out_ptr,
    W,
    Wo,
    out_beg,
    src_beg,
    run_len,
    BLOCK: tl.constexpr,
):
    # npad == 1: the leading dims are untouched, so output row == input row.
    row = tl.program_id(0)
    blk = tl.program_id(1)
    off = blk * BLOCK + tl.arange(0, BLOCK)
    m = off < run_len
    vals = tl.load(x_ptr + (row * W + src_beg) + off, mask=m)
    tl.store(out_ptr + (row * Wo + out_beg) + off, vals, mask=m)


@triton.jit
def _pad_circular_run_2d_kernel(
    x_ptr,
    out_ptr,
    H,
    Hc,
    cbh,
    W,
    Ho,
    Wo,
    plh,
    out_beg,
    src_beg,
    run_len,
    BLOCK: tl.constexpr,
):
    # npad == 2: rows enumerate the collapsed (batch, Ho) planes.  The dim -2
    # source row is a scalar per program: one divmod for (b, h), then the
    # periodic extension of the center window, 0 <= plh < Hc so one modulo.
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // Ho
    h = row - b * Ho
    src_row = b * H + (cbh + (h - plh + Hc) % Hc)
    off = blk * BLOCK + tl.arange(0, BLOCK)
    m = off < run_len
    vals = tl.load(x_ptr + (src_row * W + src_beg) + off, mask=m)
    tl.store(out_ptr + (row * Wo + out_beg) + off, vals, mask=m)


@triton.jit
def _pad_circular_run_3d_kernel(
    x_ptr,
    out_ptr,
    D,
    Dc,
    cbd,
    H,
    Hc,
    cbh,
    W,
    Do,
    Ho,
    Wo,
    pld,
    plh,
    out_beg,
    src_beg,
    run_len,
    BLOCK: tl.constexpr,
):
    # npad == 3: rows enumerate the collapsed (batch, Do, Ho) planes.
    row = tl.program_id(0)
    blk = tl.program_id(1)
    plane = Do * Ho
    b = row // plane
    rem = row - b * plane
    d = rem // Ho
    h = rem - d * Ho
    sd = cbd + (d - pld + Dc) % Dc
    sh = cbh + (h - plh + Hc) % Hc
    src_row = (b * D + sd) * H + sh
    off = blk * BLOCK + tl.arange(0, BLOCK)
    m = off < run_len
    vals = tl.load(x_ptr + (src_row * W + src_beg) + off, mask=m)
    tl.store(out_ptr + (row * Wo + out_beg) + off, vals, mask=m)


def _center(size, pl, pr):
    cb = max(-pl, 0)
    C = size - cb - max(-pr, 0)
    return cb, C, max(pl, 0), max(pr, 0)


def _last_dim_runs(C, cb, plc, prc):
    Wo = C + plc + prc
    runs = []
    w = 0
    while w < Wo:
        s = (w - plc) % C
        run_len = min(C - s, Wo - w)
        runs.append((w, run_len, cb + s))
        w += run_len
    return Wo, runs


def _pad_circular(x: torch.Tensor, pad):
    logger.debug("GEMS_KUNLUNXIN _PAD_CIRCULAR")

    ndim = x.dim()
    if len(pad) % 2 != 0:
        raise ValueError("padding length must be even")
    npad = len(pad) // 2
    if npad < 1 or npad > 3:
        raise ValueError(
            "flag_gems _pad_circular supports padding the last 1, 2 or 3 dimensions"
        )
    if ndim <= npad:
        raise ValueError("input must have at least one non-padded (batch) dimension")

    x = x.contiguous()
    in_shape = list(x.shape)
    out_shape = list(in_shape)
    # per padded dim: center base, center length, non-negative left/right pad
    cbs = [0] * ndim
    lens = list(in_shape)
    pl_pos = [0] * ndim
    pr_pos = [0] * ndim
    for i in range(npad):
        d = ndim - 1 - i
        pl = int(pad[2 * i])
        pr = int(pad[2 * i + 1])
        new_size = in_shape[d] + pl + pr
        if new_size <= 0:
            raise ValueError(
                "negative padding removed the whole dimension "
                f"{d}: input {in_shape[d]} pad ({pl}, {pr})"
            )
        # operator contract: circular padding wraps at most once
        if pl > in_shape[d] or pr > in_shape[d]:
            raise ValueError("Padding value causes wrapping around more than once.")
        cbs[d], lens[d], pl_pos[d], pr_pos[d] = _center(in_shape[d], pl, pr)
        out_shape[d] = new_size

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    total = 1
    for s in out_shape:
        total *= s
    if total == 0:
        return out

    W = in_shape[-1]
    Wo, runs = _last_dim_runs(lens[-1], cbs[-1], pl_pos[-1], pr_pos[-1])

    rows = 1
    for s in out_shape[:-1]:
        rows *= s
    if rows == 0:
        return out

    with torch_device_fn.device(x.device):
        for out_beg, run_len, src_beg in runs:
            block = min(triton.next_power_of_2(run_len), _MAX_BLOCK)
            grid = (rows, triton.cdiv(run_len, block))
            if npad == 1:
                _pad_circular_run_1d_kernel[grid](
                    x,
                    out,
                    W,
                    Wo,
                    out_beg,
                    src_beg,
                    run_len,
                    BLOCK=block,
                )
            elif npad == 2:
                _pad_circular_run_2d_kernel[grid](
                    x,
                    out,
                    in_shape[-2],
                    lens[-2],
                    cbs[-2],
                    W,
                    out_shape[-2],
                    Wo,
                    pl_pos[-2] % lens[-2],
                    out_beg,
                    src_beg,
                    run_len,
                    BLOCK=block,
                )
            else:
                _pad_circular_run_3d_kernel[grid](
                    x,
                    out,
                    in_shape[-3],
                    lens[-3],
                    cbs[-3],
                    in_shape[-2],
                    lens[-2],
                    cbs[-2],
                    W,
                    out_shape[-3],
                    out_shape[-2],
                    Wo,
                    pl_pos[-3] % lens[-3],
                    pl_pos[-2] % lens[-2],
                    out_beg,
                    src_beg,
                    run_len,
                    BLOCK=block,
                )
    return out


def _patch_generic_entry():
    generic_module = sys.modules.get(_GENERIC_MODULE_NAME)
    if generic_module is None:  # pragma: no cover - import-order safety net
        try:
            generic_module = importlib.import_module(_GENERIC_MODULE_NAME)
        except ImportError:
            return
    generic_module._pad_circular = _pad_circular


_patch_generic_entry()


__all__ = ["_pad_circular"]
