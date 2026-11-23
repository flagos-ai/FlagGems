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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

logger = logging.getLogger(__name__)

# GCU launch limit for grid.x (triton.backends.enflame backend.GcuLauncher).
_GRID_X_MAX = 65535
# Target bytes of log_probs data handled by one CTA. Calibrated on GCU400:
# whole-row 1D segments with ~64-128KB per CTA and loads issued before stores
# reach ~1.0-1.1 TB/s for this broadcast-add (vs ~20 GB/s for a 2D
# block-pointer broadcast tile with boundary checks on both operands).
_TARGET_CTA_BYTES = 131072
# Hard cap for one 1D segment so that in+out DSM of a single CTA stays well
# below the GCU DSM limit even before row grouping is applied.
_MAX_SEGMENT_BYTES = 131072
_MAX_RPP = 8


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def beam_search_score_kernel(log_probs, beam_scores):
    # Beam Search Score: Add cumulative log probability with new token log probability
    # This is the core operation in beam search: new_score = old_score + log_prob(token)
    return log_probs + beam_scores


@triton.jit
def _bss_seg(lp_ptr, bs_ptr, out_ptr, row, V, offs, EVEN_V: tl.constexpr):
    base = row * V
    if EVEN_V:
        lv = tl.load(lp_ptr + base + offs)
        bs = tl.load(bs_ptr + row)
        tl.store(out_ptr + base + offs, lv + bs)
    else:
        vmask = offs < V
        lv = tl.load(lp_ptr + base + offs, mask=vmask)
        bs = tl.load(bs_ptr + row)
        tl.store(out_ptr + base + offs, lv + bs, mask=vmask)


@triton.jit
def _bss_seg_guard(lp_ptr, bs_ptr, out_ptr, row, n_rows, V, offs, EVEN_V: tl.constexpr):
    row_ok = row < n_rows
    base = row * V
    if EVEN_V:
        vmask = tl.broadcast_to(row_ok, (offs.shape[0],))
        lv = tl.load(lp_ptr + base + offs, mask=vmask)
        bs = tl.load(bs_ptr + row, mask=row_ok, other=0)
        tl.store(out_ptr + base + offs, lv + bs, mask=vmask)
    else:
        vmask = (offs < V) & row_ok
        lv = tl.load(lp_ptr + base + offs, mask=vmask)
        bs = tl.load(bs_ptr + row, mask=row_ok, other=0)
        tl.store(out_ptr + base + offs, lv + bs, mask=vmask)


@libentry()
@triton.jit
def _beam_search_score_seg_kernel(
    lp_ptr,
    bs_ptr,
    out_ptr,
    n_rows,
    V,
    BLOCK_V: tl.constexpr,
    RPP: tl.constexpr,
    EVEN_V: tl.constexpr,
    EVEN_R: tl.constexpr,
    ONE_TILE: tl.constexpr,
):
    # One CTA covers RPP consecutive rows; each row is one contiguous 1D
    # segment of the broadcast add out[row, :] = lp[row, :] + bs[row].
    # beam_scores is consumed as a per-row scalar load (no broadcast
    # materialization of a stride-0 vector, which the GCU backend charges
    # per element).
    pid = tl.program_id(0)
    row0 = pid * RPP

    if ONE_TILE:
        offs = tl.arange(0, BLOCK_V)
        if EVEN_R:
            if EVEN_V:
                # fast path: issue all loads first, then all stores
                lv0 = tl.load(lp_ptr + (row0 + 0) * V + offs)
                bs0 = tl.load(bs_ptr + row0 + 0)
                if RPP >= 2:
                    lv1 = tl.load(lp_ptr + (row0 + 1) * V + offs)
                    bs1 = tl.load(bs_ptr + row0 + 1)
                if RPP >= 4:
                    lv2 = tl.load(lp_ptr + (row0 + 2) * V + offs)
                    bs2 = tl.load(bs_ptr + row0 + 2)
                    lv3 = tl.load(lp_ptr + (row0 + 3) * V + offs)
                    bs3 = tl.load(bs_ptr + row0 + 3)
                if RPP >= 8:
                    lv4 = tl.load(lp_ptr + (row0 + 4) * V + offs)
                    bs4 = tl.load(bs_ptr + row0 + 4)
                    lv5 = tl.load(lp_ptr + (row0 + 5) * V + offs)
                    bs5 = tl.load(bs_ptr + row0 + 5)
                    lv6 = tl.load(lp_ptr + (row0 + 6) * V + offs)
                    bs6 = tl.load(bs_ptr + row0 + 6)
                    lv7 = tl.load(lp_ptr + (row0 + 7) * V + offs)
                    bs7 = tl.load(bs_ptr + row0 + 7)
                tl.store(out_ptr + (row0 + 0) * V + offs, lv0 + bs0)
                if RPP >= 2:
                    tl.store(out_ptr + (row0 + 1) * V + offs, lv1 + bs1)
                if RPP >= 4:
                    tl.store(out_ptr + (row0 + 2) * V + offs, lv2 + bs2)
                    tl.store(out_ptr + (row0 + 3) * V + offs, lv3 + bs3)
                if RPP >= 8:
                    tl.store(out_ptr + (row0 + 4) * V + offs, lv4 + bs4)
                    tl.store(out_ptr + (row0 + 5) * V + offs, lv5 + bs5)
                    tl.store(out_ptr + (row0 + 6) * V + offs, lv6 + bs6)
                    tl.store(out_ptr + (row0 + 7) * V + offs, lv7 + bs7)
            else:
                _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 0, V, offs, EVEN_V)
                if RPP >= 2:
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 1, V, offs, EVEN_V)
                if RPP >= 4:
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 2, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 3, V, offs, EVEN_V)
                if RPP >= 8:
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 4, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 5, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 6, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 7, V, offs, EVEN_V)
        else:
            for r in range(0, RPP):
                _bss_seg_guard(
                    lp_ptr, bs_ptr, out_ptr, row0 + r, n_rows, V, offs, EVEN_V
                )
    else:
        n_cb = tl.cdiv(V, BLOCK_V)
        for cb in range(0, n_cb):
            offs = cb * BLOCK_V + tl.arange(0, BLOCK_V)
            if EVEN_R:
                _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 0, V, offs, EVEN_V)
                if RPP >= 2:
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 1, V, offs, EVEN_V)
                if RPP >= 4:
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 2, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 3, V, offs, EVEN_V)
                if RPP >= 8:
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 4, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 5, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 6, V, offs, EVEN_V)
                    _bss_seg(lp_ptr, bs_ptr, out_ptr, row0 + 7, V, offs, EVEN_V)
            else:
                for r in range(0, RPP):
                    _bss_seg_guard(
                        lp_ptr, bs_ptr, out_ptr, row0 + r, n_rows, V, offs, EVEN_V
                    )


def _beam_search_score_rows_dispatch(log_probs, beam_scores, out):
    # Shape-bucketed launch heuristics for _beam_search_score_seg_kernel,
    # calibrated by A/B probes on GCU400 (see knowledge/case_log.md).
    # Autotuner exemption: this op runs in ~3-9us, where the per-call Python
    # key lookup of libtuner costs a measurable fraction of the op itself;
    # constexpr specialization (BLOCK_V/RPP/EVEN_*/ONE_TILE + num_warps)
    # already gives one compiled kernel per shape class.
    V = log_probs.shape[-1]
    n_rows = log_probs.numel() // V
    itemsize = log_probs.element_size()
    v_pow2 = triton.next_power_of_2(V)
    block_v = min(v_pow2, _MAX_SEGMENT_BYTES // itemsize)
    bytes_per_row = block_v * itemsize
    rpp = _TARGET_CTA_BYTES // bytes_per_row
    rpp = max(1, min(_MAX_RPP, rpp))
    rpp = min(rpp, max(1, triton.next_power_of_2(n_rows) // 16))
    num_ctas = triton.cdiv(n_rows, rpp)
    if num_ctas > _GRID_X_MAX:
        return False
    bytes_per_cta = rpp * bytes_per_row
    if bytes_per_cta <= 32768:
        num_warps = 1
    elif itemsize <= 2:
        num_warps = 2
    else:
        num_warps = 4
    with torch_device_fn.device(log_probs.device):
        _beam_search_score_seg_kernel[(num_ctas, 1, 1)](
            log_probs,
            beam_scores,
            out,
            n_rows,
            V,
            BLOCK_V=block_v,
            RPP=rpp,
            EVEN_V=(V % block_v == 0),
            EVEN_R=(n_rows % rpp == 0),
            ONE_TILE=(block_v >= V),
            num_warps=num_warps,
        )
    return True


def _beam_search_score_fast_path(log_probs, beam_scores, out):
    # Eligibility: contiguous row-major log_probs whose rows are addressed by
    # a flat row id, and a contiguous beam_scores that broadcasts to
    # log_probs.shape with exactly one value per row (numel == n_rows, first
    # dim == n_rows, all trailing dims == 1) so the per-row scalar load
    # bs_ptr[row] matches torch's broadcast value. beam_scores must share
    # log_probs' dtype so the in-kernel add keeps torch's type-promotion
    # semantics. Everything else falls back to the pointwise kernel.
    if log_probs.dtype != beam_scores.dtype:
        return False
    if log_probs.dtype == torch.bool:
        return False
    if not log_probs.is_contiguous() or not beam_scores.is_contiguous():
        return False
    numel = log_probs.numel()
    if numel == 0:
        return False
    if numel > 2147483647:
        return False
    if beam_scores.dim() == 0:
        return False
    if beam_scores.dim() > log_probs.dim():
        return False
    V = log_probs.shape[-1]
    n_rows = numel // V
    if beam_scores.numel() != n_rows:
        return False
    bs_shape = beam_scores.shape
    if bs_shape[0] != n_rows or any(dim != 1 for dim in bs_shape[1:]):
        return False
    try:
        if torch.broadcast_shapes(beam_scores.shape, log_probs.shape) != tuple(
            log_probs.shape
        ):
            return False
    except RuntimeError:
        return False
    return _beam_search_score_rows_dispatch(log_probs, beam_scores, out)


def _bss_promote_dtype(log_probs, beam_scores):
    _, dtype = type_promotion(
        log_probs,
        beam_scores,
        type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )
    return dtype


def beam_search_score(log_probs, beam_scores):
    """Compute Beam Search scores by adding cumulative scores with log probabilities.

    In Beam Search, each candidate's score is computed by adding the cumulative
    log probability with the log probability of the new token.

    Args:
        log_probs: Log probabilities for new tokens, shape [batch_size, vocab_size]
        beam_scores: Cumulative beam scores, shape [batch_size] or [batch_size, 1]

    Returns:
        Updated beam scores with shape [batch_size, vocab_size]
    """
    logger.debug("GEMS_ENFLAME BEAM_SEARCH_SCORE")
    logger.debug("GEMS_ENFLAME BEAM_SEARCH_SCORE_")

    log_probs_ndim = log_probs.dim()
    beam_scores_ndim = beam_scores.dim()

    if beam_scores_ndim < log_probs_ndim and beam_scores_ndim == 1:
        beam_scores = beam_scores.unsqueeze(-1)

    if torch.is_tensor(log_probs) and log_probs.device.type == "cpu":
        return log_probs + beam_scores

    if log_probs.numel() == 0:
        return torch.empty(
            log_probs.shape,
            dtype=_bss_promote_dtype(log_probs, beam_scores),
            device=log_probs.device,
        )

    out = torch.empty(
        log_probs.shape,
        dtype=_bss_promote_dtype(log_probs, beam_scores),
        device=log_probs.device,
    )
    if _beam_search_score_fast_path(log_probs, beam_scores, out):
        return out
    return beam_search_score_kernel(log_probs, beam_scores, out0=out)


def beam_search_score_(log_probs, beam_scores):
    """In-place version of beam_search_score.

    Computes Beam Search scores in-place, modifying log_probs directly.
    """

    log_probs_ndim = log_probs.dim()
    beam_scores_ndim = beam_scores.dim()

    if beam_scores_ndim < log_probs_ndim and beam_scores_ndim == 1:
        beam_scores = beam_scores.unsqueeze(-1)

    if torch.is_tensor(log_probs) and log_probs.device.type == "cpu":
        log_probs.copy_(log_probs + beam_scores)
        return log_probs

    if log_probs.numel() > 0 and _beam_search_score_fast_path(
        log_probs, beam_scores, log_probs
    ):
        return log_probs
    return beam_search_score_kernel(log_probs, beam_scores, out0=log_probs)
