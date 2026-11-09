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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
BLOCK_B = 4
BLOCK_V = 4096


@libentry()
@triton.jit(do_not_specialize=["B", "V"])
def beam_search_score_kernel(
    LogProbs,
    BeamScores,
    Out,
    B,
    V,
    SAME_SHAPE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles_v = tl.cdiv(V, BLOCK_V)
    total_tiles = tl.cdiv(B, BLOCK_B) * num_tiles_v
    for tile in range(pid, total_tiles, num_pids):
        tile_b = tile // num_tiles_v
        tile_v = tile % num_tiles_v
        rows = tile_b * BLOCK_B + tl.arange(0, BLOCK_B)
        cols = tile_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = (rows[:, None] < B) & (cols[None, :] < V)
        offsets = rows[:, None] * V + cols[None, :]
        log_probs = tl.load(LogProbs + offsets, mask=mask, other=0.0)
        if SAME_SHAPE:
            beam_scores = tl.load(BeamScores + offsets, mask=mask, other=0.0)
            result = log_probs + beam_scores
        else:
            beam_scores = tl.load(BeamScores + rows, mask=rows < B, other=0.0)
            result = log_probs + beam_scores[:, None]
        tl.store(Out + offsets, result, mask=mask)


def _normalize(log_probs, beam_scores):
    if beam_scores.dim() < log_probs.dim() and beam_scores.dim() == 1:
        beam_scores = beam_scores.unsqueeze(-1)
    return beam_scores, beam_scores.numel() == log_probs.numel()


def _launch(log_probs, beam_scores, out, same_shape):
    B, V = log_probs.shape
    total_tiles = triton.cdiv(B, BLOCK_B) * triton.cdiv(V, BLOCK_V)
    grid = (min(total_tiles, NUM_SIPS * 2),)
    with torch_device_fn.device(log_probs.device):
        beam_search_score_kernel[grid](
            log_probs,
            beam_scores,
            out,
            B,
            V,
            BLOCK_B=BLOCK_B,
            BLOCK_V=BLOCK_V,
            SAME_SHAPE=same_shape,
            num_warps=2,
        )
    return out


def beam_search_score(log_probs, beam_scores):
    logger.debug("GEMS_ENFLAME BEAM_SEARCH_SCORE")
    beam_scores, same_shape = _normalize(log_probs, beam_scores)
    if log_probs.device.type == "cpu":
        return log_probs + beam_scores
    if beam_scores.numel() not in (log_probs.shape[0], log_probs.numel()):
        return log_probs + beam_scores
    out = torch.empty_like(log_probs)
    return _launch(log_probs, beam_scores, out, same_shape)


def beam_search_score_(log_probs, beam_scores):
    logger.debug("GEMS_ENFLAME BEAM_SEARCH_SCORE_")
    beam_scores, same_shape = _normalize(log_probs, beam_scores)
    if log_probs.device.type == "cpu":
        log_probs.copy_(log_probs + beam_scores)
        return log_probs
    if beam_scores.numel() not in (log_probs.shape[0], log_probs.numel()):
        log_probs.copy_(log_probs + beam_scores)
        return log_probs
    return _launch(log_probs, beam_scores, log_probs, same_shape)
