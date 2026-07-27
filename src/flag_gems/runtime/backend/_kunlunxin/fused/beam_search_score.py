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

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _beam_search_score_kernel(
    log_probs,
    beam_scores,
    output,
    vocab_size: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // TILES_PER_ROW
    col_offsets = (pid % TILES_PER_ROW) * BLOCK + tl.arange(0, BLOCK)
    mask = col_offsets < vocab_size
    offsets = row * vocab_size + col_offsets

    log_probs_values = tl.load(log_probs + offsets, mask=mask)
    beam_score = tl.load(beam_scores + row)
    tl.store(output + offsets, log_probs_values + beam_score, mask=mask)


def _launch_beam_search_score(log_probs, beam_scores, output):
    if log_probs.ndim != 2 or beam_scores.ndim not in (1, 2):
        raise ValueError("beam_search_score expects 2D log_probs and 1D or 2D beam_scores")
    if beam_scores.numel() != log_probs.shape[0]:
        raise ValueError("beam_scores must contain one score per batch entry")
    if not log_probs.is_contiguous() or not beam_scores.is_contiguous():
        raise ValueError("beam_search_score expects contiguous inputs on Kunlunxin")

    batch_size, vocab_size = log_probs.shape
    block = 4096
    tiles_per_row = triton.cdiv(vocab_size, block)
    grid = (batch_size * tiles_per_row,)
    _beam_search_score_kernel[grid](
        log_probs,
        beam_scores,
        output,
        vocab_size=vocab_size,
        TILES_PER_ROW=tiles_per_row,
        BLOCK=block,
        num_warps=4,
    )
    return output


def beam_search_score_(log_probs, beam_scores):
    logger.debug("GEMS_KUNLUNXIN BEAM_SEARCH_SCORE_")
    return _launch_beam_search_score(log_probs, beam_scores, log_probs)
