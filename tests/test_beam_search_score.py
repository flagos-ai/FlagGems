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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.beam_search_score
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_beam_search_score(shape, dtype):
    # beam_search_score: log_probs [batch, vocab] + beam_scores [batch] -> [batch, vocab]
    # We test with 2D shapes to ensure broadcasting works correctly
    if len(shape) < 2:
        pytest.skip("beam_search_score requires at least 2D tensors")
    batch_size = shape[0]
    vocab_size = shape[1]

    log_probs = torch.randn(
        batch_size, vocab_size, dtype=dtype, device=flag_gems.device
    )
    beam_scores = torch.randn(batch_size, dtype=dtype, device=flag_gems.device)

    # Reference: PyTorch broadcasting addition
    ref_log_probs = utils.to_reference(log_probs, True)
    ref_beam_scores = utils.to_reference(beam_scores, True)
    ref_out = ref_log_probs + ref_beam_scores.unsqueeze(-1)

    res_out = flag_gems.beam_search_score(log_probs, beam_scores)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.beam_search_score_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_beam_search_score_(shape, dtype):
    if len(shape) < 2:
        pytest.skip("beam_search_score_ requires at least 2D tensors")
    batch_size = shape[0]
    vocab_size = shape[1]

    inp = torch.randn(batch_size, vocab_size, dtype=dtype, device=flag_gems.device)
    beam_scores = torch.randn(batch_size, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_beam_scores = utils.to_reference(beam_scores, True)
    ref_out = ref_inp + ref_beam_scores.unsqueeze(-1)

    res_out = flag_gems.beam_search_score_(inp, beam_scores)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_out, dtype)


@pytest.mark.beam_search_score
@pytest.mark.parametrize("shape", [(3, 0), (0, 5), (0, 0)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_beam_search_score_empty(shape, dtype):
    # (B, 0) used to divide by zero while (0, V) produced a zero-sized grid.
    log_probs = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    beam_scores = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)

    ref_log_probs = utils.to_reference(log_probs, True)
    ref_beam_scores = utils.to_reference(beam_scores, True)
    ref_out = ref_log_probs + ref_beam_scores.unsqueeze(-1)

    res_out = flag_gems.beam_search_score(log_probs, beam_scores)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.beam_search_score
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_beam_search_score_non_contiguous(dtype):
    # A transposed input is not densely row-major, so the row * vocab_size
    # addressing used to read the wrong elements.
    base = torch.randn((8, 6), dtype=dtype, device=flag_gems.device)
    log_probs = base.t()
    assert not log_probs.is_contiguous()
    beam_scores = torch.randn(log_probs.shape[0], dtype=dtype, device=flag_gems.device)

    ref_log_probs = utils.to_reference(log_probs, True)
    ref_beam_scores = utils.to_reference(beam_scores, True)
    ref_out = ref_log_probs + ref_beam_scores.unsqueeze(-1)

    res_out = flag_gems.beam_search_score(log_probs, beam_scores)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.beam_search_score
@pytest.mark.parametrize(
    "log_probs_dtype, beam_scores_dtype",
    [
        (torch.float16, torch.float32),
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_beam_search_score_mixed_dtype(log_probs_dtype, beam_scores_dtype):
    # DEFAULT promotion: the result dtype is promote_types of both inputs, so
    # FP16 log probs plus FP32 beam scores must return FP32.
    log_probs = torch.randn((4, 8), dtype=log_probs_dtype, device=flag_gems.device)
    beam_scores = torch.randn(4, dtype=beam_scores_dtype, device=flag_gems.device)
    expected_dtype = torch.promote_types(log_probs_dtype, beam_scores_dtype)

    ref_log_probs = utils.to_reference(log_probs, True)
    ref_beam_scores = utils.to_reference(beam_scores, True)
    ref_out = ref_log_probs + ref_beam_scores.unsqueeze(-1)

    res_out = flag_gems.beam_search_score(log_probs, beam_scores)

    assert res_out.dtype == expected_dtype
    utils.gems_assert_close(res_out, ref_out, expected_dtype)
