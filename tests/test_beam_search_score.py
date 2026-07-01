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

    with flag_gems.use_gems():
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

    with flag_gems.use_gems():
        res_out = flag_gems.beam_search_score_(inp, beam_scores)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_out, dtype)
