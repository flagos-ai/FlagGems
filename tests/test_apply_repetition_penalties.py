import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _init_vllm():
    if not torch.cuda.is_available():
        return None, False
    try:
        from vllm._custom_ops import apply_repetition_penalties as fn

        t, m = torch.randn(2, 1024, device="cuda"), torch.zeros(
            2, 1024, dtype=torch.bool, device="cuda"
        )
        fn(t, m, m, torch.full((2,), 1.2, device="cuda"))
        return fn, True
    except (ImportError, RuntimeError):

        def fallback(logits, pm, om, pens):
            for i in range(logits.shape[0]):
                m = pm[i] | om[i]
                logits[i][m] = torch.where(
                    logits[i][m] > 0, logits[i][m] / pens[i], logits[i][m] * pens[i]
                )

        return fallback, True


_vllm_fn, _VLLM_OK = _init_vllm()

_REP_PENALTY_CFG = {
    "shapes": [
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (8, 4096),
        (16, 4096),
        (32, 1024),
        (8, 8192),
    ],
    "penalties": [1.0, 1.2, 1.5],
    "device": torch.device("cuda:0"),
}


@pytest.mark.apply_repetition_penalties
@pytest.mark.skipif(
    not _VLLM_OK or not torch.cuda.is_available(), reason="need VLLM+CUDA"
)
@pytest.mark.parametrize("shape", _REP_PENALTY_CFG["shapes"])
@pytest.mark.parametrize("penalty", _REP_PENALTY_CFG["penalties"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mask_mode", ["random", "empty"])
def test_apply_repetition_penalty(shape, penalty, dtype, mask_mode):
    device = _REP_PENALTY_CFG["device"]

    logits = torch.randn(shape, dtype=dtype, device=device).contiguous()
    logits_ori = logits.clone()

    if mask_mode == "random":
        prompt_mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
        output_mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    else:
        prompt_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        output_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    penalties = torch.full((shape[0],), penalty, dtype=dtype, device=device)

    logits_vllm = logits.clone()
    _vllm_fn(logits_vllm, prompt_mask.clone(), output_mask.clone(), penalties.clone())
    ref = utils.to_reference(logits_vllm, True).to(dtype)

    with flag_gems.use_gems():
        flag_gems.apply_repetition_penalties(
            logits, prompt_mask, output_mask, penalties
        )
    res = utils.to_reference(logits, True).to(dtype)

    utils.gems_assert_close(res, ref, dtype)

    has_mask = (prompt_mask | output_mask).any().item()
    should_modify = has_mask and penalty != 1.0
    if should_modify:
        assert not torch.equal(
            utils.to_reference(logits, True), utils.to_reference(logits_ori, True)
        ), "In-place not working"
    elif mask_mode == "empty":
        utils.gems_assert_close(
            res, utils.to_reference(logits_ori, True).to(dtype), dtype
        )
