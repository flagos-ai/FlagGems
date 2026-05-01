import random
import time

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

random.seed(time.time() // 100)

_RED_INT = {"none": 0, "mean": 1, "sum": 2}


@pytest.mark.ctc_loss
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("zero_infinity", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ctc_loss_forward(reduction, zero_infinity, dtype):
    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    device = flag_gems.device
    t_steps, n_batch, n_cls = 32, 4, 20
    max_target = 8
    log_probs = torch.randn(
        t_steps, n_batch, n_cls, device=device, dtype=dtype
    ).log_softmax(2)
    targets = torch.randint(
        1, n_cls, (n_batch, max_target), dtype=torch.long, device=device
    )
    input_lengths = torch.full((n_batch,), t_steps, dtype=torch.long, device=device)
    target_lengths = torch.randint(
        2, max_target + 1, (n_batch,), dtype=torch.long, device=device
    )

    ref_lp = utils.to_reference(log_probs, True)
    ref_tgt = utils.to_reference(targets, False)
    ref_il = utils.to_reference(input_lengths, False)
    ref_tl = utils.to_reference(target_lengths, False)

    ref_out = F.ctc_loss(
        ref_lp,
        ref_tgt,
        ref_il,
        ref_tl,
        blank=0,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )

    with flag_gems.use_gems():
        res_out = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    reduce_dim = 1 if reduction != "none" else max(n_batch, 1)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ctc_loss_intlist_aten(dtype):
    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    device = flag_gems.device
    t_steps, n_batch, n_cls = 24, 3, 15
    max_target = 6
    log_probs = torch.randn(
        t_steps, n_batch, n_cls, device=device, dtype=dtype
    ).log_softmax(2)
    targets = torch.randint(
        1, n_cls, (n_batch, max_target), dtype=torch.long, device=device
    )
    input_lens = [t_steps] * n_batch
    target_lens = torch.randint(
        2, max_target + 1, (n_batch,), dtype=torch.long, device=device
    )
    tl_list = target_lens.cpu().tolist()

    ref_lp = utils.to_reference(log_probs, True)
    ref_tgt = utils.to_reference(targets, False)
    ref_out = torch.ops.aten.ctc_loss.IntList(
        ref_lp,
        ref_tgt,
        input_lens,
        tl_list,
        0,
        _RED_INT["mean"],
        False,
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.ctc_loss.IntList(
            log_probs,
            targets,
            input_lens,
            tl_list,
            0,
            _RED_INT["mean"],
            False,
        )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)
