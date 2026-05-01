"""CTC loss forward (reference path via ``aten::_ctc_loss`` on CPU).

PyTorch applies ``mean`` reduction as ``(nll / target_lengths.float()).mean()``
over the batch, where ``nll`` is the first output of ``aten::_ctc_loss``. Running
that op on detached CPU tensors avoids recursive dispatch when FlagGems registers
the same dispatch key on the original device.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_REDUCTION_NONE = 0
_REDUCTION_MEAN = 1
_REDUCTION_SUM = 2


def _cpu_log_probs_for_ctc(log_probs: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if log_probs.dtype in (torch.float16, torch.bfloat16):
        return log_probs.detach().cpu().to(torch.float32), True
    return log_probs.detach().cpu(), False


def _tensor_to_int_list(t: torch.Tensor) -> list[int]:
    return t.detach().cpu().reshape(-1).tolist()


def _apply_reduction(
    nll: torch.Tensor,
    target_lengths_float: torch.Tensor,
    reduction: int,
) -> torch.Tensor:
    if reduction == _REDUCTION_NONE:
        return nll
    if reduction == _REDUCTION_MEAN:
        return (nll / target_lengths_float).mean()
    if reduction == _REDUCTION_SUM:
        return nll.sum()
    raise RuntimeError(f"ctc_loss: invalid reduction {reduction}")


def _restore_out(
    out: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    promoted: bool,
) -> torch.Tensor:
    if promoted:
        return out.to(device=device, dtype=dtype)
    return out.to(device=device)


def _ctc_forward(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: list[int],
    target_lengths: list[int],
    blank: int,
    reduction: int,
    zero_infinity: bool,
) -> torch.Tensor:
    lp_cpu, promoted = _cpu_log_probs_for_ctc(log_probs)
    tgt_cpu = targets.detach().cpu()
    tl_f = torch.tensor(target_lengths, dtype=torch.float32, device="cpu")

    nll, _ = torch.ops.aten._ctc_loss.default(
        lp_cpu,
        tgt_cpu,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    )
    out_cpu = _apply_reduction(nll, tl_f, reduction)
    return _restore_out(
        out_cpu, device=log_probs.device, dtype=log_probs.dtype, promoted=promoted
    )


def ctc_loss_intlist(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: list[int],
    target_lengths: list[int],
    blank: int = 0,
    reduction: int = 1,
    zero_infinity: bool = False,
) -> torch.Tensor:
    """``aten::ctc_loss.IntList`` — CTC loss with length arguments as Python lists."""
    logger.debug("GEMS CTC_LOSS IntList")
    return _ctc_forward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction,
        zero_infinity,
    )


def ctc_loss_tensor(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: int = 1,
    zero_infinity: bool = False,
) -> torch.Tensor:
    """``aten::ctc_loss.Tensor`` — CTC loss with length arguments as 1-D tensors."""
    logger.debug("GEMS CTC_LOSS Tensor")
    il = _tensor_to_int_list(input_lengths)
    tl = _tensor_to_int_list(target_lengths)
    return _ctc_forward(
        log_probs, targets, il, tl, blank, reduction, zero_infinity
    )
