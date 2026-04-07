import logging

import torch

logger = logging.getLogger(__name__)


# CTC Loss (Connectionist Temporal Classification):
# Used in sequence-to-sequence models (e.g., speech recognition, OCR).
# Computes the loss between a continuous (unsegmented) time series
# and a target sequence. Registered as FlagGems dispatch entry.
def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    logger.debug("GEMS CTC_LOSS")
    return torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
