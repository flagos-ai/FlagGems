import logging

import torch

logger = logging.getLogger(__name__)


def _assert_async(tensor: torch.Tensor, msg: str = "Assertion failed"):
    logger.debug("GEMS_KUNLUNXIN ASSERT_ASYNC")
    if tensor.numel() != 1:
        raise RuntimeError(
            f"Boolean value of Tensor with shape {list(tensor.shape)} is ambiguous"
        )
    # The Triton XPU backend lowers `tl.device_assert` to a no-op, so the
    # falsy branch never fires on-device. Read the single element back to the
    # host (an implicit stream sync) and raise if it is falsy, restoring the
    # ATen semantics without the cost of a Triton launch + scratch round-trip.
    if not tensor.item():
        raise RuntimeError(msg)
