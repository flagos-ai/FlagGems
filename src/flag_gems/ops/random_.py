import logging

import torch

from flag_gems.ops.uniform import uniform_

logger = logging.getLogger(__name__)


def _default_random_to(dtype: torch.dtype) -> int:
    if dtype == torch.bool:
        return 2
    if dtype.is_floating_point:
        return (1 << 24) + 1
    return torch.iinfo(dtype).max


def random_(self: torch.Tensor, from_=0, to=None, *, generator=None):
    logger.debug("GEMS RANDOM_")
    if self.is_complex():
        raise RuntimeError("random_ is not implemented for complex tensors")

    if to is None:
        if from_ != 0:
            to = int(from_)
            from_ = 0
        else:
            to = _default_random_to(self.dtype)

    low = int(from_)
    high = int(to)
    if high <= low:
        raise RuntimeError("random_ expects 'from' to be less than 'to'")

    if self.dtype == torch.bool:
        tmp = torch.empty(self.shape, device=self.device, dtype=torch.float32)
        uniform_(tmp, 0.0, 2.0, generator=generator)
        tmp.floor_()
        self.copy_(tmp > 0.0)
        return self

    tmp = torch.empty(self.shape, device=self.device, dtype=torch.float32)
    uniform_(tmp, float(low), float(high), generator=generator)
    tmp.floor_()
    tmp.clamp_(min=low, max=high - 1)
    self.copy_(tmp.to(dtype=self.dtype))
    return self
