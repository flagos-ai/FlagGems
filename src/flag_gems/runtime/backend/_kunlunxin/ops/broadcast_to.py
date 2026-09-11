import logging

import torch

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def broadcast_to(x, size):
    """Broadcast ``x`` to ``size``.

    ``aten::broadcast_to`` is a view op (schema ``broadcast_to(Tensor(a) self,
    SymInt[] size) -> Tensor(a)``) and is documented as equivalent to
    ``x.expand(size)``. The generic implementation instead materializes a fresh
    tensor through a hand-written gather kernel whose ``offset // stride % size``
    index math defeats XPU OffsetAnalysis (``offsetState=-1`` discrete
    gather/scatter, ``syncMode=1`` synchronous DMA) and is launch-bound at
    ``BLOCK_SIZE=1024``. Returning the zero-copy expand view matches torch's own
    semantics exactly, avoids that slow path entirely, and is also safer for CUDA
    graph capture (no implicit CPU->device index-array copy).
    """
    logger.debug("GEMS_KUNLUNXIN BROADCAST_TO")

    if not isinstance(size, (list, tuple, torch.Size)):
        raise TypeError("broadcast_to size must be a list/tuple/torch.Size of ints")

    return x.expand(list(size))
