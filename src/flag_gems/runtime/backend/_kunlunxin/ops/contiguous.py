import logging
import os

import torch
from _kunlunxin.ops.copy import copy, copy_slice

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def contiguous(inp, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    logger.debug("GEMS CONTIGUOUS")
    if inp.is_contiguous(memory_format=memory_format):
        return inp
    out = torch.empty_like(inp, memory_format=memory_format)
    if "TRITONXPU_IS_SCATTER_SLICE" in os.environ:
        return copy_slice(inp, out0=out)
    else:
        return copy(inp, out0=out)
