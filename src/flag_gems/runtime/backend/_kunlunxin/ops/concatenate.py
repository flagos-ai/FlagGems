# Kunlunxin (XPU) override of concatenate.
#
# `concatenate` is a pure alias of `cat` (aten::concatenate(Tensor[], int dim=0)).
# kunlunxin already ships a tuned `cat` override (`_kunlunxin/ops/cat.py`, a
# StridedBuffer + pointwise_dynamic `copy_func` path with bounded tiles +
# autoGrid + libentry cache), but `concatenate` was NOT overridden. The generic
# `ops/concatenate.py` binds `from flag_gems.ops.cat import cat` at import time,
# so it always calls the GENERIC cat whose hand-written raw @triton.jit
# `cat_copy_func_kernel_4` (fixed BLOCK, no @libentry caching) recompiles per
# shape/launch -> the IR dump (`harness/perf_ir_3/ir-concatenate-dev7.log`,
# 482K lines / 512KB) shows the module explosion -> discrete/slow on XPU.
#
# Fix: route `concatenate` through the exact same tuned kunlunxin `cat`. Zero
# algorithm change (byte-identical to the covered cat path) -> zero correctness
# risk. Same class as the cat_out uncovered-variant fix.
import logging
from typing import List, Tuple, Union

import torch

from .cat import cat

logger = logging.getLogger(__name__)


def concatenate(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CONCATENATE")
    return cat(A, dim=dim)
