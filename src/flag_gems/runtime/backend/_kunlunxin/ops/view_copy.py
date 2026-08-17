import logging

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# view_copy is a pure contiguous copy (reshape + materialize). The generic
# KernelGen kernel (src/flag_gems/ops/view_copy.py) launches a raw
# BLOCK_SIZE=1024 tile with no libentry/unroll/tuned config, which
# underutilizes the XPU badly: baseline is 0.04-0.17x torch on large shapes
# and blows up to 200-1400ms (0.000x) on some dtype/shape combos (see
# ir-view_copy-dev4.log tail). Fix: route the flat copy through the tuned
# pointwise_dynamic path (larger buffer + unroll8 + autoGrid + libentry
# caching), like copy.py. Being a pure memory-bound copy, vectorization must
# stay OPEN (isCloseVectorization=False): with vec CLOSED, fp16 1D large
# collapses to ~1465ms (fp16 fails to vectorize), same signal as addcmul.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    unroll_num=8,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _view_copy_flat(src):
    return src


def view_copy(x: torch.Tensor, size) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN VIEW_COPY")

    # Handle SymInt[] - convert to tuple of ints
    if isinstance(size, torch.SymInt):
        size = (int(size),)
    elif isinstance(size, (list, tuple)):
        size = tuple(int(s) if isinstance(s, torch.SymInt) else s for s in size)

    n_elements = x.numel()

    # Handle -1 (infer this dimension)
    if -1 in size:
        if size.count(-1) > 1:
            raise RuntimeError(f"view_copy: only one dimension can be -1, got {size}")
        target_numel_except_minus1 = 1
        for s in size:
            if s != -1:
                target_numel_except_minus1 *= s
        inferred_dim = n_elements // target_numel_except_minus1
        size = tuple(inferred_dim if s == -1 else s for s in size)

    # Validate total number of elements matches
    target_numel = 1
    for s in size:
        target_numel *= s
    if n_elements != target_numel:
        raise RuntimeError(
            f"view_copy: cannot reshape tensor of size {n_elements} into shape {size}"
        )

    if n_elements == 0:
        return torch.empty(size, dtype=x.dtype, device=x.device)

    out = torch.empty(size, dtype=x.dtype, device=x.device)

    # Flatten both to 1D so the elementwise copy sees identical shapes (source
    # and target may differ in rank but share numel). Contiguous flat copy is
    # what the tuned config accelerates.
    src = x.contiguous() if not x.is_contiguous() else x
    _view_copy_flat(src.view(-1), out0=out.view(-1))
    return out
