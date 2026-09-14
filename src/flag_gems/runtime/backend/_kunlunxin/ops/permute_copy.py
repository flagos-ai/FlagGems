import logging

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# permute_copy materializes a permuted (strided) view into a fresh contiguous
# tensor. The generic KernelGen kernels (src/flag_gems/ops/permute_copy.py) take
# src_ptr/dst_ptr/n_elements/shapes/strides/perm as ordinary @triton.jit args, so
# Triton specializes each launch on pointer divisibility (=16) and scalar-int
# divisibility. permute_copy allocates a FRESH `out` every call and the XPU
# caching allocator hands back pointers/blocks whose specialization class varies
# across allocations -> each new class triggers a fresh ~150-3700ms XPU compile.
# In a repeated-call / benchmark workload this is a recompilation STORM: the 2D
# shapes measure 6-113ms per call (0.0001-0.02x torch, randomly scattered across
# dtype cells) even though their true kernel is ~0.05-0.2ms.
#
# Fix: express the copy through the tuned pointwise_dynamic path exactly like the
# view_copy sibling. We read the permuted VIEW `x.permute(dims)` (a strided,
# zero-copy view whose shape == out_shape) and write the contiguous `out`;
# pointwise_dynamic auto-generates the strided-input offset math. Crucially the
# CodeGenConfig has kunlunAutoGrid=True + prefer_1d_tile, so ONE shape/pointer-
# independent kernel compiles and is cached (via libentry) -> the storm is gone
# and every shape/dtype is consistent (2D ~0.08ms, no lottery). Vectorization
# stays OPEN (isCloseVectorization=False) for the memory-bound copy, matching
# view_copy/addcmul. Pure identity kernel -> byte-identical output, zero accuracy
# change.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def _permute_copy_pw(src):
    return src


def permute_copy(x: torch.Tensor, dims):
    """Wrapper for aten::permute_copy: return a copy of x with permuted dims."""
    logger.debug("GEMS_KUNLUNXIN PERMUTE_COPY")
    ndim = x.ndim
    if ndim == 0:
        return x.clone()

    dims = [d if d >= 0 else d + ndim for d in dims]
    out_shape = [x.shape[d] for d in dims]

    if x.numel() == 0:
        return torch.empty(out_shape, dtype=x.dtype, device=x.device)

    src = x.contiguous() if not x.is_contiguous() else x
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # x.permute(dims) is a strided zero-copy view with shape == out_shape.
    permuted = src.permute(dims)
    # tle takes it as a TMA tile when the permutation keeps the innermost axis
    # contiguous, and otherwise transposes a 64x64 tile on chip -- the shape
    # XDNN's transpose_021_sdnn_bsp uses. Measured on KL3 for f16 transposes, the
    # pointwise kernel is never the cheaper option here: 124us vs 83us at 8KB,
    # 2.1ms vs 90us at 2MB, 32ms vs 207us at 32MB.
    if tle_copy(permuted, out):
        return out
    _permute_copy_pw(permuted, out0=out)
    return out
