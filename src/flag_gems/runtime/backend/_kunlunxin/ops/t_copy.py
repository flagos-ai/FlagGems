import logging

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

import flag_gems

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

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
def _t_copy_pw(src):
    return src


def _launch_t_copy_kernel(inp: torch.Tensor, out: torch.Tensor):
    if inp.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"t_copy kernels require {flag_gems.device} tensors")
    assert inp.dtype == out.dtype, "dtype mismatch between input and output"

    dim = inp.dim()
    if dim > 2:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")
    if inp.numel() == 0:
        return

    if dim == 2:
        M, N = inp.shape  # input dims
        # out should be (N, M)
        assert (
            out.dim() == 2 and out.shape[0] == N and out.shape[1] == M
        ), "Output shape must be (input.size(1), input.size(0)) for t_copy"
        src = inp.transpose(0, 1)  # shape (N, M), arbitrary strides
    else:
        # 0-D / 1-D t_copy is an identity copy.
        assert out.numel() == inp.numel(), "Output size mismatch for t_copy"
        src = inp

    # tle takes the whole transpose when it can: the 2-D copy, whose two sides
    # disagree about which run is contiguous, is exactly what the on-chip
    # transposed tile covers (and it matches the vendor strided-copy engine on
    # KL3 while a pointwise transpose was ~100x slower); 0-D/1-D and
    # already-contiguous views ride the TMA tile path. What tle cannot express
    # (e.g. an 8-byte dtype whose inner run is not unit-strided) goes to the
    # pointwise copy kernel instead of `torch.ops.aten._copy_from`, which
    # bypasses gems and dispatches to the vendor fallback.
    if tle_copy(src, out):
        return
    _t_copy_pw(src, out0=out)


def t_copy_out(
    input: torch.Tensor,
    out: torch.Tensor,
    memory_format: torch.memory_format | None = None,
):
    logger.debug("GEMS_KUNLUNXIN T_COPY_OUT")
    _launch_t_copy_kernel(input, out)
    return out


def t_copy(input: torch.Tensor, memory_format: torch.memory_format | None = None):
    logger.debug("GEMS_KUNLUNXIN T_COPY")
    dim = input.dim()
    if dim == 0:
        out = torch.empty((), dtype=input.dtype, device=input.device)
    elif dim == 1:
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
    elif dim == 2:
        M, N = input.shape
        out = torch.empty(
            (N, M),
            dtype=input.dtype,
            device=input.device,
            memory_format=torch.contiguous_format,
        )
    else:
        raise RuntimeError("t_copy expects a tensor with <= 2 dims")
    _launch_t_copy_kernel(input, out)
    return out
