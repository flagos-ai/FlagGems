# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""special_logit enflame 副本：flat grid-stride 家族制式（sigmoid/square 同款）。

公版 one-tile-per-program（grid=cdiv(n,1024)）在 enflame 2^29 元素 core bench
形状上请求 grid.x=524288，超 GCU 硬件上限 65535 直接 OutOfResources（已知
grid.x>65535 聚类）。本副本改 48-SIP 持久 grid-stride + BLOCK 阶梯，任意 n 均
不超过 48 个 CTA；小形状/非连续/非 fp16-bf16-fp32/int32 溢出域回退公版实现，
语义与公版逐位一致（NaN 透传、eps clamp、out= 校验全保留）。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
INT32_MAX = torch.iinfo(torch.int32).max


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def special_logit_flat_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    eps,
    HAS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        offsets = block_id * BLOCK + arange
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        if HAS_EPS:
            lo = eps
            hi = 1.0 - eps
            x_f32 = tl.where(
                x_f32 != x_f32, x_f32, tl.minimum(tl.maximum(x_f32, lo), hi)
            )

        y = tl.log(x_f32 / (1.0 - x_f32))
        tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def _choose_block(n_elements, itemsize):
    # 1R1W pointwise ladder validated on this kernel family (sigmoid/square):
    # tiny shapes want the exact next_pow2 tile; 2-byte large shapes peak at
    # 131072 (fp16 536M), 4-byte mid shapes at 32768 and 536M at 65536.
    if n_elements <= 8192:
        return max(256, triton.next_power_of_2(n_elements))
    if n_elements <= 65536:
        return 8192
    if n_elements <= 524288:
        return 32768
    if itemsize <= 2:
        return 131072 if n_elements > (1 << 28) else 65536
    return 32768 if n_elements <= (1 << 25) else 65536


def _launch_flat(input, result, n_elements, eps):
    BLOCK = _choose_block(n_elements, input.element_size())
    if triton.cdiv(n_elements, BLOCK) * BLOCK > INT32_MAX:
        return False
    grid_size = min(triton.cdiv(n_elements, BLOCK), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (atan2_/sinh/rad2deg/square precedents for this toolchain family);
    # retrying the launch recompiles and succeeds.
    for attempt in range(3):
        try:
            with torch_device_fn.device(input.device):
                special_logit_flat_kernel[(grid_size,)](
                    input,
                    result,
                    n_elements,
                    eps if eps is not None else 0.0,
                    HAS_EPS=(eps is not None),
                    BLOCK=BLOCK,
                    num_warps=2,
                )
            return True
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "special_logit flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _special_logit_impl(input: torch.Tensor, eps=None, out: torch.Tensor = None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not input.is_floating_point():
        raise TypeError("special_logit expected a floating point tensor as input")

    if eps is not None:
        eps = float(eps)
        if not (0.0 <= eps <= 0.5):
            raise ValueError("eps must be in the range [0.0, 0.5].")

    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.dtype != input.dtype:
            raise TypeError("For special_logit.out, out.dtype must match input.dtype")
        result = out
    else:
        result = torch.empty_like(input)

    n_elements = result.numel()
    if (
        n_elements > 0
        and input.is_contiguous()
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _launch_flat(input, result, n_elements, eps)
    ):
        return result

    # Parity fallback: the public generic implementation keeps bit-identical
    # semantics for the guarded-out cases (empty, non-contiguous, fp64,
    # int32 offset domain).
    from flag_gems.ops.special_logit import _special_logit_impl as _public_impl

    return _public_impl(input, eps=eps, out=out)


def special_logit(input, eps=None, out=None):
    """Compute the logit of the input tensor.

    Args:
        input: Input tensor with values in [0, 1] (or outside if eps is None)
        eps: Optional epsilon for clamping. If provided, input is clamped to [eps, 1-eps]
        out: Optional output tensor

    Returns:
        Tensor with logit values
    """
    logger.debug("GEMS_ENFLAME SPECIAL_LOGIT")
    return _special_logit_impl(input, eps=eps, out=out)


def special_logit_out(input, eps=None, out=None):
    logger.debug("GEMS_ENFLAME SPECIAL_LOGIT_OUT")
    if out is None:
        raise TypeError("special_logit_out requires an 'out' tensor.")
    return _special_logit_impl(input, eps=eps, out=out)
