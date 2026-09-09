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

import logging

import torch
import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# where(cond, a, b) is a pure memory-bound 3-in/1-out select: it streams a bool
# mask plus two data tensors and writes one. The XPU default CodeGenConfig
# (prefer_1d_tile=True, vectorization OPEN, unroll_num=0) generates a
# scalar-ish inner loop for the mixed-type tl.where (i1/i8 mask combined with
# f16/bf16/f32 data), which collapses large-shape bandwidth to ~250-300 GB/s.
# Measured on a single XPU (fresh compile, [10000, 65536], wall clock):
#
#   config                                      fp16     fp32     bf16
#   default (baseline)                         14.75ms  14.37ms  17.89ms
#   isCloseVectorization=True                  11.89ms  12.59ms  11.89ms
#   isCloseVectorization=True + unroll_num=16    7.36ms   8.17ms   7.36ms
#   unroll_num=8/16 with vectorization OPEN    14.77ms  14.34ms  17.87ms
#
# The two knobs are only effective together: closing vectorization lets the
# backend emit contiguous block DMA for the mixed-type select (same recipe as
# masked_fill.py, since masked_fill == where(mask, value, inp)), and
# unroll_num=16 gives the DMA engine enough in-flight tiles to saturate HBM.
# Ablations that changed nothing (left out to keep the config minimal):
# max_tile_size 1024/2048/4096/8192, buffer_size_limit 2048/4096/8192,
# kunlunAutoGrid, isCloseDtypeConvert -- all reproduce 7.36ms, because with
# unrolled 1d tiles the effective tiling is decided by the unroll factor.
# Ablations that regressed: unroll_num=32 (10.32ms), isCloseInterleave=True
# (8.97ms).
#
# WARNING: do NOT use a non-power-of-two unroll_num here. unroll_num=12 emits a
# kernel that issues a garbage AXI read address and wedges the device (dmesg:
# KL_XID64_AXI_ADDR_ERROR / KL_XID_KERNEL_EXCEPTION, then
# kl3_wait_for_noc_idle() timeout), requiring a soft_reset to recover. This was
# reproduced twice on a healthy card, so treat it as a compiler-side
# constraint, not a transient hardware fault. 8/16/32 are all safe.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseVectorization=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def where_inner(condition, self, other):
    return tl.where(condition, self, other)


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS_KUNLUNXIN WHERE_SELF_OUT")
    result_type = torch.result_type(self, other)
    if out is not None:
        assert (
            out.dtype == result_type
        ), f"Expected out type to be {result_type}, but got {out.dtype}."

    c, a, b = list(
        map(
            lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x),
            (condition, self, other),
        )
    )

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    devices = map(lambda x: x.device, (c, a, b))
    devices = list(filter(lambda k: k.type != "cpu", devices))

    assert len(devices), "CPU only. There seems a mistake to dispatch to here."

    device = devices[0]
    if c.device != device and c.ndim == 0:
        c = c.to(device)
    if a.device != device and a.ndim == 0:
        a = a.to(device)
    if b.device != device and b.ndim == 0:
        b = b.to(device)

    assert (
        len(set(devices)) == 1
    ), f"Expected all tensors to be on the same device, but found at least two devices, {devices}"
    assert (
        c.dtype == torch.bool
    ), f"where expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"

    if out is None:
        out_shape = torch.broadcast_shapes(c.shape, a.shape, b.shape)
        out = torch.empty(out_shape, dtype=result_type, device=device)

    ndim = max(c.ndim, a.ndim, b.ndim)
    where_inner.instantiate(ndim)
    where_inner(c, a, b, out0=out)
    return out


def where_self(condition, self, other):
    logger.debug("GEMS_KUNLUNXIN WHERE_SELF")
    return where_self_out(condition, self, other)


def where_scalar_self(condition, self, other):
    logger.debug("GEMS_KUNLUNXIN WHERE_SCALAR_SELF")
    return where_self_out(condition, self, other)


def where_scalar_other(condition, self, other):
    logger.debug("GEMS_KUNLUNXIN WHERE_SCALAR_OTHER")
    return where_self_out(condition, self, other)
