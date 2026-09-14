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

from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# WARNING: do NOT use a non-power-of-two unroll_num here. unroll_num=12 emits a
# kernel that issues a garbage AXI read address and wedges the device (dmesg:
# KL_XID64_AXI_ADDR_ERROR / KL_XID_KERNEL_EXCEPTION, then
# kl3_wait_for_noc_idle() timeout), requiring a soft_reset to recover. This was
# reproduced twice on a healthy card, so treat it as a compiler-side
# constraint, not a transient hardware fault. 8/16/32 are all safe.
#
# masked_fill-style config: isCloseVectorization keeps the mixed i1-mask
# tl.where on the fast path on XPU (masked_fill reaches 0.56x fp32 / 0.36x
# fp16 on 4096x4096, 2026-09-04). without it the bare pointwise_dynamic
# produces discrete access -> catastrophic latency (where_self dtype-balanced
# speedup 0.19 on the acceptance shape set, 2026-09-04 baseline). The
# sub/less_equal_ recipe (isCloseVectorization off) scalarizes the whole where
# kernel to 0.35x/0.19x because the bool CONDITION input is the bottleneck --
# not the tl.where vselect itself (an arithmetic rewrite a*c+b*(1-c) gives the
# same 0.35x; an int8 view of the condition plus sitofp hits a TritonXPU
# vector-widen lowering bug). isCloseVectorization is the only lever that helps.

# 2026-09-14: the "bool CONDITION input is the bottleneck" finding is now
# resolved on the float path. Root cause (verified by probe, evidence:
# artifacts/op-perf-batch-2026-09/evidence/w0-lowering-diagnosis/): an i1
# condition -- whether loaded directly, or produced by cmpi/cmpf -- pins the
# condition component to the scalar layout (three i1 routes all measured
# ~383us on 4096^2 fp32: load-i1+where / i8+cmpf+where / i8+cmpi+where). The
# way out is to keep i1 out of the data flow entirely: view the condition as
# int8 (zero-copy, layout-preserving), widen with a vectorized sitofp, and
# blend arithmetically -- a*c + b*(1-c) is bit-exact for a 0/1 mask (every
# multiply is by 0 or 1). Probe: 131.9us vs 383.5us = 2.9x. This needs the
# triton-fork VSIToFPOpConversion (i8->f32 1:4 segment lowering, 2026-09-14);
# the historical "int8 view + sitofp hits a vector-widen lowering bug" note
# above is exactly that bug, now fixed.
config_openvec_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# Non-float self/other keep the generic i1 tl.where path: the arithmetic blend
# is float-only. isCloseVectorization stays on here so the mixed i1-mask
# tl.where is not scalarized by the vectorizer.
config_closevec_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=8192,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
    config=config_openvec_,
)
@triton.jit
def where_inner(condition, self, other):
    c = condition.to(tl.float32)
    return self * c + other * (1.0 - c)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
    config=config_closevec_,
)
@triton.jit
def where_inner_generic(condition, self, other):
    return tl.where(condition, self, other)


def where_self_out(condition, self, other, out=None):
    logger.debug("GEMS_KUNLUNXIN WHERE_SELF_OUT")
    result_type = torch.result_type(self, other)
    if out is not None:
        assert (
            out.dtype == result_type
        ), f"Expected out type to be {result_type}, but got {out.dtype}."

    c, a, b = condition, self, other

    if a.dtype != result_type:
        a = a.to(result_type)
    if b.dtype != result_type:
        b = b.to(result_type)

    devices = map(lambda x: x.device, (c, a, b))
    devices = list(filter(lambda k: k.type != "cpu", devices))

    assert len(devices), "CPU only. There seems a mistake to dispatch to here."

    device = devices[0]
    if c.device != device and c.ndim == 0:
        c = torch.scalar_tensor(c.item(), dtype=c.dtype, device=device)
    if a.device != device and a.ndim == 0:
        a = torch.scalar_tensor(a.item(), dtype=a.dtype, device=device)
    if b.device != device and b.ndim == 0:
        b = torch.scalar_tensor(b.item(), dtype=b.dtype, device=device)

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
    if result_type.is_floating_point:
        # bool -> int8 view is zero-copy and layout-preserving (same element
        # size); keeps i1 out of the kernel entirely (see config_openvec_ note).
        where_inner.instantiate(ndim)
        where_inner(c.view(torch.int8), a, b, out0=out)
    else:
        where_inner_generic.instantiate(ndim)
        where_inner_generic(c, a, b, out0=out)
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
