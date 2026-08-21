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

_BLOCK_SIZE = 16384
_FLAG_BLOCK_SIZE = 8192
_NUM_WARPS = 8
# Tensors with at least this many elements take the legacy pointwise path:
# on this platform the per-lane scalar ALU math (no vectorized SIMD math for
# elementwise ops) makes the raw-Triton pipeline ~3x slower than the native
# path once tensors exceed a few MB of lanes.
_LARGE_NUMEL = 2 * 1024 * 1024

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def unscale_func(value, inv_scale):
    return (value.to(tl.float32) * inv_scale).to(value.dtype)


@triton.jit
def _unscale_kernel(
    x_ptr,
    inv_scale,
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
    out = (x.to(tl.float32) * tl.load(inv_scale)).to(x.dtype)
    if NEED_MASK:
        tl.store(x_ptr + offs, out, mask=mask)
    else:
        tl.store(x_ptr + offs, out)


@triton.jit
def _non_finite_flag_kernel(
    x_ptr,
    flag_ptr,
    flag_base,
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    else:
        x = tl.load(x_ptr + offs)
    # Non-finite detection via ordered range comparisons only: NaN is not
    # <= MAX nor >= -MAX, so it is caught as well; the unordered "x != x"
    # idiom is NOT selectable by this backend's LLVM for fp32 and must be
    # avoided. inf/nan survive unscaling unchanged, so testing the already
    # scaled in-place data is equivalent to testing the original values.
    nf = ~((x <= 3.4028235e38) & (x >= -3.4028235e38))
    tl.store(flag_ptr + (flag_base + pid), tl.max(tl.where(nf, 1.0, 0.0)))


@triton.jit
def _reduce_flags_kernel(flag_ptr, num_flags, found_ptr, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for base in tl.range(0, num_flags, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        m = offs < num_flags
        acc = tl.maximum(acc, tl.load(flag_ptr + offs, mask=m, other=0.0))
    m = tl.max(acc, axis=0)
    # Only ever raise found_inf to 1.0 (never clear it), preserving the
    # accumulate semantics of found_inf. Masked stores are unreliable on this
    # backend, so the stored VALUE is conditioned instead of the mask.
    idx0 = tl.arange(0, 1)
    cur = tl.load(found_ptr + idx0)
    tl.store(found_ptr + idx0, tl.where(m > 0, 1.0, cur))


def _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale):
    logger.debug("GEMS_KUNLUNXIN AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE")
    large = []
    small = []
    for tensor in tensors:
        n = tensor.numel()
        if n == 0 or not tensor.is_floating_point():
            continue
        if n >= _LARGE_NUMEL:
            large.append(tensor)
        else:
            small.append(tensor)

    # Fast fused path: in-place unscale + per-block non-finite flags in two
    # elementwise launches per tensor plus one tiny op-level reduce.
    total_blocks = sum(triton.cdiv(t.numel(), _FLAG_BLOCK_SIZE) for t in small)
    if total_blocks > 0:
        flags = torch.empty(total_blocks, device=tensors[0].device, dtype=torch.float32)
        base = 0
        for tensor in small:
            n = tensor.numel()
            nb = triton.cdiv(n, _FLAG_BLOCK_SIZE)
            _unscale_kernel[(triton.cdiv(n, _BLOCK_SIZE),)](
                tensor,
                inv_scale,
                n,
                BLOCK=_BLOCK_SIZE,
                NEED_MASK=n % _BLOCK_SIZE != 0,
                num_warps=_NUM_WARPS,
            )
            _non_finite_flag_kernel[(nb,)](
                tensor,
                flags,
                base,
                n,
                BLOCK=_FLAG_BLOCK_SIZE,
                NEED_MASK=n % _FLAG_BLOCK_SIZE != 0,
                num_warps=_NUM_WARPS,
            )
            base += nb
        _reduce_flags_kernel[(1,)](
            flags,
            total_blocks,
            found_inf,
            BLOCK=4096,
        )

    # Huge-tensor path: the vendor-native pointwise engine beats elementwise
    # Triton ALU once the tensors are large.
    if large:
        scale = inv_scale.item()
        has_non_finite = False
        for tensor in large:
            unscale_func(tensor, scale, out0=tensor)
            if not torch.isfinite(tensor).all():
                has_non_finite = True
        if has_non_finite:
            found_inf.fill_(1.0)