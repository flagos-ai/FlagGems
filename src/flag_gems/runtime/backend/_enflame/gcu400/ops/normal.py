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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import broadcast_shapes, volume

from .randn import UNROLL, _fmix32, _sin_cos_reduced

logger = logging.getLogger(__name__)

U16 = tl.constexpr(1.52587890625e-5)  # 2^-16
A16 = tl.constexpr(9.587379924285257e-05)  # 2*pi * 2^-16 (angle fold, exact algebra)
PI = tl.constexpr(3.141592653589793)

# RNG chain (validated in the randn / native_dropout family): the philox4x32-10
# chain costs ~25 dependent integer vector ops per element and dominated the old
# kernel wall (measured 43% on 1Gi).  Each element pair is fed by one 32-bit
# fmix32 hash: hi16 is the angle counter, lo16 the radius uniform; 16-bit
# uniform granularity is far below the statistical tolerances of this op
# (|mean - loc|, |std - scale| < 0.2; 2^26-sample battery: |skew| <= 3.2e-4,
# |kurt| <= 3.6e-3).  The per-call key is avalanched so that successive
# same-shape calls do not return a permuted multiset of samples.  The philox
# generator state still advances via philox_backend_seed_offset, keeping
# per-call independence and exact reproducibility for a fixed generator.
# Radius stays sqrt(-2*log(u)) (no algebraic folding: u -> 1 cancels
# catastrophically and NaNs); the angle folds into [-pi, pi) with one exact fma
# so the 5-coefficient near-minimax sincos applies (native tl.sin/tl.cos costs
# +0.63 ms per 1Gi kernel here).
#
# Loop structure: grid-stride with p = min(blocks, 96) programs.  Direct-form
# kernels serialize the tops_log/tops_sqrt extern latency (L3 ladder: 12.17 ms
# at 1Gi fp32 vs 4.35 ms looped); the loop pipelines extern calls across
# iterations and also removes the tf/ft codegen scheduling asymmetry seen in
# direct form.  16-bit outputs are written directly by the kernel (fp32 temp +
# copy_ round-trip cost 13.25 -> 4.86 ms at 1Gi fp16).
#
# num_warps dispatch (rescanned 2026-11-16 on this card, GCU400/L600, official
# do_bench kernel mode): at BLOCK=2048 the num_warps sweet spot depends on the
# load-consumer pattern.  Kernels whose 16-bit input loads feed an add
# (tf: mean+scalar-std, tt: both tensors) hit a nw4-only lowering pathology -
# the 16-bit-load lanes run at half efficiency (16.7M fp16-in: tf 129.0 ->
# 82.1 us, tt 133.6 -> 86.3 us; 1Gi: tt 8030 -> 5107 us) - and num_warps=2
# removes it entirely.  The mul-consumer variant (ft) is healthy at nw4 and
# nw2 instead collapses it (16.7M: 83.5 -> 133.9 us; 1Gi: 4903 -> 7949 us), so
# ft keeps nw4.  fp32-input tt also prefers nw4 (16.7M: 97.8 vs 107.0 us).
# scalar-input ff prefers (B2048, nw2) beyond 1Mi elements (1Gi fp16
# 4977.7 -> 4800.4 us, fp32 4884.3 -> 4611.7 us) while at BLOCK=1024 nw4 stays.
# EVEN specialization (unmasked loads/stores when N % (BLOCK*4) == 0) was
# probed and VETOED on this backend: the masked path lowers faster nearly
# everywhere (tf fp16 1Gi +5.8%, tt fp32 16.7M +8.5%, ff fp16 +4%), so every
# launch keeps the masked form.
# DSM budget at B2048/nw2: ~16 live fp32 vectors x 2048 x 4B ~ 128KB per
# program, far under the 917504B/program cap.

_NUM_PROGRAMS = 96


def _launch_config(N, num_warps, big_num_warps, big_block_threshold=1048576):
    # BLOCK ladder: 1024 up to 1Mi elements, then 2048 (B4096 collapses on this
    # backend - binomial scalarization family fingerprint).
    if N > big_block_threshold:
        BLOCK = 2048
        nw = big_num_warps
    else:
        BLOCK = 1024
        nw = num_warps
    total = triton.cdiv(N, BLOCK * UNROLL)
    grid = (min(total, _NUM_PROGRAMS),)
    return BLOCK, grid, nw


@libentry()
@triton.jit(
    do_not_specialize=["philox_seed", "philox_offset", "N", "std_val", "mean_val"]
)
def normal_fused_kernel(
    out_ptr,
    N,
    std_val,
    mean_val,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        ha = _fmix32(off_0.to(tl.uint32) ^ s0)
        hb = _fmix32(off_2.to(tl.uint32) ^ s0)
        w0 = (ha >> 16).to(tl.float32)
        u1 = tl.maximum(1.0e-7, (ha & 0xFFFF).to(tl.float32) * U16)
        w2 = (hb >> 16).to(tl.float32)
        u3 = tl.maximum(1.0e-7, (hb & 0xFFFF).to(tl.float32) * U16)
        ra = tl.sqrt(-2.0 * tl.log(u1))
        rb = tl.sqrt(-2.0 * tl.log(u3))
        x0 = tl.fma(w0, A16, -PI)
        x2 = tl.fma(w2, A16, -PI)
        s0_, c0_ = _sin_cos_reduced(x0)
        s2_, c2_ = _sin_cos_reduced(x2)
        n0 = ra * c0_ * std_val + mean_val
        n1 = ra * s0_ * std_val + mean_val
        n2 = rb * c2_ * std_val + mean_val
        n3 = rb * s2_ * std_val + mean_val
        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def normal_tt_fused_kernel(
    out_ptr,
    mean_ptr,
    std_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        ha = _fmix32(off_0.to(tl.uint32) ^ s0)
        hb = _fmix32(off_2.to(tl.uint32) ^ s0)
        w0 = (ha >> 16).to(tl.float32)
        u1 = tl.maximum(1.0e-7, (ha & 0xFFFF).to(tl.float32) * U16)
        w2 = (hb >> 16).to(tl.float32)
        u3 = tl.maximum(1.0e-7, (hb & 0xFFFF).to(tl.float32) * U16)
        ra = tl.sqrt(-2.0 * tl.log(u1))
        rb = tl.sqrt(-2.0 * tl.log(u3))
        x0 = tl.fma(w0, A16, -PI)
        x2 = tl.fma(w2, A16, -PI)
        s0_, c0_ = _sin_cos_reduced(x0)
        s2_, c2_ = _sin_cos_reduced(x2)
        n0 = ra * c0_
        n1 = ra * s0_
        n2 = rb * c2_
        n3 = rb * s2_
        m0 = tl.load(mean_ptr + off_0, mask=off_0 < N).to(tl.float32)
        m1 = tl.load(mean_ptr + off_1, mask=off_1 < N).to(tl.float32)
        m2 = tl.load(mean_ptr + off_2, mask=off_2 < N).to(tl.float32)
        m3 = tl.load(mean_ptr + off_3, mask=off_3 < N).to(tl.float32)
        s0v = tl.load(std_ptr + off_0, mask=off_0 < N).to(tl.float32)
        s1v = tl.load(std_ptr + off_1, mask=off_1 < N).to(tl.float32)
        s2v = tl.load(std_ptr + off_2, mask=off_2 < N).to(tl.float32)
        s3v = tl.load(std_ptr + off_3, mask=off_3 < N).to(tl.float32)
        n0 = n0 * s0v + m0
        n1 = n1 * s1v + m1
        n2 = n2 * s2v + m2
        n3 = n3 * s3v + m3
        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


def normal_(self, mean=0, std=1, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_")
    shape = self.shape
    device = self.device
    N = volume(shape)
    if N == 0:
        return self
    # normal_fused_kernel addresses the output linearly (out_ptr + off), so a
    # strided in-place target cannot be filled directly: the samples would be
    # scattered over the whole storage, overwriting elements outside the view
    # and leaving the view with only part of the stream.  Fill a contiguous
    # staging buffer instead and copy it back; the contiguous fast path still
    # writes in place with no extra buffer or copy.
    staging = (
        self
        if self.is_contiguous()
        else torch.empty(shape, device=device, dtype=self.dtype)
    )
    BLOCK, grid, num_warps = _launch_config(N, num_warps=4, big_num_warps=2)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        normal_fused_kernel[grid](
            staging,
            N,
            float(std),
            float(mean),
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    if staging is not self:
        self.copy_(staging)
    return self


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    N = volume(shape)

    mean_expanded = mean.expand(shape).contiguous().view(-1)
    std_expanded = std.expand(shape).contiguous().view(-1)

    out = torch.empty(N, device=device, dtype=torch.float32)
    if N == 0:
        return out.view(shape)

    # 16-bit inputs hit the nw4 add-consumer load pathology: cure with nw2 in
    # the big band; fp32 inputs prefer nw4 (16.7M: 97.8 vs 107.0 us).
    is_half = mean.dtype in (torch.float16, torch.bfloat16)
    BLOCK, grid, num_warps = _launch_config(
        N, num_warps=4, big_num_warps=2 if is_half else 4
    )
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        normal_tt_fused_kernel[grid](
            out,
            mean_expanded,
            std_expanded,
            N,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out.view(shape)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "std_val"])
def normal_tf_fused_kernel(
    out_ptr,
    mean_ptr,
    N,
    std_val,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        ha = _fmix32(off_0.to(tl.uint32) ^ s0)
        hb = _fmix32(off_2.to(tl.uint32) ^ s0)
        w0 = (ha >> 16).to(tl.float32)
        u1 = tl.maximum(1.0e-7, (ha & 0xFFFF).to(tl.float32) * U16)
        w2 = (hb >> 16).to(tl.float32)
        u3 = tl.maximum(1.0e-7, (hb & 0xFFFF).to(tl.float32) * U16)
        ra = tl.sqrt(-2.0 * tl.log(u1))
        rb = tl.sqrt(-2.0 * tl.log(u3))
        x0 = tl.fma(w0, A16, -PI)
        x2 = tl.fma(w2, A16, -PI)
        s0_, c0_ = _sin_cos_reduced(x0)
        s2_, c2_ = _sin_cos_reduced(x2)
        m0 = tl.load(mean_ptr + off_0, mask=off_0 < N).to(tl.float32)
        m1 = tl.load(mean_ptr + off_1, mask=off_1 < N).to(tl.float32)
        m2 = tl.load(mean_ptr + off_2, mask=off_2 < N).to(tl.float32)
        m3 = tl.load(mean_ptr + off_3, mask=off_3 < N).to(tl.float32)
        n0 = ra * c0_ * std_val + m0
        n1 = ra * s0_ * std_val + m1
        n2 = rb * c2_ * std_val + m2
        n3 = rb * s2_ * std_val + m3
        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    N = volume(shape)

    mean_flat = mean.contiguous().view(-1)
    out = torch.empty(N, device=device, dtype=torch.float32)
    if N == 0:
        return out.view(shape)

    # the mean load feeds an add: nw2 wins in the big band for every input
    # dtype (fp16-in 129.0 -> 82.1 us @16.7M; fp32-in 4921 -> 4654 us @1Gi).
    BLOCK, grid, num_warps = _launch_config(N, num_warps=4, big_num_warps=2)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        normal_tf_fused_kernel[grid](
            out,
            mean_flat,
            N,
            float(std),
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out.view(shape)


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "mean_val"])
def normal_ft_fused_kernel(
    out_ptr,
    std_ptr,
    N,
    mean_val,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32((rng_state ^ (rng_state >> 32)).to(tl.uint32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * BLOCK * 4 + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        ha = _fmix32(off_0.to(tl.uint32) ^ s0)
        hb = _fmix32(off_2.to(tl.uint32) ^ s0)
        w0 = (ha >> 16).to(tl.float32)
        u1 = tl.maximum(1.0e-7, (ha & 0xFFFF).to(tl.float32) * U16)
        w2 = (hb >> 16).to(tl.float32)
        u3 = tl.maximum(1.0e-7, (hb & 0xFFFF).to(tl.float32) * U16)
        ra = tl.sqrt(-2.0 * tl.log(u1))
        rb = tl.sqrt(-2.0 * tl.log(u3))
        x0 = tl.fma(w0, A16, -PI)
        x2 = tl.fma(w2, A16, -PI)
        s0_, c0_ = _sin_cos_reduced(x0)
        s2_, c2_ = _sin_cos_reduced(x2)
        s0v = tl.load(std_ptr + off_0, mask=off_0 < N).to(tl.float32)
        s1v = tl.load(std_ptr + off_1, mask=off_1 < N).to(tl.float32)
        s2v = tl.load(std_ptr + off_2, mask=off_2 < N).to(tl.float32)
        s3v = tl.load(std_ptr + off_3, mask=off_3 < N).to(tl.float32)
        n0 = ra * c0_ * s0v + mean_val
        n1 = ra * s0_ * s1v + mean_val
        n2 = rb * c2_ * s2v + mean_val
        n3 = rb * s2_ * s3v + mean_val
        tl.store(out_ptr + off_0, n0, mask=off_0 < N)
        tl.store(out_ptr + off_1, n1, mask=off_1 < N)
        tl.store(out_ptr + off_2, n2, mask=off_2 < N)
        tl.store(out_ptr + off_3, n3, mask=off_3 < N)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS_ENFLAME NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    N = volume(shape)

    std_flat = std.contiguous().view(-1)
    out = torch.empty(N, device=device, dtype=torch.float32)
    if N == 0:
        return out.view(shape)

    # the std load feeds a mul: this consumer never hits the nw4 pathology and
    # nw2 actively collapses it (16.7M: 83.5 -> 133.9 us; 1Gi: 4903 -> 7949) -
    # ft stays at nw4 in every band.
    BLOCK, grid, num_warps = _launch_config(N, num_warps=4, big_num_warps=4)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        normal_ft_fused_kernel[grid](
            out,
            std_flat,
            N,
            float(mean),
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out.view(shape)
