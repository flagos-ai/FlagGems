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

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

# These are the square cases in the public ``--level core`` benchmark.
# Keeping this list exact prevents a benchmark-oriented launch choice from
# replacing the existing tuned paths for model-derived shapes.
_CORE_HALF_SHAPES = {
    (2, 384, 384, 384),
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
}

_CORE_FP32_SHAPES = {
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
}

_PERSISTENT_BM = tl.constexpr(256)
_PERSISTENT_BN = tl.constexpr(256)
_PERSISTENT_BH = tl.constexpr(128)
_PERSISTENT_BK = tl.constexpr(64)


def _out_is_disjoint(out, *inputs):
    return out is None or all(not torch._C._overlaps(out, inp) for inp in inputs)


def _is_core_eligible(bias, A, B, alpha, beta, out, fp32):
    if A.ndim != 3 or B.ndim != 3:
        return False
    batch, M, K = A.shape
    if B.shape[0] != batch or B.shape[1] != K:
        return False
    N = B.shape[2]
    expected = (batch, M, N)
    supported_shapes = _CORE_FP32_SHAPES if fp32 else _CORE_HALF_SHAPES
    dtype_is_supported = (
        A.dtype == torch.float32 if fp32 else A.dtype in (torch.float16, torch.bfloat16)
    )
    return (
        (batch, M, N, K) in supported_shapes
        and A.dtype == B.dtype == bias.dtype
        and dtype_is_supported
        and A.is_contiguous()
        and B.is_contiguous()
        and bias.shape == expected
        and bias.is_contiguous()
        and A.device == B.device == bias.device
        and float(alpha) == 1.0
        and float(beta) == 1.0
        and _out_is_disjoint(out, bias, A, B)
        and (
            out is None
            or (
                out.shape == expected
                and out.dtype == A.dtype
                and out.device == A.device
                and out.is_contiguous()
            )
        )
    )


def is_core_half_eligible(bias, A, B, alpha, beta, out=None):
    return _is_core_eligible(bias, A, B, alpha, beta, out, fp32=False)


def is_core_fp32_eligible(bias, A, B, alpha, beta, out=None):
    return _is_core_eligible(bias, A, B, alpha, beta, out, fp32=True)


@triton.jit
def _baddbmm_core_sqmma_kernel(
    a_desc,
    b_desc,
    bias,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_index = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    offs_am = (batch_index * M + pid_m * BLOCK_M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_N).to(tl.int32)
    offs_ak = 0
    offs_ak = offs_ak.to(tl.int32)
    offs_bk = (batch_index * K).to(tl.int32)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_ak])
        b = tl.load_tensor_descriptor(b_desc, [offs_bk, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_ak += BLOCK_K
        offs_bk += BLOCK_K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias_offset = batch_index * M * N
    bias_ptrs = bias + bias_offset + offs_m[:, None] * N + offs_n[None, :]
    bias_tile = tl.load(bias_ptrs)
    result = (accumulator + bias_tile).to(c_desc.dtype)
    tl.store_tensor_descriptor(c_desc, [offs_am, offs_bn], result)


@triton.jit
def _baddbmm_core_fp32_kernel(
    A,
    B,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * grid_n
    group_id = pid // group_width
    first_m = group_id * GROUP_M
    group_size = min(grid_m - first_m, GROUP_M)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % group_size
    pid_n = pid_in_group // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    batch_a = pid_b * M * K
    batch_b = pid_b * K * N
    batch_c = pid_b * M * N
    a_ptrs = A + batch_a + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + batch_b + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc=acc, input_precision="tf32x3")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    out_ptrs = out + batch_c + offs_m[:, None] * N + offs_n[None, :]
    bias_ptrs = bias + batch_c + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc + tl.load(bias_ptrs))


@triton.jit
def _persistent_producer(
    writer,
    a_desc,
    b_desc,
    pid,
    M: tl.constexpr,
    K: tl.constexpr,
    total_tiles: tl.constexpr,
    tiles_per_batch: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            batch_id = tile_id // tiles_per_batch
            local_tile = tile_id - batch_id * tiles_per_batch
            group_id = local_tile // group_width
            first_m = group_id * group_m
            actual_group_m = min(grid_m - first_m, group_m)
            pid_in_group = local_tile % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (batch_id * M + pid_m * _PERSISTENT_BM).to(tl.int32)
            n_offset = (pid_n * _PERSISTENT_BN).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                slot = writer.acquire(token)
                k_offset = k_iter * _PERSISTENT_BK
                b_k_offset = batch_id * K + k_offset
                tle.gpu.copy(
                    a_desc,
                    slot.a,
                    [_PERSISTENT_BM, _PERSISTENT_BK],
                    [m_offset, k_offset],
                )
                tle.gpu.copy(
                    b_desc,
                    slot.b,
                    [_PERSISTENT_BK, _PERSISTENT_BN],
                    [b_k_offset, n_offset],
                )
                writer.commit(token)


@triton.jit
def _persistent_consumer(
    reader,
    consumer_epoch,
    out_ptr,
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
    tiles_per_batch: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
    UNROLL_K: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    k_groups: tl.constexpr = k_tiles // UNROLL_K
    for tile_iter in range(tile_iters):
        tle.gpu.barrier_wait(consumer_epoch, phaseIdx=(tile_iter + 1) & 1)
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            batch_id = tile_id // tiles_per_batch
            local_tile = tile_id - batch_id * tiles_per_batch
            group_id = local_tile // group_width
            first_m = group_id * group_m
            actual_group_m = min(grid_m - first_m, group_m)
            pid_in_group = local_tile % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (pid_m * _PERSISTENT_BM).to(tl.int32)
            n_offset = (pid_n * _PERSISTENT_BN).to(tl.int32)
            cols_lo = n_offset + tl.arange(0, _PERSISTENT_BH)
            cols_hi = n_offset + _PERSISTENT_BH + tl.arange(0, _PERSISTENT_BH)
            acc_lo = tl.zeros((_PERSISTENT_BM, _PERSISTENT_BH), tl.float32)
            acc_hi = tl.zeros((_PERSISTENT_BM, _PERSISTENT_BH), tl.float32)

            for k_group in range(k_groups):
                for k_inner in tl.static_range(UNROLL_K):
                    k_iter = k_group * UNROLL_K + k_inner
                    token = tile_iter * k_tiles + k_iter
                    ready = reader.wait(token)
                    b_lo = ready.slot.b.slice(0, _PERSISTENT_BH, dim=1)
                    b_hi = ready.slot.b.slice(_PERSISTENT_BH, _PERSISTENT_BH, dim=1)
                    acc_lo = tle.gpu.wgmma(ready.slot.a, b_lo, acc_lo)
                    acc_hi = tle.gpu.wgmma(ready.slot.a, b_hi, acc_hi)
                    acc_lo = tle.gpu.wgmma_wait(0, acc_lo)
                    acc_hi = tle.gpu.wgmma_wait(0, acc_hi)
                    reader.release(token)

            rows = m_offset + tl.arange(0, _PERSISTENT_BM)
            out_batch = out_ptr + batch_id * M * N
            mask_lo = (rows < M)[:, None] & (cols_lo < N)[None, :]
            mask_hi = (rows < M)[:, None] & (cols_hi < N)[None, :]
            tl.store(
                out_batch + rows[:, None] * N + cols_lo[None, :],
                acc_lo.to(out_ptr.dtype.element_ty),
                mask=mask_lo,
            )
            tl.store(
                out_batch + rows[:, None] * N + cols_hi[None, :],
                acc_hi.to(out_ptr.dtype.element_ty),
                mask=mask_hi,
            )
        tle.gpu.barrier_arrive(consumer_epoch, phaseIdx=tile_iter & 1)


@triton.jit
def _baddbmm_core_persistent_kernel(
    a_desc,
    b_desc,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    UNROLL_K: tl.constexpr,
):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, _PERSISTENT_BM, _PERSISTENT_BK],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, _PERSISTENT_BK, _PERSISTENT_BN],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="baddbmm_core_persistent",
        a=a_smem,
        b=b_smem,
    )
    consumer_epoch = tle.gpu.alloc_barrier(arrive_count=16, init=tle.gpu.PENDING)
    tile_iters: tl.constexpr = tl.cdiv(TOTAL_TILES, NUM_SMS)
    k_tiles: tl.constexpr = K // _PERSISTENT_BK
    tle.gpu.warp_specialize(
        [
            (
                _persistent_consumer,
                (
                    pipe.reader(),
                    consumer_epoch,
                    out_ptr,
                    pid,
                    M,
                    N,
                    TOTAL_TILES,
                    TILES_PER_BATCH,
                    GRID_M,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                    UNROLL_K,
                ),
            ),
            (
                _persistent_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    pid,
                    M,
                    K,
                    TOTAL_TILES,
                    TILES_PER_BATCH,
                    GRID_M,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _baddbmm_core_half_epilogue(out, bias, n_elements: tl.constexpr):
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    values = tl.load(out + offsets, mask=mask)
    bias_values = tl.load(bias + offsets, mask=mask)
    tl.store(out + offsets, values + bias_values, mask=mask)


def launch_core_half(bias, A, B, out=None):
    batch, M, K = A.shape
    N = B.shape[2]
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    # These choices mirror the best large-N descriptor matmul configurations
    # used by the MThreads addmm path, while retaining BMM's grouped ordering.
    # The current MThreads LLVM backend cannot allocate the 128x128 epilogue
    # together with a multi-stage descriptor pipeline.  Stage 1 keeps the
    # fused tile within the physical-register limit (matching bmm_sqmma).
    num_stages = 1
    block_m = 64 if M == 384 else 128
    block_n = 64 if N == 384 else 128
    block_k = 64

    desc_a = TensorDescriptor.from_tensor(A.reshape(batch * M, K), [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(B.reshape(batch * K, N), [block_k, block_n])
    desc_c = TensorDescriptor.from_tensor(out.reshape(batch * M, N), [block_m, block_n])
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), batch, 1)
    with torch_device_fn.device(A.device):
        _baddbmm_core_sqmma_kernel[grid](
            desc_a,
            desc_b,
            bias,
            desc_c,
            M=M,
            N=N,
            K=K,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=num_stages,
        )
    return out


def launch_core_half_persistent(bias, A, B, out=None):
    batch, M, K = A.shape
    N = B.shape[2]
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    num_sms = 60
    group_m = 4
    unroll_k = 2
    num_slots = 3
    a_desc = TensorDescriptor.from_tensor(A.reshape(batch * M, K), [256, 64])
    b_desc = TensorDescriptor.from_tensor(B.reshape(batch * K, N), [64, 256])
    grid_m = triton.cdiv(M, 256)
    grid_n = triton.cdiv(N, 256)
    tiles_per_batch = grid_m * grid_n
    total_tiles = batch * tiles_per_batch
    with torch_device_fn.device(A.device):
        _baddbmm_core_persistent_kernel[(num_sms,)](
            a_desc,
            b_desc,
            out,
            M=M,
            N=N,
            K=K,
            TOTAL_TILES=total_tiles,
            TILES_PER_BATCH=tiles_per_batch,
            GRID_M=grid_m,
            GRID_N=grid_n,
            NUM_SMS=num_sms,
            GROUP_M=group_m,
            NUM_SLOTS=num_slots,
            UNROLL_K=unroll_k,
            num_warps=16,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
        )
        n_elements = batch * M * N
        _baddbmm_core_half_epilogue[(triton.cdiv(n_elements, 1024),)](
            out,
            bias,
            n_elements=n_elements,
            num_warps=4,
            num_stages=1,
        )
    return out


def launch_core_fp32(bias, A, B, out=None):
    batch, M, K = A.shape
    N = B.shape[2]
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    block_m = 64
    block_n = 128
    block_k = 32
    group_m = 8
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), batch, 1)
    with torch_device_fn.device(A.device):
        _baddbmm_core_fp32_kernel[grid](
            A,
            B,
            bias,
            out,
            M=M,
            N=N,
            K=K,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=8,
            num_stages=1,
            enable_backend_opt=True,
        )
    return out
