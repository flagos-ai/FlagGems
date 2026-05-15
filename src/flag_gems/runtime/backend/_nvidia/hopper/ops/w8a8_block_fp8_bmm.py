from typing import List, Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

_TORCH_TO_GL_DTYPE = {
    torch.float8_e4m3fn: gl.float8e4nv,
    torch.float8_e5m2: gl.float8e5,
    torch.bfloat16: gl.bfloat16,
    torch.float16: gl.float16,
    torch.float32: gl.float32,
}


def _gl_dtype(t: torch.Tensor):
    try:
        return _TORCH_TO_GL_DTYPE[t.dtype]
    except KeyError as e:
        raise TypeError(f"Unsupported tensor dtype: {t.dtype}") from e


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    m_reps = triton.cdiv(BLOCK_M, m)
    n_reps = triton.cdiv(num_warps, m_reps)
    max_n = max(BLOCK_N // n_reps, 8)
    n = 256
    while n > max_n or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )


@aggregate
class Config:
    B: gl.constexpr
    M: gl.constexpr
    M_aligned: gl.constexpr
    N: gl.constexpr
    K: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    TILE_ORDER: gl.constexpr
    SWAP_AB: gl.constexpr
    num_warps: gl.constexpr
    num_stages: gl.constexpr
    num_sms: gl.constexpr
    # xs (per-token scale) strides into the caller's [B, M, num_kb] tensor.
    xs_sB: gl.constexpr
    xs_sM: gl.constexpr
    xs_sKb: gl.constexpr
    # Derived: tile counts.
    num_m_tiles: gl.constexpr
    num_n_tiles: gl.constexpr
    num_k_blocks: gl.constexpr
    num_tiles_per_batch: gl.constexpr
    num_tiles: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        B,
        M,
        M_aligned,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TILE_ORDER,
        SWAP_AB,
        num_warps,
        num_stages,
        num_sms,
        xs_sB,
        xs_sM,
        xs_sKb,
    ):
        self.B = gl.constexpr(B)
        self.M = gl.constexpr(M)
        self.M_aligned = gl.constexpr(M_aligned)
        self.N = gl.constexpr(N)
        self.K = gl.constexpr(K)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.TILE_ORDER = gl.constexpr(TILE_ORDER)
        self.SWAP_AB = gl.constexpr(SWAP_AB)
        self.num_warps = gl.constexpr(num_warps)
        self.num_stages = gl.constexpr(num_stages)
        self.num_sms = gl.constexpr(num_sms)
        self.xs_sB = gl.constexpr(xs_sB)
        self.xs_sM = gl.constexpr(xs_sM)
        self.xs_sKb = gl.constexpr(xs_sKb)
        num_m = M_aligned // BLOCK_M
        num_n = N // BLOCK_N
        self.num_m_tiles = gl.constexpr(num_m)
        self.num_n_tiles = gl.constexpr(num_n)
        self.num_k_blocks = gl.constexpr(K // BLOCK_K)
        self.num_tiles_per_batch = gl.constexpr(num_m * num_n)
        self.num_tiles = gl.constexpr(B * num_m * num_n)


@aggregate
class BarrierCounter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def increment(self):
        if self.num_barriers == 1:
            return BarrierCounter(gl.to_tensor(0), self.phase ^ 1, self.num_barriers)
        next_index = self.index + 1
        rollover = next_index == self.num_barriers
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return BarrierCounter(index, phase, self.num_barriers)


@aggregate
class Channel:
    x_smem: gl.shared_memory_descriptor
    y_smem: gl.shared_memory_descriptor
    ready_bars: gl.shared_memory_descriptor
    empty_bars: gl.shared_memory_descriptor
    num_stages: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, x_smem, y_smem, ready_bars, empty_bars, num_stages):
        self.x_smem = x_smem
        self.y_smem = y_smem
        self.ready_bars = ready_bars
        self.empty_bars = empty_bars
        self.num_stages = gl.constexpr(num_stages)

    @gluon.jit
    def alloc(
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr,
        x_dtype: gl.constexpr,
        x_layout: gl.constexpr,
        y_dtype: gl.constexpr,
        y_layout: gl.constexpr,
        num_stages: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        # x: 3D box [1, BLOCK_M, BLOCK_K] (x is permuted/non-contig at the global level).
        # y: 2D box. xs is loaded directly with gl.load (not staged through smem).
        x_smem = gl.allocate_shared_memory(
            x_dtype, [num_stages, 1, BLOCK_M, BLOCK_K], x_layout
        )
        y_smem = gl.allocate_shared_memory(
            y_dtype, [num_stages, BLOCK_N, BLOCK_K], y_layout
        )
        ready_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        empty_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        for i in gl.static_range(num_stages):
            mbarrier.init(ready_bars.index(i), count=1)
            mbarrier.init(empty_bars.index(i), count=1)
            mbarrier.arrive(empty_bars.index(i), count=1)
        return Channel(x_smem, y_smem, ready_bars, empty_bars, num_stages)

    @gluon.jit
    def release(self):
        self.x_smem._keep_alive()
        self.y_smem._keep_alive()
        for i in gl.static_range(self.num_stages):
            mbarrier.invalidate(self.ready_bars.index(i))
            mbarrier.invalidate(self.empty_bars.index(i))


@gluon.jit
def get_tile(tile_id, config):
    # TILE_ORDER: 0 = horizontal (N fastest within batch — favours x reuse across N sweep)
    #             1 = vertical   (M fastest within batch — favours y reuse across M sweep)
    batch_id = tile_id // config.num_tiles_per_batch
    local_id = tile_id % config.num_tiles_per_batch
    if config.TILE_ORDER == 0:
        m_tile_id = local_id // config.num_n_tiles
        n_tile_id = local_id % config.num_n_tiles
    else:
        n_tile_id = local_id // config.num_m_tiles
        m_tile_id = local_id % config.num_m_tiles
    return batch_id, m_tile_id, n_tile_id


@gluon.jit
def compute_partition(channel, config, tensors):
    x_desc, y_desc, xs_ptr, z_desc, ys_ptr = tensors
    start_pid = gl.program_id(0)
    counter = BarrierCounter(
        index=gl.to_tensor(0), phase=gl.to_tensor(0), num_barriers=config.num_stages
    )

    if config.SWAP_AB:
        mma_layout: gl.constexpr = pick_wgmma_layout(
            x_desc.dtype, config.BLOCK_N, config.BLOCK_M, num_warps=config.num_warps
        )
        xs_load_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    else:
        mma_layout: gl.constexpr = pick_wgmma_layout(
            x_desc.dtype, config.BLOCK_M, config.BLOCK_N, num_warps=config.num_warps
        )
        xs_load_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)

    z_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [1, config.BLOCK_M, config.BLOCK_N], z_desc.dtype
    )
    z_smem = gl.allocate_shared_memory(
        z_desc.dtype, [1, config.BLOCK_M, config.BLOCK_N], z_smem_layout
    )

    # xs in-tile lane indices (one fp32 per token along BLOCK_M).
    xs_lane = gl.arange(0, config.BLOCK_M, layout=xs_load_layout)

    for tile_id in range(start_pid, config.num_tiles, config.num_sms):
        batch_id, m_tile_id, n_tile_id = get_tile(tile_id, config)
        m_start = m_tile_id * config.BLOCK_M
        n_start = n_tile_id * config.BLOCK_N
        # ys layout matches the scale grid (N/BLOCK_N, K/BLOCK_K); one scale per (n_tile, k_block).
        ys_base = (batch_id * config.num_n_tiles + n_tile_id) * config.num_k_blocks
        # xs is the caller's [B, M, num_kb] tensor (strided, possibly non-contig).
        xs_m = m_start + xs_lane
        xs_mask = xs_m < config.M
        xs_row_base = batch_id * config.xs_sB + xs_m * config.xs_sM

        if config.SWAP_AB:
            partial_zero = gl.zeros(
                (config.BLOCK_N, config.BLOCK_M), dtype=gl.float32, layout=mma_layout
            )
            acc = gl.zeros(
                (config.BLOCK_N, config.BLOCK_M), dtype=gl.float32, layout=mma_layout
            )
        else:
            partial_zero = gl.zeros(
                (config.BLOCK_M, config.BLOCK_N), dtype=gl.float32, layout=mma_layout
            )
            acc = gl.zeros(
                (config.BLOCK_M, config.BLOCK_N), dtype=gl.float32, layout=mma_layout
            )

        for k in range(0, config.K, config.BLOCK_K):
            k_block_idx = k // config.BLOCK_K
            index, phase = counter.index, counter.phase
            x_slot = channel.x_smem.index(index)  # [1, BLOCK_M, BLOCK_K]
            y_slot = channel.y_smem.index(index)  # [BLOCK_N, BLOCK_K]
            ready_bar = channel.ready_bars.index(index)
            empty_bar = channel.empty_bars.index(index)
            mbarrier.wait(ready_bar, phase)

            x = x_slot.reshape((config.BLOCK_M, config.BLOCK_K))
            y = y_slot

            x_s = gl.load(
                xs_ptr + xs_row_base + k_block_idx * config.xs_sKb,
                mask=xs_mask,
                other=0.0,
            )
            y_s = gl.load(ys_ptr + ys_base + k_block_idx)
            xy_s = x_s * y_s

            if config.SWAP_AB:
                x_t = x.permute((1, 0))
                partial_async = warpgroup_mma(
                    y, x_t, partial_zero, use_acc=False, is_async=True
                )
                partial = warpgroup_mma_wait(num_outstanding=0, deps=(partial_async,))
                acc = acc + partial * xy_s[None, :]
            else:
                y_t = y.permute((1, 0))
                partial_async = warpgroup_mma(
                    x, y_t, partial_zero, use_acc=False, is_async=True
                )
                partial = warpgroup_mma_wait(num_outstanding=0, deps=(partial_async,))
                acc = acc + partial * xy_s[:, None]

            mbarrier.arrive(empty_bar)
            counter = counter.increment()

        acc_out = acc.to(z_desc.dtype)
        if config.SWAP_AB:
            acc_out = acc_out.permute((1, 0))
        tma.store_wait(pendings=0)
        z_smem.reshape((config.BLOCK_M, config.BLOCK_N)).store(acc_out)
        fence_async_shared()
        tma.async_copy_shared_to_global(z_desc, [batch_id, m_start, n_start], z_smem)

    tma.store_wait(pendings=0)


@gluon.jit
def load_partition(channel, config, tensors):
    x_desc, y_desc, xs_ptr, z_desc, ys_ptr = tensors
    start_pid = gl.program_id(0)
    counter = BarrierCounter(
        index=gl.to_tensor(0), phase=gl.to_tensor(0), num_barriers=config.num_stages
    )

    nbytes: gl.constexpr = (
        config.BLOCK_M * config.BLOCK_K + config.BLOCK_N * config.BLOCK_K
    )

    for tile_id in range(start_pid, config.num_tiles, config.num_sms):
        batch_id, m_tile_id, n_tile_id = get_tile(tile_id, config)
        m_start = m_tile_id * config.BLOCK_M
        n_start = n_tile_id * config.BLOCK_N

        y_row = batch_id * config.N + n_start

        for k in range(0, config.K, config.BLOCK_K):
            index, phase = counter.index, counter.phase
            x_slot = channel.x_smem.index(index)
            y_slot = channel.y_smem.index(index)
            ready_bar = channel.ready_bars.index(index)
            empty_bar = channel.empty_bars.index(index)
            mbarrier.wait(empty_bar, phase)

            mbarrier.expect(ready_bar, nbytes)
            tma.async_copy_global_to_shared(
                x_desc, [batch_id, m_start, k], ready_bar, x_slot
            )
            tma.async_copy_global_to_shared(y_desc, [y_row, k], ready_bar, y_slot)

            counter = counter.increment()


@triton.autotune(
    configs=[
        triton.Config({"TILE_ORDER": tile_order}, num_warps=nw, num_stages=ns)
        for nw in (4, 8)
        for ns in (4, 6, 8)
        for tile_order in (0, 1)  # 0=horizontal (n fastest), 1=vertical (m fastest)
    ],
    key=["B", "M_aligned", "N", "K"],
)
@gluon.jit
def w8a8_block_fp8_bmm_kernel(
    x_desc,
    y_desc,
    xs_ptr,
    z_desc,
    ys_ptr,
    xs_sB: gl.constexpr,
    xs_sM: gl.constexpr,
    xs_sKb: gl.constexpr,
    B: gl.constexpr,
    M: gl.constexpr,
    M_aligned: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    TILE_ORDER: gl.constexpr,
    SWAP_AB: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    num_sms: gl.constexpr,
):
    config = Config(
        B=B,
        M=M,
        M_aligned=M_aligned,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILE_ORDER=TILE_ORDER,
        SWAP_AB=SWAP_AB,
        num_warps=num_warps,
        num_stages=num_stages,
        num_sms=num_sms,
        xs_sB=xs_sB,
        xs_sM=xs_sM,
        xs_sKb=xs_sKb,
    )
    tensors = (x_desc, y_desc, xs_ptr, z_desc, ys_ptr)
    channel = Channel.alloc(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        x_dtype=x_desc.dtype,
        x_layout=gl.constexpr(x_desc.layout),
        y_dtype=y_desc.dtype,
        y_layout=gl.constexpr(y_desc.layout),
        num_stages=num_stages,
        num_warps=num_warps,
    )

    gl.warp_specialize(
        [
            (compute_partition, (channel, config, tensors)),
            (load_partition, (channel, config, tensors)),
        ],
        [1],
        [24],
    )

    channel.release()


def w8a8_block_fp8_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    block_size: List[int] = [128, 128],
    z: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
):
    # x: [B, M, K]  fp8
    # y: [B, N, K]  fp8
    # xs: [B, M, K // block_k]      f32
    # ys: [B, N // block_n, K // block_k]  f32
    # z:  [B, M, N]  out_dtype
    assert len(block_size) == 2
    BLOCK_N, BLOCK_K = block_size
    assert (
        BLOCK_N == 128 and BLOCK_K == 128
    ), "this kernel assumes 128x128 block-wise FP8 scales"

    assert x.ndim == 3 and y.ndim == 3 and xs.ndim == 3 and ys.ndim == 3
    assert x.shape[0] == y.shape[0] == xs.shape[0] == ys.shape[0]
    assert x.shape[-1] == y.shape[-1]
    assert x.shape[:-1] == xs.shape[:-1]
    assert x.stride(-1) == 1 and y.stride(-1) == 1

    device = x.device
    B, M, K = x.shape
    _, N, _ = y.shape
    assert K % BLOCK_K == 0 and N % BLOCK_N == 0
    num_kb = K // BLOCK_K

    if z is None:
        z = torch.empty((B, M, N), device=device, dtype=output_dtype)
    else:
        assert z.shape == (B, M, N) and z.device == device and z.dtype == output_dtype
        assert z.stride(-1) == 1

    BLOCK_M = max(8, min(64, 1 << ((M - 1).bit_length())))
    SWAP_AB = 1 if BLOCK_M < 64 else 0

    M_aligned = triton.cdiv(M, BLOCK_M) * BLOCK_M

    x_gl_dtype = _gl_dtype(x)
    y_gl_dtype = _gl_dtype(y)
    z_gl_dtype = _gl_dtype(z)

    x_layout = gl.NVMMASharedLayout.get_default_for([1, BLOCK_M, BLOCK_K], x_gl_dtype)
    x_desc = TensorDescriptor.from_tensor(
        x, block_shape=[1, BLOCK_M, BLOCK_K], layout=x_layout
    )

    assert y.is_contiguous(), "y must be contiguous so it can be viewed as (B*N, K)"
    y_flat = y.view(B * N, K)
    y_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], y_gl_dtype)
    y_desc = TensorDescriptor.from_tensor(
        y_flat, block_shape=[BLOCK_N, BLOCK_K], layout=y_layout
    )

    assert xs.ndim == 3 and xs.shape == (B, M, num_kb)
    xs_sB, xs_sM, xs_sKb = xs.stride()

    z_layout = gl.NVMMASharedLayout.get_default_for([1, BLOCK_M, BLOCK_N], z_gl_dtype)
    z_desc = TensorDescriptor.from_tensor(
        z, block_shape=[1, BLOCK_M, BLOCK_N], layout=z_layout
    )

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    w8a8_block_fp8_bmm_kernel[(num_sms,)](
        x_desc,
        y_desc,
        xs,
        z_desc,
        ys,
        xs_sB=xs_sB,
        xs_sM=xs_sM,
        xs_sKb=xs_sKb,
        B=B,
        M=M,
        M_aligned=M_aligned,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SWAP_AB=SWAP_AB,
        num_sms=num_sms,
    )

    return z
