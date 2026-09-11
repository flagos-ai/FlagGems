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

from typing import List, Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTuner
from flag_gems.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_W8A8_BLOCK_FP8_BMM = hasattr(tle.gpu, "alloc_barriers")
    except ImportError:
        tle = None
        HAS_TLE_W8A8_BLOCK_FP8_BMM = False
else:
    tle = None
    HAS_TLE_W8A8_BLOCK_FP8_BMM = False


def _set_triton_descriptor_allocator(device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


def _get_tle_w8a8_block_fp8_bmm_configs():
    # TLE shared-memory tensors require power-of-two shapes, so stage counts
    # like 6 cannot compile.  Stage 8 compiles, but hits launch failures on
    # large DeepSeek-V4 Pro shapes (for example BMM K=7168), so keep the stable
    # Hopper TLE path on 4 stages for now.
    return [
        config
        for config in runtime.get_tuned_config("w8a8_block_fp8_bmm")
        if config.num_stages == 4
    ]


def _filter_tle_w8a8_block_fp8_bmm_configs(configs):
    return [config for config in configs if config.num_stages == 4]


class _TLEW8A8BlockFP8BMMTuner(LibTuner.get("default")):
    def _keep_tle_configs(self):
        configs = _filter_tle_w8a8_block_fp8_bmm_configs(self.configs)
        if not configs:
            configs = _filter_tle_w8a8_block_fp8_bmm_configs(
                self._flagtune_default_configs
            )
        if configs and len(configs) != len(self.configs):
            self._set_configs_and_strategy(configs, self.strategy)
            return True
        return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keep_tle_configs()

    def apply_flagtune(self):
        changed = super().apply_flagtune()
        return self._keep_tle_configs() or changed


@triton.jit
def _get_tile(
    tile_id,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    num_tiles_per_batch: tl.constexpr,
    TILE_ORDER: tl.constexpr,
):
    # TILE_ORDER: 0 = horizontal (N fastest within batch — favours x reuse across N sweep)
    #             1 = vertical   (M fastest within batch — favours y reuse across M sweep)
    batch_id = tile_id // num_tiles_per_batch
    local_id = tile_id % num_tiles_per_batch
    if TILE_ORDER == 0:
        m_tile_id = local_id // num_n_tiles
        n_tile_id = local_id % num_n_tiles
    else:
        n_tile_id = local_id // num_m_tiles
        m_tile_id = local_id % num_m_tiles
    return batch_id, m_tile_id, n_tile_id


@triton.jit
def _tle_w8a8_block_fp8_bmm_compute_partition(
    x_smem,
    y_smem,
    empty_x,
    empty_y,
    full_x,
    full_y,
    xs_ptr,
    z_ptr,
    ys_ptr,
    xs_sB: tl.constexpr,
    xs_sM: tl.constexpr,
    xs_sKb: tl.constexpr,
    z_sB: tl.constexpr,
    z_sM: tl.constexpr,
    z_sN: tl.constexpr,
    B: tl.constexpr,
    M: tl.constexpr,
    M_aligned: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_ORDER: tl.constexpr,
    SWAP_AB: tl.constexpr,
    num_stages: tl.constexpr,
    num_sms: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_m_tiles: tl.constexpr = M_aligned // BLOCK_M
    num_n_tiles: tl.constexpr = N // BLOCK_N
    num_k_blocks: tl.constexpr = K // BLOCK_K
    num_tiles_per_batch: tl.constexpr = num_m_tiles * num_n_tiles
    num_tiles: tl.constexpr = B * num_tiles_per_batch
    xs_lane = tl.arange(0, BLOCK_M)

    for tile_id in range(start_pid, num_tiles, num_sms):
        batch_id, m_tile_id, n_tile_id = _get_tile(
            tile_id, num_m_tiles, num_n_tiles, num_tiles_per_batch, TILE_ORDER
        )
        m_start = m_tile_id * BLOCK_M
        n_start = n_tile_id * BLOCK_N
        # ys layout matches the scale grid (N/BLOCK_N, K/BLOCK_K); one scale per (n_tile, k_block).
        ys_base = (batch_id * num_n_tiles + n_tile_id) * num_k_blocks
        # xs is the caller's [B, M, num_kb] tensor (strided, possibly non-contig).
        xs_m = m_start + xs_lane
        xs_mask = xs_m < M
        xs_row_base = batch_id * xs_sB + xs_m * xs_sM

        if SWAP_AB:
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block_idx in range(0, num_k_blocks):
            index = k_block_idx % num_stages
            phase = k_block_idx // num_stages
            x_slot = x_smem.slot(index).slot(0)  # [BLOCK_M, BLOCK_K]
            y_slot = y_smem.slot(index)  # [BLOCK_N, BLOCK_K]
            tle.gpu.barrier_wait(full_x[index], phaseIdx=phase)
            tle.gpu.barrier_wait(full_y[index], phaseIdx=phase)

            x_s = tl.load(
                xs_ptr + xs_row_base + k_block_idx * xs_sKb,
                mask=xs_mask,
                other=0.0,
            )
            y_s = tl.load(ys_ptr + ys_base + k_block_idx)
            xy_s = x_s * y_s

            if SWAP_AB:
                partial = tle.gpu.wgmma(
                    y_slot,
                    x_slot,
                    out_dtype=tl.float32,
                    trans_b=True,
                )
                partial = tle.gpu.wgmma_wait(0, partial)
                acc = acc + partial * xy_s[None, :]
            else:
                partial = tle.gpu.wgmma(
                    x_slot,
                    y_slot,
                    out_dtype=tl.float32,
                    trans_b=True,
                )
                partial = tle.gpu.wgmma_wait(0, partial)
                acc = acc + partial * xy_s[:, None]

            tle.gpu.barrier_arrive(empty_x[index], phaseIdx=phase)
            tle.gpu.barrier_arrive(empty_y[index], phaseIdx=phase)

        if SWAP_AB:
            acc_out = tl.trans(acc)
        else:
            acc_out = acc
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = n_start + tl.arange(0, BLOCK_N)
        z_ptrs = (
            z_ptr + batch_id * z_sB + offs_m[:, None] * z_sM + offs_n[None, :] * z_sN
        )
        z_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(z_ptrs, acc_out, mask=z_mask)


@triton.jit
def _tle_w8a8_block_fp8_bmm_load_partition(
    x_desc,
    y_desc,
    x_smem,
    y_smem,
    empty_x,
    empty_y,
    full_x,
    full_y,
    B: tl.constexpr,
    M: tl.constexpr,
    M_aligned: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_ORDER: tl.constexpr,
    num_stages: tl.constexpr,
    num_sms: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_m_tiles: tl.constexpr = M_aligned // BLOCK_M
    num_n_tiles: tl.constexpr = N // BLOCK_N
    num_tiles_per_batch: tl.constexpr = num_m_tiles * num_n_tiles
    num_tiles: tl.constexpr = B * num_tiles_per_batch

    for tile_id in range(start_pid, num_tiles, num_sms):
        batch_id, m_tile_id, n_tile_id = _get_tile(
            tile_id, num_m_tiles, num_n_tiles, num_tiles_per_batch, TILE_ORDER
        )
        m_start = m_tile_id * BLOCK_M
        n_start = n_tile_id * BLOCK_N
        y_row = batch_id * N + n_start

        for k_block_idx in range(0, K // BLOCK_K):
            index = k_block_idx % num_stages
            phase = k_block_idx // num_stages
            k_start = k_block_idx * BLOCK_K
            tle.gpu.barrier_wait(empty_x[index], phaseIdx=phase)
            tle.gpu.copy(
                x_desc,
                x_smem.slot(index),
                [1, BLOCK_M, BLOCK_K],
                [batch_id, m_start, k_start],
                barrier=full_x[index],
            )
            tle.gpu.barrier_wait(empty_y[index], phaseIdx=phase)
            tle.gpu.copy(
                y_desc,
                y_smem.slot(index),
                [BLOCK_N, BLOCK_K],
                [y_row, k_start],
                barrier=full_y[index],
            )


if HAS_TLE_W8A8_BLOCK_FP8_BMM:

    @libentry()
    @libtuner(
        configs=_get_tle_w8a8_block_fp8_bmm_configs(),
        key=["B", "M_aligned", "N", "K"],
        strategy=["default", "align32", "align32", "align32"],
        policy=_TLEW8A8BlockFP8BMMTuner,
        flagtune_op_name="w8a8_block_fp8_bmm",
        flagtune_expand_op_name="w8a8_block_fp8_bmm",
    )
    @triton.jit
    def w8a8_block_fp8_bmm_kernel(
        x_desc,
        y_desc,
        xs_ptr,
        z_ptr,
        ys_ptr,
        xs_sB: tl.constexpr,
        xs_sM: tl.constexpr,
        xs_sKb: tl.constexpr,
        z_sB: tl.constexpr,
        z_sM: tl.constexpr,
        z_sN: tl.constexpr,
        B: tl.constexpr,
        M: tl.constexpr,
        M_aligned: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        TILE_ORDER: tl.constexpr,
        SWAP_AB: tl.constexpr,
        X_ELEM_BYTES: tl.constexpr,
        Y_ELEM_BYTES: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr,
        num_sms: tl.constexpr,
    ):
        _ = num_warps
        x_smem = tle.gpu.alloc(
            [num_stages, 1, BLOCK_M, BLOCK_K],
            dtype=x_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        y_smem = tle.gpu.alloc(
            [num_stages, BLOCK_N, BLOCK_K],
            dtype=y_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        empty_x = tle.gpu.alloc_barriers(
            num_barriers=num_stages, arrive_count=1, init=tle.gpu.READY
        )
        empty_y = tle.gpu.alloc_barriers(
            num_barriers=num_stages, arrive_count=1, init=tle.gpu.READY
        )
        full_x = tle.gpu.alloc_barriers(
            num_barriers=num_stages,
            arrive_count=1,
            expect_bytes=BLOCK_M * BLOCK_K * X_ELEM_BYTES,
        )
        full_y = tle.gpu.alloc_barriers(
            num_barriers=num_stages,
            arrive_count=1,
            expect_bytes=BLOCK_N * BLOCK_K * Y_ELEM_BYTES,
        )

        tle.gpu.warp_specialize(
            [
                (
                    _tle_w8a8_block_fp8_bmm_compute_partition,
                    (
                        x_smem,
                        y_smem,
                        empty_x,
                        empty_y,
                        full_x,
                        full_y,
                        xs_ptr,
                        z_ptr,
                        ys_ptr,
                        xs_sB,
                        xs_sM,
                        xs_sKb,
                        z_sB,
                        z_sM,
                        z_sN,
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
                        num_stages,
                        num_sms,
                    ),
                ),
                (
                    _tle_w8a8_block_fp8_bmm_load_partition,
                    (
                        x_desc,
                        y_desc,
                        x_smem,
                        y_smem,
                        empty_x,
                        empty_y,
                        full_x,
                        full_y,
                        B,
                        M,
                        M_aligned,
                        N,
                        K,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        TILE_ORDER,
                        num_stages,
                        num_sms,
                    ),
                ),
            ],
            [1],
            [24],
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_fp8_bmm_splitk"),
    key=["B", "M", "N", "K", "stride_xm", "stride_yk"],
    reset_to_zero=["Z"],
    strategy=["default", "align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
    flagtune_op_name="w8a8_block_fp8_bmm",
    flagtune_expand_op_name="w8a8_block_fp8_bmm_splitk",
)
@triton.jit
def w8a8_block_fp8_bmm_kernel_splitk(
    X,
    Y,
    XS,
    Z,
    YS,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_yb: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_yk: tl.constexpr,
    stride_xsb: tl.constexpr,
    stride_xsm: tl.constexpr,
    stride_xsk: tl.constexpr,
    stride_zb: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_ysb: tl.constexpr,
    stride_ysn: tl.constexpr,
    stride_ysk: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n
    batch_id = pid // tiles_per_batch
    local_pid = pid % tiles_per_batch
    pid_m = local_pid // grid_n
    pid_n = local_pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(k_start, k_end):
        offs_k = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)

        x = tl.load(
            X
            + batch_id * stride_xb
            + offs_m[:, None] * stride_xm
            + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        y = tl.load(
            Y
            + batch_id * stride_yb
            + offs_n[None, :] * stride_yn
            + offs_k[:, None] * stride_yk,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        x_s = tl.load(
            XS + batch_id * stride_xsb + offs_m * stride_xsm + k_iter * stride_xsk,
            mask=offs_m < M,
            other=0.0,
        )
        y_s = tl.load(
            YS
            + batch_id * stride_ysb
            + (offs_n // SCALE_BLOCK_N) * stride_ysn
            + k_iter * stride_ysk,
            mask=offs_n < N,
            other=0.0,
        )
        acc += tl.dot(x, y, out_dtype=tl.float32) * x_s[:, None] * y_s[None, :]

    z_ptrs = (
        Z
        + batch_id * stride_zb
        + offs_m[:, None] * stride_zm
        + offs_n[None, :] * stride_zn
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if Z.dtype.element_ty == tl.bfloat16:
        tl.atomic_add(z_ptrs, acc.to(tl.bfloat16), mask=mask)
    elif Z.dtype.element_ty == tl.float16:
        tl.atomic_add(z_ptrs, acc.to(tl.float16), mask=mask)
    else:
        tl.atomic_add(z_ptrs, acc.to(tl.float32), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_fp8_bmm_general"),
    key=["B", "M", "N", "K", "stride_xm", "stride_yk"],
    strategy=["default", "align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
    flagtune_op_name="w8a8_block_fp8_bmm",
    flagtune_expand_op_name="w8a8_block_fp8_bmm_general",
)
@triton.jit
def w8a8_block_fp8_bmm_kernel_general(
    X,
    Y,
    XS,
    Z,
    YS,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_yb: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_yk: tl.constexpr,
    stride_xsb: tl.constexpr,
    stride_xsm: tl.constexpr,
    stride_xsk: tl.constexpr,
    stride_zb: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_ysb: tl.constexpr,
    stride_ysn: tl.constexpr,
    stride_ysk: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    batch_id = (pid // tiles_per_batch).to(tl.int64)
    local_pid = pid % tiles_per_batch
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = local_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    local_group_pid = local_pid % num_pid_in_group
    pid_m = first_pid_m + (local_group_pid % group_size_m)
    pid_n = local_group_pid // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = (k_iter * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)

        x = tl.load(
            X
            + batch_id * stride_xb
            + offs_m[:, None] * stride_xm
            + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        y = tl.load(
            Y
            + batch_id * stride_yb
            + offs_n[None, :] * stride_yn
            + offs_k[:, None] * stride_yk,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        x_s = tl.load(
            XS + batch_id * stride_xsb + offs_m * stride_xsm + k_iter * stride_xsk,
            mask=offs_m < M,
            other=0.0,
        )
        y_s = tl.load(
            YS
            + batch_id * stride_ysb
            + (offs_n // SCALE_BLOCK_N) * stride_ysn
            + k_iter * stride_ysk,
            mask=offs_n < N,
            other=0.0,
        )
        acc += tl.dot(x, y, out_dtype=tl.float32) * x_s[:, None] * y_s[None, :]

    z_ptrs = (
        Z
        + batch_id * stride_zb
        + offs_m[:, None] * stride_zm
        + offs_n[None, :] * stride_zn
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if Z.dtype.element_ty == tl.bfloat16:
        tl.store(z_ptrs, acc.to(tl.bfloat16), mask=mask)
    elif Z.dtype.element_ty == tl.float16:
        tl.store(z_ptrs, acc.to(tl.float16), mask=mask)
    else:
        tl.store(z_ptrs, acc.to(tl.float32), mask=mask)


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
    if B == 1 and M < 300 and N < 2112 and K >= 4096:
        z.zero_()
        splitk_grid = lambda META: (
            B * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )
        w8a8_block_fp8_bmm_kernel_splitk[splitk_grid](
            x,
            y,
            xs,
            z,
            ys,
            B=B,
            M=M,
            N=N,
            K=K,
            stride_xb=x.stride(0),
            stride_xm=x.stride(1),
            stride_xk=x.stride(2),
            stride_yb=y.stride(0),
            stride_yn=y.stride(1),
            stride_yk=y.stride(2),
            stride_xsb=xs.stride(0),
            stride_xsm=xs.stride(1),
            stride_xsk=xs.stride(2),
            stride_zb=z.stride(0),
            stride_zm=z.stride(1),
            stride_zn=z.stride(2),
            stride_ysb=ys.stride(0),
            stride_ysn=ys.stride(1),
            stride_ysk=ys.stride(2),
            SCALE_BLOCK_N=BLOCK_N,
        )
        return z

    if not HAS_TLE_W8A8_BLOCK_FP8_BMM:
        general_grid = lambda META: (
            B * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        w8a8_block_fp8_bmm_kernel_general[general_grid](
            x,
            y,
            xs,
            z,
            ys,
            B=B,
            M=M,
            N=N,
            K=K,
            stride_xb=x.stride(0),
            stride_xm=x.stride(1),
            stride_xk=x.stride(2),
            stride_yb=y.stride(0),
            stride_yn=y.stride(1),
            stride_yk=y.stride(2),
            stride_xsb=xs.stride(0),
            stride_xsm=xs.stride(1),
            stride_xsk=xs.stride(2),
            stride_zb=z.stride(0),
            stride_zm=z.stride(1),
            stride_zn=z.stride(2),
            stride_ysb=ys.stride(0),
            stride_ysn=ys.stride(1),
            stride_ysk=ys.stride(2),
            SCALE_BLOCK_N=BLOCK_N,
        )
        return z

    from triton.tools.tensor_descriptor import TensorDescriptor

    _set_triton_descriptor_allocator(device)
    x_desc = TensorDescriptor.from_tensor(x, block_shape=[1, BLOCK_M, BLOCK_K])

    assert y.is_contiguous(), "y must be contiguous so it can be viewed as (B*N, K)"
    y_flat = y.view(B * N, K)
    y_desc = TensorDescriptor.from_tensor(y_flat, block_shape=[BLOCK_N, BLOCK_K])

    assert xs.ndim == 3 and xs.shape == (B, M, num_kb)
    xs_sB, xs_sM, xs_sKb = xs.stride()
    z_sB, z_sM, z_sN = z.stride()

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    w8a8_block_fp8_bmm_kernel[(num_sms,)](
        x_desc,
        y_desc,
        xs,
        z,
        ys,
        xs_sB=xs_sB,
        xs_sM=xs_sM,
        xs_sKb=xs_sKb,
        z_sB=z_sB,
        z_sM=z_sM,
        z_sN=z_sN,
        B=B,
        M=M,
        M_aligned=M_aligned,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SWAP_AB=SWAP_AB,
        X_ELEM_BYTES=x.element_size(),
        Y_ELEM_BYTES=y.element_size(),
        num_sms=num_sms,
    )

    return z
