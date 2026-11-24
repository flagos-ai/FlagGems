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

"""PPU-specialized batched GEMM with an optional fused bias epilogue."""

import logging
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.ops.bmm import bmm as _generic_bmm
from flag_gems.ops.bmm import bmm_out as _generic_bmm_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from .gemm_utils import (
    _GEMV_PROGRAM_WIDTH,
    _LOAD_A_AIU,
    _LOAD_B_AIU,
    _LOAD_BOTH_AIU,
    _LOAD_REGULAR,
    _PPU_DESCRIPTOR_CHUNK_N,
    _PPU_DESCRIPTOR_MAX_N,
    _PPU_SMS,
    EXPAND_CONFIG_FILENAME,
    HAS_PPU_TLE,
    _aiu_load_mask,
    _configs_from_specs,
    _output_overlaps_inputs,
    _ppu_bucket_strategy,
    _ppu_gemm_tile,
    _ppu_reduction_bucket_strategy,
    _prefer_small_m_kernel,
    _prune_gemm_configs,
    _prune_gemv_configs,
    _prune_split_k_configs,
    _should_use_multi_row_gemv,
    _should_use_row_vector_gemv,
    _split_k_wave_plan,
    tle,
)
from .mm import (
    _dispatch_ppu_gemm,
    _ppu_multi_row_gemv_configs,
    _ppu_narrow_columns_configs,
    _ppu_small_m_configs,
)

logger = logging.getLogger(__name__)


class _PPUBMMRoute(Enum):
    """Executable batched families, including narrow-N MM forwarding."""

    MM = "mm"
    GEMV = "bmm_gemv_kernel_ppu"
    MULTI_ROW_GEMV = "bmm_multi_row_gemv_kernel_ppu"
    SMALL_M = "bmm_small_m_kernel_ppu"
    NARROW_COLUMNS = "bmm_narrow_columns_kernel_ppu"
    NARROW_N = "bmm_narrow_n_kernel_ppu"
    SPLIT_K = "bmm_split_k_kernel_ppu"
    MAIN = "bmm_kernel_ppu"


def _b_transposed_layout(B: torch.Tensor) -> bool:
    return B.ndim == 3 and not B.is_contiguous() and B.transpose(1, 2).is_contiguous()


def _full_k_tiles(args):
    """Whether the selected reduction tile divides the logical K extent."""
    return int(args["K"]) % int(args["BLOCK_K"]) == 0


def _full_m_tiles(args):
    # bmm_small_m_kernel_ppu uses a literal physical BM=16 tile.
    return int(args["M"]) % int(args.get("BLOCK_M", 16)) == 0


def _full_n_tiles(args):
    return int(args["N"]) % int(args["BLOCK_N"]) == 0


def _ppu_bmm_configs():
    """Bounded PPU search covering narrow, balanced, and wide batched GEMMs."""
    specs = (
        # Narrow output tiles reuse the physical MM tile family; padding N=16
        # to a 64-column batched tile wastes most of the matrix-unit work.
        (32, 16, 32, 4, 4, 1, _LOAD_REGULAR),
        (64, 16, 64, 4, 5, 1, _LOAD_BOTH_AIU),
        (128, 16, 64, 4, 5, 1, _LOAD_REGULAR),
        (32, 64, 32, 4, 2, 4, _LOAD_BOTH_AIU),
        # Expanded winner for batch-2 short-M projections.
        (32, 128, 64, 8, 5, 2, _LOAD_A_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_BOTH_AIU),
        (128, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        # Expanded winner for batch-1 long-K projections.
        (128, 128, 64, 8, 4, 4, _LOAD_A_AIU),
        (128, 256, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (256, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (128, 64, 64, 8, 3, 8, _LOAD_BOTH_AIU),
        # Long-reduction, low-wave winner.  The dispatch remains wave-based;
        # this candidate is also available to every neighboring shape bucket.
        (64, 64, 128, 4, 4, 1, _LOAD_B_AIU),
        (64, 128, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 256, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 256, 32, 4, 4, 1, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 8, _LOAD_BOTH_AIU),
        # Expanded FP16 fused-bias winner for wide-N, K=1024 projections.
        (64, 256, 32, 8, 4, 1, _LOAD_A_AIU),
        (128, 128, 32, 4, 5, 1, _LOAD_BOTH_AIU),
        (128, 128, 32, 4, 5, 8, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 4, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 8, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 1, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 8, _LOAD_REGULAR),
        (512, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        (256, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        (256, 64, 32, 8, 4, 4, _LOAD_BOTH_AIU),
        (256, 64, 64, 8, 3, 1, _LOAD_BOTH_AIU),
        (128, 64, 32, 4, 3, 1, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 2, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 8, _LOAD_REGULAR),
        (128, 64, 64, 4, 5, 1, _LOAD_BOTH_AIU),
    )
    return _configs_from_specs(
        specs,
        (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "num_stages",
            "GROUP_M",
            "LOAD_MODE",
        ),
    )


def _ppu_bmm_narrow_n_configs():
    """BN64 tiles for deep batched products with a single output column tile."""
    specs = (
        (64, 64, 32, 4, 4, 1, _LOAD_REGULAR),
        (64, 64, 64, 4, 4, 1, _LOAD_BOTH_AIU),
        (128, 64, 32, 4, 4, 1, _LOAD_REGULAR),
        (128, 64, 64, 4, 5, 1, _LOAD_BOTH_AIU),
        (256, 64, 32, 8, 4, 4, _LOAD_BOTH_AIU),
        (256, 64, 64, 8, 3, 1, _LOAD_BOTH_AIU),
        (256, 64, 64, 8, 3, 1, _LOAD_REGULAR),
        (512, 64, 32, 8, 4, 1, _LOAD_REGULAR),
    )
    return _configs_from_specs(
        specs,
        (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "num_stages",
            "GROUP_M",
            "LOAD_MODE",
        ),
    )


def _ppu_small_m_bmm_configs():
    """The batched BM16 kernel accepts exactly the MM BM16 candidates."""
    return _ppu_small_m_configs()


def _ppu_multi_row_bmm_configs():
    return _ppu_multi_row_gemv_configs()


def _ppu_narrow_columns_bmm_configs():
    return _ppu_narrow_columns_configs()


def _ppu_split_k_bmm_configs():
    """Correctness-first batched split-K candidates with bounded workspaces."""
    specs = (
        (2, 32, 128, 64, 4, 4, _LOAD_REGULAR, False),
        (4, 32, 128, 64, 4, 4, _LOAD_REGULAR, False),
        (2, 64, 128, 64, 4, 4, _LOAD_REGULAR, False),
        (4, 64, 128, 64, 4, 4, _LOAD_REGULAR, False),
        (2, 64, 256, 64, 4, 4, _LOAD_REGULAR, False),
        (4, 64, 256, 64, 4, 4, _LOAD_REGULAR, False),
        (2, 128, 128, 32, 4, 4, _LOAD_REGULAR, False),
        (4, 128, 128, 32, 4, 4, _LOAD_REGULAR, False),
        (2, 64, 256, 32, 4, 4, _LOAD_A_AIU, False),
        (4, 64, 256, 32, 4, 4, _LOAD_A_AIU, False),
    )
    return _configs_from_specs(
        specs,
        (
            "SPLIT_K",
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_stages",
            "num_warps",
            "LOAD_MODE",
            "INTERLEAVED",
        ),
    )


def _prune_bmm_small_m_configs(configs, named_args, **kwargs):
    """Apply GEMM legality to a masked BM16 tile, including scalar M."""
    args = dict(named_args)
    if args.get("M") is not None:
        # BM16 uses MASK_M=True and is valid with fewer than four rows. The
        # generic BM>=32 underfill rule would otherwise reject every config
        # and let its fallback reintroduce unsupported B-AIU load modes.
        args["M"] = max(4, int(args["M"]))
    return _prune_gemm_configs(configs, args, **kwargs)


def _ppu_split_k_bmm_reduce_configs():
    return [
        triton.Config({"BLOCK": block, "VEC": vec}, num_warps=warps, num_stages=1)
        for block, vec, warps in ((128, 8, 4), (256, 4, 4))
    ]


def _ppu_batched_gemv_configs():
    """Regular-load reduction configs for batched row/column GEMV."""
    specs = (
        (1, 256, 4, 3),
        (1, 512, 4, 3),
        (2, 256, 4, 3),
        (2, 512, 4, 3),
        (4, 256, 4, 3),
        (4, 512, 8, 3),
        (8, 128, 4, 3),
        (8, 256, 8, 3),
        (8, 512, 8, 3),
        (16, 128, 8, 3),
        (16, 256, 8, 3),
        (16, 512, 8, 3),
        (8, 1024, 8, 3),
        (16, 1024, 8, 3),
        (16, 2048, 8, 3),
    )
    return [
        triton.Config(
            {"BLOCK_M": block_m, "BLOCK_K": block_k},
            num_warps=warps,
            num_stages=stages,
        )
        for block_m, block_k, warps, stages in specs
    ]


if HAS_PPU_TLE:

    @libentry()
    @libtuner(
        configs=_ppu_batched_gemv_configs(),
        key=[
            "FUSE_BIAS",
            "READ_BIAS",
            "B_TRANSPOSED",
            "ROW_VECTOR",
            "batch",
            "OUT_SIZE",
            "K",
        ],
        strategy=[
            "default",
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemv_configs},
        warmup=5,
        rep=20,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_gemv_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_gemv_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        OUT_SIZE,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        ROW_VECTOR: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
        READ_BIAS: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Batched GEMV for either ``[M,K]@[K,1]`` or ``[1,K]@[K,N]``."""
        pid_batch = tl.program_id(1).to(tl.int64)
        rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)

        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        if ROW_VECTOR:
            # Keep output columns on the contiguous inner tensor dimension;
            # B is row-major [K, N], so this maps adjacent N values to adjacent
            # lanes before reducing the leading K dimension.
            acc = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
            for k_start in tl.range(0, K, BLOCK_K):
                ks = (k_start + offs_k).to(tl.int64)
                a = tl.load(
                    A + ks * stride_ak,
                    mask=ks < K,
                    other=0.0,
                )
                b = tl.load(
                    B
                    + ks[:, None] * stride_bk
                    + rows[None, :].to(tl.int64) * stride_bn,
                    mask=(ks[:, None] < K) & (rows[None, :] < OUT_SIZE),
                    other=0.0,
                )
                acc += b.to(tl.float32) * a[:, None].to(tl.float32)
            result = tl.sum(acc, axis=0)
        else:
            # A is row-major [M, K], so K is already the contiguous inner
            # dimension for a block of output rows.
            acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            for k_start in tl.range(0, K, BLOCK_K):
                ks = (k_start + offs_k).to(tl.int64)
                a = tl.load(
                    A
                    + rows[:, None].to(tl.int64) * stride_am
                    + ks[None, :] * stride_ak,
                    mask=(rows[:, None] < OUT_SIZE) & (ks[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    B + ks * stride_bk,
                    mask=ks < K,
                    other=0.0,
                )
                acc += a.to(tl.float32) * b[None, :].to(tl.float32)
            result = tl.sum(acc, axis=1)

        if ROW_VECTOR:
            c_ptrs = C + pid_batch * stride_cb + rows * stride_cn
            bias_ptrs = Bias + pid_batch * stride_bias_b + rows * stride_bias_n
        else:
            c_ptrs = C + pid_batch * stride_cb + rows * stride_cm
            bias_ptrs = Bias + pid_batch * stride_bias_b + rows * stride_bias_m
        mask = rows < OUT_SIZE
        result *= alpha
        if READ_BIAS:
            result += beta * tl.load(bias_ptrs, mask=mask, other=0.0)
        tl.store(c_ptrs, result.to(C.dtype.element_ty), mask=mask)

    @libentry()
    @libtuner(
        configs=_ppu_multi_row_bmm_configs(),
        key=["FUSE_BIAS", "READ_BIAS", "B_TRANSPOSED", "batch", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemv_configs},
        warmup=5,
        rep=20,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_multi_row_gemv",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_multi_row_gemv_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        FUSE_BIAS: tl.constexpr,
        READ_BIAS: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        """One scalar row reduction per batched output-column tile."""
        pid_batch = tl.program_id(2).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)
        cols = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            ks = (k_start + offs_k).to(tl.int64)
            a = tl.load(A + pid_m * stride_am + ks * stride_ak, mask=ks < K, other=0.0)
            b = tl.load(
                B + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                mask=(ks[:, None] < K) & (cols[None, :] < N),
                other=0.0,
            )
            acc += tl.sum(b.to(tl.float32) * a[:, None].to(tl.float32), axis=0)
        result = alpha * acc
        mask = cols < N
        if READ_BIAS:
            result += beta * tl.load(
                Bias + pid_m * stride_bias_m + cols * stride_bias_n,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            C + pid_m * stride_cm + cols * stride_cn,
            result.to(C.dtype.element_ty),
            mask=mask,
        )

    @libentry()
    @libtuner(
        configs=_ppu_narrow_columns_bmm_configs(),
        key=["FUSE_BIAS", "READ_BIAS", "B_TRANSPOSED", "batch", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemv_configs},
        warmup=5,
        rep=20,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_narrow_columns",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_narrow_columns_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        FUSE_BIAS: tl.constexpr,
        READ_BIAS: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        """Fuse N scalar column reductions for each batch in one launch."""
        pid_batch = tl.program_id(2).to(tl.int64)
        pid_n = tl.program_id(0).to(tl.int64)
        rows = (tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = rows < M
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            ks = (k_start + offs_k).to(tl.int64)
            a = tl.load(
                A + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=row_mask[:, None] & (ks[None, :] < K),
                other=0.0,
            )
            b = tl.load(B + ks * stride_bk + pid_n * stride_bn, mask=ks < K, other=0.0)
            acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)
        result = alpha * acc
        if READ_BIAS:
            result += beta * tl.load(
                Bias + rows * stride_bias_m + pid_n * stride_bias_n,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            C + rows * stride_cm + pid_n * stride_cn,
            result.to(C.dtype.element_ty),
            mask=row_mask & (pid_n < N),
        )

    @libentry()
    @libtuner(
        configs=_ppu_bmm_configs(),
        key=[
            "FUSE_BIAS",
            "READ_BIAS",
            "B_TRANSPOSED",
            "aiu_load_mask",
            "batch",
            "M",
            "N",
            "K",
        ],
        strategy=[
            "default",
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        ALIGNED_A_512X128: tl.constexpr,
        ALIGNED_B_128X128: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
        READ_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_batch = tl.program_id(1).to(tl.int64)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b

        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + pid % group_size
        pid_n = (pid % width) // group_size
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            # Modes 0/1/2 require both/A/B AIU respectively; 3 is regular.
            (
                LOAD_MODE
                if (
                    LOAD_MODE == 3
                    or (LOAD_MODE == 0 and aiu_load_mask == 3)
                    or (LOAD_MODE == 1 and aiu_load_mask & 1)
                    or (LOAD_MODE == 2 and aiu_load_mask & 2)
                )
                else 3
            ),
            B_TRANSPOSED,
            PIPE_STAGES,
            False,
            ALIGNED_A_512X128,
            ALIGNED_B_128X128,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            EVEN_M,
            EVEN_N,
            FUSE_BIAS,
            READ_BIAS,
        )

    @libentry()
    @libtuner(
        configs=_ppu_bmm_narrow_n_configs(),
        key=[
            "FUSE_BIAS",
            "READ_BIAS",
            "B_TRANSPOSED",
            "aiu_load_mask",
            "batch",
            "M",
            "N",
            "K",
        ],
        strategy=[
            "default",
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_narrow_n",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_narrow_n_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        ALIGNED_A_512X128: tl.constexpr,
        ALIGNED_B_128X128: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
        READ_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_batch = tl.program_id(1).to(tl.int64)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b

        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + pid % group_size
        pid_n = (pid % width) // group_size
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            # Modes 0/1/2 require both/A/B AIU respectively; 3 is regular.
            (
                LOAD_MODE
                if (
                    LOAD_MODE == 3
                    or (LOAD_MODE == 0 and aiu_load_mask == 3)
                    or (LOAD_MODE == 1 and aiu_load_mask & 1)
                    or (LOAD_MODE == 2 and aiu_load_mask & 2)
                )
                else 3
            ),
            B_TRANSPOSED,
            PIPE_STAGES,
            False,
            ALIGNED_A_512X128,
            ALIGNED_B_128X128,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            EVEN_M,
            EVEN_N,
            FUSE_BIAS,
            READ_BIAS,
        )

    @libentry()
    @libtuner(
        configs=_ppu_small_m_bmm_configs(),
        key=[
            "FUSE_BIAS",
            "READ_BIAS",
            "B_TRANSPOSED",
            "aiu_load_mask",
            "batch",
            "M",
            "N",
            "K",
        ],
        strategy=[
            "default",
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_bmm_small_m_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_small_m",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_small_m_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
        READ_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_batch = tl.program_id(1).to(tl.int64)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b

        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            16,
            BLOCK_N,
            BLOCK_K,
            # The guard also protects launches using a previously cached
            # tuner winner from before the stricter small-M pruning rule.
            (
                LOAD_MODE
                if (
                    LOAD_MODE == 3
                    or (LOAD_MODE == 0 and aiu_load_mask == 3)
                    or (LOAD_MODE == 1 and aiu_load_mask & 1)
                    or (LOAD_MODE == 2 and aiu_load_mask & 2)
                )
                else 3
            ),
            B_TRANSPOSED,
            PIPE_STAGES,
            True,
            False,
            False,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            False,
            EVEN_N,
            FUSE_BIAS,
            READ_BIAS,
        )

    @libentry()
    @libtuner(
        configs=_ppu_split_k_bmm_configs(),
        key=["B_TRANSPOSED", "aiu_load_mask", "batch", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_split_k_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_split_k",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def bmm_split_k_kernel_ppu(
        A,
        B,
        Workspace,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_MN: tl.constexpr,
    ):
        """Write FP32 batched K slices without output atomics."""
        # Keep the split dimension in program_id(0); the PPU compiler's
        # legacy divergence pass does not accept split_id in program_id(1).
        linear_id = tl.program_id(0)
        split_id = linear_id % SPLIT_K
        tile_id = linear_id // SPLIT_K
        batch_id = tl.program_id(1).to(tl.int64)
        A += batch_id * stride_ab
        B += batch_id * stride_bb
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        k_per_split = K // SPLIT_K
        if INTERLEAVED:
            k_per_split = tl.cdiv(K, BLOCK_K * SPLIT_K) * BLOCK_K
            k_begin = split_id * BLOCK_K
            k_advance = BLOCK_K * SPLIT_K
        else:
            k_begin = split_id * k_per_split
            k_advance = BLOCK_K
        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, k_begin),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        if B_TRANSPOSED:
            b_block_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(k_begin, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
        else:
            b_block_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(k_begin, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
        load_mode = (
            LOAD_MODE
            if (LOAD_MODE == 3 or (LOAD_MODE == 1 and aiu_load_mask & 1))
            else 3
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_offset in tl.range(0, k_per_split, BLOCK_K, num_stages=PIPE_STAGES):
            if INTERLEAVED:
                offs_k = (k_begin + k_offset * SPLIT_K + tl.arange(0, BLOCK_K)).to(
                    tl.int64
                )
            else:
                offs_k = (k_begin + k_offset + tl.arange(0, BLOCK_K)).to(tl.int64)
            if load_mode == 1:
                a = tle.load(
                    a_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                a = tl.load(
                    A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0,
                )
            b = tl.load(
                B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
            a_block_ptr = tl.advance(a_block_ptr, (0, k_advance))
            b_block_ptr = tl.advance(b_block_ptr, (k_advance, 0))
        matrix_elements = M.to(tl.int64) * N
        workspace_ptrs = (
            Workspace
            + batch_id * SPLIT_K * matrix_elements
            + split_id.to(tl.int64) * matrix_elements
            + offs_m[:, None] * N
            + offs_n[None, :]
        )
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if EVEN_MN:
            tl.store(workspace_ptrs, acc)
        else:
            tl.store(workspace_ptrs, acc, mask=mask)

    @libtuner(
        configs=_ppu_split_k_bmm_reduce_configs(),
        key=["FUSE_BIAS", "READ_BIAS", "M", "N", "total_elements", "SPLIT_K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            "default",
        ],
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_split_k_reduce",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_split_k_reduce_kernel_ppu(
        Workspace,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        total_elements,
        SPLIT_K: tl.constexpr,
        BLOCK: tl.constexpr,
        VEC: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
        READ_BIAS: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0).to(tl.int64) * BLOCK * VEC
            + tl.arange(0, BLOCK)[:, None] * VEC
            + tl.arange(0, VEC)[None, :]
        )
        offsets = tl.max_contiguous(offsets, (1, VEC))
        mask = offsets < total_elements
        matrix_elements = tl.full((), M, tl.int64) * N
        batch_ids = offsets // matrix_elements
        matrix_offsets = offsets % matrix_elements
        workspace_ptrs = (
            Workspace + batch_ids * SPLIT_K * matrix_elements + matrix_offsets
        )
        acc = tl.zeros((BLOCK, VEC), dtype=tl.float32)
        for _ in range(SPLIT_K):
            if EVEN_N:
                acc += tl.load(workspace_ptrs)
            else:
                acc += tl.load(workspace_ptrs, mask=mask, other=0.0)
            workspace_ptrs += matrix_elements

        offs_m = matrix_offsets // N
        offs_n = matrix_offsets % N
        c_ptrs = C + batch_ids * stride_cb + offs_m * stride_cm + offs_n * stride_cn
        acc *= alpha
        if READ_BIAS:
            bias_ptrs = (
                Bias
                + batch_ids * stride_bias_b
                + offs_m * stride_bias_m
                + offs_n * stride_bias_n
            )
            bias_value = tl.load(bias_ptrs, mask=mask, other=0.0)
            acc += beta * bias_value
        if EVEN_N:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


def _can_use_ppu_bmm_inputs(A: torch.Tensor, B: torch.Tensor) -> bool:
    if not (
        HAS_PPU_TLE
        and A.ndim == B.ndim == 3
        and A.dtype in (torch.float16, torch.bfloat16)
        and B.dtype == A.dtype
        and A.device == B.device
        and A.is_contiguous()
        and (B.is_contiguous() or (_b_transposed_layout(B) and A.shape[-1] % 128 == 0))
    ):
        return False

    batch, M, K = A.shape
    batch_b, b_k, N = B.shape
    return batch == batch_b and batch > 0 and M > 0 and N > 0 and K == b_k and K > 0


def _can_use_ppu_bmm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        _can_use_ppu_bmm_inputs(A, B)
        and out.ndim == 3
        and out.shape == (A.shape[0], A.shape[1], B.shape[2])
        and out.dtype == A.dtype
        and out.device == A.device
        and out.is_contiguous()
    )


def _should_use_ppu_bmm_gemv(batch: int, M: int, N: int, K: int) -> bool:
    """Choose batched GEMV from a monotonic scalar-reduction work model."""
    return N == 1 or (M == 1 and _should_use_row_vector_gemv(batch, N, K))


def _select_ppu_bmm_route(
    batch: int, M: int, N: int, K: int, *, b_transposed: bool, fuse_bias: bool = False
) -> _PPUBMMRoute:
    """Choose one physical kernel family without launching a tensor operation."""
    if batch == 1:
        # Very tall, shallow NT BMM is throughput-bound. The batched main
        # family has a measured larger-row winner at the same physical M
        # boundary where the common pruner excludes narrow launch tiles.
        if not fuse_bias and b_transposed and M > 32768 and K <= 1024 and N >= 128:
            return _PPUBMMRoute.MAIN
        return _PPUBMMRoute.MM
    if _should_use_ppu_bmm_gemv(batch, M, N, K):
        # A one-row product can still launch hundreds of scalar GEMV output
        # programs. A masked BM16 dot tile wins for wide shallow reductions
        # in either layout, and for wider NN reductions whose scalar loads
        # cannot amortize their program count. Deep NT reductions retain GEMV.
        if M == 1 and (
            (N >= 2048 and K <= 256)
            or (not b_transposed and N >= 6144 and K <= 4096)
            or (not b_transposed and N >= 2560 and batch * K >= 8192 and K <= 4096)
        ):
            return _PPUBMMRoute.SMALL_M
        return _PPUBMMRoute.GEMV
    if (
        batch <= 8
        and b_transposed
        and 2 <= M <= 4
        and _should_use_multi_row_gemv(M, N, K)
        and batch * M * triton.cdiv(N, _GEMV_PROGRAM_WIDTH) < 8 * _PPU_SMS
    ):
        return _PPUBMMRoute.MULTI_ROW_GEMV
    if batch <= 8 and not b_transposed and 64 < M <= 320 and N <= 8 and K >= 8 * 1024:
        return _PPUBMMRoute.NARROW_COLUMNS
    # A few narrow outputs and a very deep reduction leave too little grid
    # parallelism for the batched BM16/BM32 dots. The tuned 2-D MM narrow-N
    # path is much faster even after one launch per batch. Bound the total
    # launched row work so larger batches do not spend too much time in the
    # extra MM launches.
    mm_row_budget = 1024 if batch <= 4 else 512
    if (
        batch <= 8
        and N <= 4
        and K >= 8 * 1024
        and (batch * M <= mm_row_budget or (not b_transposed and batch * M <= 2048))
    ):
        return _PPUBMMRoute.MM
    # BN64 with larger M tiles is absent from the general Expanded space.
    # Keep its tuning local to this deep, single-column-tile family.
    if N == 64 and M >= 256 and K >= 2048:
        return _PPUBMMRoute.NARROW_N
    # Once the output spans many N tiles, BM32 wastes most AIU rows for a
    # one-tile M dimension. The BM16 family is faster in NN and avoids a
    # severe NT bandwidth regression; the runner chunks descriptor-width N.
    if M <= 16 and N >= 8 * 1024:
        return _PPUBMMRoute.SMALL_M
    if M >= 32 and N > _PPU_DESCRIPTOR_MAX_N:
        return _PPUBMMRoute.MAIN
    if _prefer_small_m_kernel(batch, M, N, K):
        return _PPUBMMRoute.SMALL_M
    # Only a small B=2 NN output grid amortizes the FP32 workspace and
    # reduction launch. Wider outputs and larger batches already expose
    # enough independent GEMM tiles to favor the ordinary batched kernel.
    if (
        batch == 2
        and not b_transposed
        and M >= 64
        and 256 <= N <= 384
        and K >= 7 * 1024
        and batch * triton.cdiv(M, 64) * triton.cdiv(N, 128) <= 12
        and _split_k_wave_plan(batch, M, N, K) is not None
    ):
        return _PPUBMMRoute.SPLIT_K
    return _PPUBMMRoute.MAIN


def _run_ppu_bmm_reduce(workspace, out, bias, alpha, beta, split_k, fuse_bias):
    """Accumulate workspace values in FP32 and apply the batched epilogue."""
    batch, M, N = out.shape
    total_elements = batch * M * N
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK"] * meta["VEC"]),)
    with torch_device_fn.device(out.device):
        bmm_split_k_reduce_kernel_ppu[grid](
            workspace,
            out,
            bias,
            alpha,
            beta,
            M,
            N,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            total_elements,
            SPLIT_K=split_k,
            FUSE_BIAS=fuse_bias,
            READ_BIAS=fuse_bias and beta != 0,
            EVEN_N=False,
        )
    return out


def _run_ppu_bmm_as_mm(A, B, out, bias, alpha, beta, fuse_bias):
    for batch_id in range(A.shape[0]):
        routed = _dispatch_ppu_gemm(
            A[batch_id],
            B[batch_id],
            out[batch_id],
            bias=bias[batch_id] if fuse_bias else None,
            alpha=alpha,
            beta=beta,
        )
        if routed is None:
            return None
    return out


def _run_ppu_bmm_gemv(A, B, out, bias, alpha, beta, fuse_bias):
    batch, M, K = A.shape
    N = B.shape[2]
    read_bias = fuse_bias and beta != 0
    b_transposed = _b_transposed_layout(B)
    row_vector = M == 1
    out_size = N if row_vector else M
    grid = lambda META: (
        triton.cdiv(out_size, META["BLOCK_M"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_gemv_kernel_ppu[grid](
            A,
            B,
            out,
            bias,
            alpha,
            beta,
            batch,
            out_size,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            ROW_VECTOR=row_vector,
            FUSE_BIAS=fuse_bias,
            READ_BIAS=read_bias,
            B_TRANSPOSED=b_transposed,
        )
    return out


def _launch_ppu_bmm_small_m(A, B, out, bias, alpha, beta, fuse_bias, *, b_transposed):
    batch, M, K = A.shape
    N = B.shape[2]
    read_bias = fuse_bias and beta != 0
    kernel = bmm_small_m_kernel_ppu

    def small_grid(meta):
        return (
            triton.cdiv(M, 16) * triton.cdiv(N, meta["BLOCK_N"]),
            batch,
        )

    with torch_device_fn.device(A.device):
        kernel[small_grid](
            A,
            B,
            out,
            bias,
            alpha,
            beta,
            batch,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            B_TRANSPOSED=b_transposed,
            aiu_load_mask=_aiu_load_mask(A, B),
            EVEN_K=K % 128 == 0,
            EVEN_N=N % 1024 == 0,
            FUSE_BIAS=fuse_bias,
            READ_BIAS=read_bias,
        )
    return out


def _run_ppu_bmm_small_m(A, B, out, bias, alpha, beta, fuse_bias):
    """Run BM16 tiles, chunking output columns beyond the descriptor limit."""
    batch, M, _ = A.shape
    N = B.shape[2]
    b_transposed = _b_transposed_layout(B)
    if N <= _PPU_DESCRIPTOR_MAX_N:
        return _launch_ppu_bmm_small_m(
            A,
            B,
            out,
            bias,
            alpha,
            beta,
            fuse_bias,
            b_transposed=b_transposed,
        )

    # The physical NN row pitch remains ultra-wide even after a logical
    # slice. Fuse bias through the FP32 epilogue after a bounded BF16 GEMM
    # workspace to avoid the fused-load compiler error on that stride.
    separate_epilogue = (
        fuse_bias
        and beta != 0
        and not b_transposed
        and B.stride(-2) > _PPU_DESCRIPTOR_MAX_N
    )
    chunk_n = _PPU_DESCRIPTOR_CHUNK_N
    if separate_epilogue:
        workspace_budget = 128 * 1024 * 1024
        max_cols = workspace_budget // (batch * M * out.element_size())
        chunk_n = min(chunk_n, max(128, (max_cols // 128) * 128))

    for n_start in range(0, N, chunk_n):
        width = min(chunk_n, N - n_start)
        b_chunk = B[:, :, n_start : n_start + width]
        out_chunk = out[:, :, n_start : n_start + width]
        bias_chunk = bias[:, :, n_start : n_start + width]
        workspace = (
            torch.empty((batch, 1, M, width), device=out.device, dtype=out.dtype)
            if separate_epilogue
            else None
        )
        gemm_out = workspace[:, 0] if separate_epilogue else out_chunk
        _launch_ppu_bmm_small_m(
            A,
            b_chunk,
            gemm_out,
            gemm_out if separate_epilogue else bias_chunk,
            1.0 if separate_epilogue else alpha,
            0.0 if separate_epilogue else beta,
            fuse_bias and not separate_epilogue,
            b_transposed=b_transposed,
        )
        if separate_epilogue:
            _run_ppu_bmm_reduce(workspace, out_chunk, bias_chunk, alpha, beta, 1, True)
    return out


def _run_ppu_bmm_multi_row_gemv(A, B, out, bias, alpha, beta, fuse_bias):
    batch, M, K = A.shape
    N = B.shape[2]
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), M, batch)
    with torch_device_fn.device(A.device):
        bmm_multi_row_gemv_kernel_ppu[grid](
            A,
            B,
            out,
            bias,
            alpha,
            beta,
            batch,
            M,
            N,
            K,
            *A.stride(),
            *B.stride(),
            *out.stride(),
            *bias.stride(),
            FUSE_BIAS=fuse_bias,
            READ_BIAS=fuse_bias and beta != 0,
            B_TRANSPOSED=_b_transposed_layout(B),
        )
    return out


def _run_ppu_bmm_narrow_columns(A, B, out, bias, alpha, beta, fuse_bias):
    batch, M, K = A.shape
    N = B.shape[2]
    grid = lambda meta: (N, triton.cdiv(M, meta["BLOCK_M"]), batch)
    with torch_device_fn.device(A.device):
        bmm_narrow_columns_kernel_ppu[grid](
            A,
            B,
            out,
            bias,
            alpha,
            beta,
            batch,
            M,
            N,
            K,
            *A.stride(),
            *B.stride(),
            *out.stride(),
            *bias.stride(),
            FUSE_BIAS=fuse_bias,
            READ_BIAS=fuse_bias and beta != 0,
            B_TRANSPOSED=_b_transposed_layout(B),
        )
    return out


def _run_ppu_bmm_main(
    A,
    B,
    out,
    bias,
    alpha,
    beta,
    fuse_bias,
    *,
    kernel=bmm_kernel_ppu if HAS_PPU_TLE else None,
):
    batch, M, K = A.shape
    N = B.shape[2]
    read_bias = fuse_bias and beta != 0
    b_transposed = _b_transposed_layout(B)
    if N > _PPU_DESCRIPTOR_MAX_N:
        # TLE descriptors have a 17-bit width limit.  Present descriptor-safe
        # views to the same LibTuner kernel so staged AIU loads remain
        # available for arbitrary ultra-wide outputs, including the tail.
        chunk_n = _PPU_DESCRIPTOR_CHUNK_N
        separate_epilogue = (
            read_bias and not b_transposed and B.stride(-2) > _PPU_DESCRIPTOR_MAX_N
        )
        if separate_epilogue:
            # The fused path miscompiles with an oversized physical NN row
            # pitch. Keep B in place, compute each chunk in output precision,
            # then reuse the FP32 split reducer for the bias epilogue.
            workspace_budget = 128 * 1024 * 1024
            max_cols = workspace_budget // (batch * M * out.element_size())
            chunk_n = min(chunk_n, max(128, (max_cols // 128) * 128))
        for n_start in range(0, N, chunk_n):
            width = min(chunk_n, N - n_start)
            b_chunk = B[:, :, n_start : n_start + width]
            out_chunk = out[:, :, n_start : n_start + width]
            bias_chunk = bias[:, :, n_start : n_start + width]
            workspace = (
                torch.empty((batch, 1, M, width), device=out.device, dtype=out.dtype)
                if separate_epilogue
                else None
            )
            gemm_out = workspace[:, 0] if separate_epilogue else out_chunk
            gemm_bias = gemm_out if separate_epilogue else bias_chunk

            def chunk_grid(meta):
                return (
                    triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(width, meta["BLOCK_N"]),
                    batch,
                )

            with torch_device_fn.device(A.device):
                kernel[chunk_grid](
                    A,
                    b_chunk,
                    gemm_out,
                    gemm_bias,
                    1.0 if separate_epilogue else alpha,
                    0.0 if separate_epilogue else beta,
                    batch,
                    M,
                    width,
                    K,
                    A.stride(0),
                    A.stride(1),
                    A.stride(2),
                    b_chunk.stride(0),
                    b_chunk.stride(1),
                    b_chunk.stride(2),
                    gemm_out.stride(0),
                    gemm_out.stride(1),
                    gemm_out.stride(2),
                    gemm_bias.stride(0),
                    gemm_bias.stride(1),
                    gemm_bias.stride(2),
                    B_TRANSPOSED=b_transposed,
                    aiu_load_mask=_aiu_load_mask(A, b_chunk),
                    ALIGNED_A_512X128=M % 512 == 0 and K % 128 == 0,
                    ALIGNED_B_128X128=width % 128 == 0 and K % 128 == 0,
                    EVEN_K=K % 128 == 0,
                    EVEN_M=M % 512 == 0,
                    EVEN_N=width % 1024 == 0,
                    FUSE_BIAS=fuse_bias and not separate_epilogue,
                    READ_BIAS=read_bias and not separate_epilogue,
                )
            if separate_epilogue:
                _run_ppu_bmm_reduce(
                    workspace, out_chunk, bias_chunk, alpha, beta, 1, True
                )
        return out

    def grid(meta):
        block_m = meta["BLOCK_M"]
        return (
            triton.cdiv(M, block_m) * triton.cdiv(N, meta["BLOCK_N"]),
            batch,
        )

    with torch_device_fn.device(A.device):
        kernel[grid](
            A,
            B,
            out,
            bias,
            alpha,
            beta,
            batch,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            B_TRANSPOSED=b_transposed,
            aiu_load_mask=_aiu_load_mask(A, B),
            ALIGNED_A_512X128=M % 512 == 0 and K % 128 == 0,
            ALIGNED_B_128X128=N % 128 == 0 and K % 128 == 0,
            EVEN_K=K % 128 == 0,
            EVEN_N=N % 1024 == 0,
            FUSE_BIAS=fuse_bias,
            READ_BIAS=read_bias,
            EVEN_M=M % 512 == 0,
        )
    return out


def _run_ppu_bmm_narrow_n(A, B, out, bias, alpha, beta, fuse_bias):
    return _run_ppu_bmm_main(
        A, B, out, bias, alpha, beta, fuse_bias, kernel=bmm_narrow_n_kernel_ppu
    )


def _run_ppu_bmm_split_k(A, B, out, bias, alpha, beta, fuse_bias):
    batch, M, K = A.shape
    N = B.shape[2]
    max_split = 4 if 4 * batch * M * N * 4 <= 128 * 1024 * 1024 else 2
    workspace = torch.empty(
        (batch, max_split, M, N), device=out.device, dtype=torch.float32
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"])
        * triton.cdiv(N, meta["BLOCK_N"])
        * meta["SPLIT_K"],
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_split_k_kernel_ppu[grid](
            A,
            B,
            workspace,
            batch,
            M,
            N,
            K,
            *A.stride(),
            *B.stride(),
            B_TRANSPOSED=_b_transposed_layout(B),
            aiu_load_mask=_aiu_load_mask(A, B),
            EVEN_MN=False,
        )
        tuner = bmm_split_k_kernel_ppu
        while not hasattr(tuner, "best_config"):
            tuner = getattr(tuner, "fn", None)
            if tuner is None:
                raise RuntimeError("BMM split-K tuner did not expose best_config")
        split_k = tuner.best_config.kwargs["SPLIT_K"]
        _run_ppu_bmm_reduce(workspace, out, bias, alpha, beta, split_k, fuse_bias)
    return out


_PPU_BMM_RUNNERS = {
    _PPUBMMRoute.MM: _run_ppu_bmm_as_mm,
    _PPUBMMRoute.GEMV: _run_ppu_bmm_gemv,
    _PPUBMMRoute.MULTI_ROW_GEMV: _run_ppu_bmm_multi_row_gemv,
    _PPUBMMRoute.SMALL_M: _run_ppu_bmm_small_m,
    _PPUBMMRoute.NARROW_COLUMNS: _run_ppu_bmm_narrow_columns,
    _PPUBMMRoute.NARROW_N: _run_ppu_bmm_narrow_n,
    _PPUBMMRoute.SPLIT_K: _run_ppu_bmm_split_k,
    _PPUBMMRoute.MAIN: _run_ppu_bmm_main,
}


def _dispatch_ppu_bmm(A, B, out, *, bias=None, alpha=1.0, beta=0.0):
    """Broadcast the epilogue input, select a route, and call its runner."""
    batch, M, K = A.shape
    N = B.shape[2]
    expanded_bias = bias.broadcast_to((batch, M, N)) if bias is not None else out
    route = _select_ppu_bmm_route(
        batch,
        M,
        N,
        K,
        b_transposed=_b_transposed_layout(B),
        fuse_bias=bias is not None,
    )
    return _PPU_BMM_RUNNERS[route](
        A, B, out, expanded_bias, alpha, beta, bias is not None
    )


def bmm(A, B):
    logger.debug("GEMS_THEAD BMM")
    if A.ndim == B.ndim == 3 and A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            dtype=A.dtype,
            device=A.device,
        )
        if _can_use_ppu_bmm(A, B, out):
            return _dispatch_ppu_bmm(A, B, out)
    return _generic_bmm(A, B)


def bmm_out(A, B, out):
    logger.debug("GEMS_THEAD BMM_OUT")
    if _can_use_ppu_bmm(A, B, out):
        if _output_overlaps_inputs(out, A, B):
            return out.copy_(bmm(A, B))
        return _dispatch_ppu_bmm(A, B, out)
    return _generic_bmm_out(A, B, out)


__all__ = ["bmm", "bmm_out"]
