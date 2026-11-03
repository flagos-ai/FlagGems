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
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

# Reuse the warp-specialized mm building blocks: the TLE tile pipeline, the
# column-panel rasterization helper and the tuned 256x256x64 tile shape are
# shared with the single-GEMM path, only the batch bookkeeping differs.
from .mm import (  # noqa: E402
    _WS_BLOCK_K,
    _WS_BLOCK_M,
    _WS_BLOCK_N,
    _WS_NUM_WARPS,
    _WS_PANEL_WIDTH,
    _WS_STAGES,
    TLE_WS_ON,
    _ws_rasterization_2d_column,
)

try:
    import triton.experimental.tle.language as tle
except ImportError:  # pragma: no cover - depends on the installed Triton build
    tle = None

logger = logging.getLogger(__name__)

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "bmm_mthreads_expand.yaml")
)


def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, N, K):
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and N % 8 == 0
        and K % 8 == 0
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
)
@triton.heuristics(runtime.get_heuristic_config("bmm"))
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    if IS_FP64:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a)
        b = tl.load(b_ptrs, mask_b)

        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def bmm_fma(A, B, C=None):
    logger.debug("GEMS_MTHREADS BMM_FMA")
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = C
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K, IS_FP64=A.dtype == torch.float64)
    return out


def bmm_sqmma_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_SIZE_K"], nargs["BLOCK_SIZE_N"]]
    nargs["c_desc"].block_shape = [nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"]]


# bmm_sqmma_kernel can only fold a partial M tile back onto the previous one
# while a whole BLOCK_SIZE_M window still fits inside the batch, so it needs the
# masked-store fallback for every M below the largest BLOCK_SIZE_M in the tuned
# config list below.  Keep the two in sync.
_SQMMA_MAX_BLOCK_M = 128


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_M": 4,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 256,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        ),
    ],
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=5,
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm_sqmma",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=bmm_sqmma_descriptor_pre_hook,
)
@triton.jit
def bmm_sqmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    c_ptr,
    batch,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NEEDS_M_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_index = tl.program_id(axis=1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    # When M is not a multiple of BLOCK_SIZE_M the last tile of every batch
    # would run BLOCK_SIZE_M rows past its own slice.  The A/C descriptors are
    # flattened to (batch * M, .), so those rows belong to the *next* batch:
    # the load would mix in the next matrix and the store would overwrite its
    # first rows (and read/write past the buffer for the final batch).  Clamp
    # the tile start into the batch instead, so the edge tile overlaps the
    # previous one; the redundant rows recompute the exact same dot products,
    # which keeps the result correct for every M >= BLOCK_SIZE_M.
    offs_m = tl.minimum(pid_m * BLOCK_SIZE_M, tl.maximum(M - BLOCK_SIZE_M, 0))
    offs_am = (offs_m + batch_index * M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_SIZE_N).to(tl.int32)
    offs_ak = 0
    offs_ak = offs_ak.to(tl.int32)
    offs_bk = (batch_index * K).to(tl.int32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_ak])
        b = tl.load_tensor_descriptor(b_desc, [offs_bk, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_ak += BLOCK_SIZE_K
        offs_bk += BLOCK_SIZE_K
    if NEEDS_M_MASK:
        # M is smaller than a single tile, so no clamped window of BLOCK_SIZE_M
        # rows fits inside the batch and the tile above still reaches into the
        # next matrix.  Those extra rows only exist to fill the tile, so keep
        # the loads as they are and mask the store down to the live rows rather
        # than writing them past the end of the batch.
        offs_cm = (batch_index * M + offs_m + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
        offs_cn = offs_bn + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_m + tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (offs_cn < N)[
            None, :
        ]
        tl.store(
            c_ptr + offs_cm[:, None] * N + offs_cn[None, :],
            accumulator.to(c_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        tl.store_tensor_descriptor(
            c_desc, [offs_am, offs_bn], accumulator.to(c_desc.dtype)
        )


def bmm_sqmma(A, B, elem_type, batch, M, N, K, C=None):
    device = "musa"
    if C is None:
        C = torch.empty((batch, M, N), dtype=elem_type, device=device)
    c_flat = C.reshape(batch * M, N)
    desc_a = TensorDescriptor.from_tensor(A.reshape(batch * M, K), [1, 1])
    desc_b = TensorDescriptor.from_tensor(B.reshape(batch * K, N), [1, 1])
    desc_c = TensorDescriptor.from_tensor(c_flat, [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        batch,
        1,
    )
    bmm_sqmma_kernel[grid](
        desc_a,
        desc_b,
        desc_c,
        c_flat,
        batch,
        M,
        N,
        K,
        NEEDS_M_MASK=M < _SQMMA_MAX_BLOCK_M,
    )
    return C


@triton.jit
def _ws_bmm_consumer(
    a_reader,
    b_reader,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    batch_index,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    offset_m = pid_m * block_m + tl.arange(0, block_m)
    offset_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in tl.range(
        0,
        k_tiles,
        num_stages=1,
        loop_unroll_factor=k_tiles,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle.gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    c_block_ptr = (
        c_ptr + batch_index * (M * N) + N * offset_m[:, None] + offset_n[None, :]
    )
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ws_bmm_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    M: tl.constexpr,
    K: tl.constexpr,
    batch_index,
    pid_m,
    pid_n,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    # A/B descriptors are flattened over the batch axis, so the batch index is
    # just an extra row offset on top of the in-batch tile offset.
    offset_m = batch_index * M + pid_m * block_m
    offset_n = pid_n * block_n
    offset_bk = batch_index * K
    k_tiles: tl.constexpr = tl.cdiv(K, block_k)

    for k_iter in tl.range(
        0,
        k_tiles,
        num_stages=1,
        loop_unroll_factor=k_tiles,
    ):
        offset_k = k_iter * block_k
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc,
            a_slot.a,
            (block_m, block_k),
            (offset_m, offset_k),
        )
        tle.gpu.copy(
            b_desc,
            b_slot.b,
            (block_k, block_n),
            (offset_bk + offset_k, offset_n),
        )
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def bmm_sqmma_ws_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    panel_width: tl.constexpr,
    pipeline_stages: tl.constexpr,
    full_panel: tl.constexpr,
):
    raw_bx = tl.program_id(axis=0)
    raw_by = tl.program_id(axis=1)
    batch_index = tl.program_id(axis=2)
    grid_x = tl.num_programs(axis=0)
    grid_y = tl.num_programs(axis=1)
    block_idx = raw_bx + raw_by * grid_x
    pid_n, pid_m = _ws_rasterization_2d_column(
        block_idx,
        grid_x,
        grid_y,
        panel_width,
        full_panel,
    )

    a_smem = tle.gpu.alloc(
        (pipeline_stages, block_m, block_k),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (pipeline_stages, block_k, block_n),
        dtype=c_ptr.dtype.element_ty,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="bmm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=pipeline_stages, scope="cta", name="bmm_b", b=b_smem)

    tle.gpu.warp_specialize(
        [
            (
                _ws_bmm_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    M,
                    N,
                    K,
                    batch_index,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
            (
                _ws_bmm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    M,
                    K,
                    batch_index,
                    pid_m,
                    pid_n,
                    block_m,
                    block_n,
                    block_k,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


# Thresholds measured on an idle MTT S5000 (fp16/bf16), comparing this
# pipeline against the plain SQMMA kernel across batch in 1..16 and
# M/N in 256..4096.  The warp-specialized path costs more per launch than it
# saves whenever any of these three is too small:
#   * fewer than a 2x2 grid of 256x256 tiles, so a partial edge tile with only
#     a few live rows/columns dominates the block;
#   * fewer than eight 64-wide K tiles, so the 3-deep tile pipeline never
#     reaches steady state;
#   * fewer than 16 CTAs in the launch, so the grid cannot keep the machine
#     busy while the 16-warp consumer is spinning up.
# Outside those bounds bmm_sqmma stays faster, so it keeps those shapes.
_WS_MIN_OUTPUT_TILES = 16
_WS_MIN_K = 512


def is_bmm_ws_compatible(a, b, M, N, K):
    """Warp-specialized SQMMA for bmm needs TLE plus enough work per launch."""
    batch = a.shape[0]
    output_tiles = triton.cdiv(M, _WS_BLOCK_M) * triton.cdiv(N, _WS_BLOCK_N) * batch
    return (
        TLE_WS_ON
        and M >= 2 * _WS_BLOCK_M
        and N >= 2 * _WS_BLOCK_N
        and is_sqmma_compatible(a, b, N, K)
        and K >= _WS_MIN_K
        and output_tiles >= _WS_MIN_OUTPUT_TILES
    )


def bmm_sqmma_ws(A, B, batch, M, N, K, C=None):
    logger.debug("GEMS_MTHREADS BMM_SQMMA_WS")
    device = A.device
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    assert A.dtype == B.dtype, "Mat A and Mat B should have the same dtype"
    if C is None:
        C = torch.empty((batch, M, N), dtype=A.dtype, device=device)
    # The SQMMA consumer stores straight into a row-major buffer, so C must be
    # allocated at the *operand* dtype: bmm_sqmma used to round-trip bf16
    # through an fp16 scratch buffer, which silently downcast the output dtype.
    desc_a = TensorDescriptor.from_tensor(
        A.reshape(batch * M, K), [_WS_BLOCK_M, _WS_BLOCK_K]
    )
    desc_b = TensorDescriptor.from_tensor(
        B.reshape(batch * K, N), [_WS_BLOCK_K, _WS_BLOCK_N]
    )
    grid = (triton.cdiv(N, _WS_BLOCK_N), triton.cdiv(M, _WS_BLOCK_M), batch)
    bmm_sqmma_ws_kernel[grid](
        desc_a,
        desc_b,
        C,
        M,
        N,
        K,
        _WS_BLOCK_M,
        _WS_BLOCK_N,
        _WS_BLOCK_K,
        _WS_PANEL_WIDTH,
        _WS_STAGES,
        grid[0] % _WS_PANEL_WIDTH == 0,
        num_warps=_WS_NUM_WARPS,
        num_stages=_WS_STAGES,
    )
    return C


def bmm(a, b):
    a_dtype = a.dtype
    batch, M, K = a.shape
    _, _, N = b.shape
    if is_bmm_ws_compatible(a, b, M, N, K):
        return bmm_sqmma_ws(a, b, batch, M, N, K)
    elif is_sqmma_compatible(a, b, N, K) and M >= 128:
        return bmm_sqmma(a, b, a_dtype, batch, M, N, K)
    else:
        return bmm_fma(a, b)


def bmm_out(a, b, *, out):
    logger.debug("GEMS_MTHREADS BMM_OUT")
    batch, M, K = a.shape
    _, _, N = b.shape
    assert (
        out.shape == (batch, M, N) and out.dtype == a.dtype
    ), "Incompatible output shape or dtype for bmm.out"
    # All three specialized kernels store straight into a row-major buffer of
    # the operand dtype, so a contiguous `out` can be the destination itself
    # instead of paying for a temporary result plus a full-tensor copy.
    if out.is_contiguous():
        with torch_device_fn.device(a.device):
            if is_bmm_ws_compatible(a, b, M, N, K):
                return bmm_sqmma_ws(a, b, batch, M, N, K, C=out)
            if is_sqmma_compatible(a, b, N, K) and M >= 128:
                return bmm_sqmma(a, b, a.dtype, batch, M, N, K, C=out)
            return bmm_fma(a, b, C=out)
    out.copy_(bmm(a, b))
    return out
