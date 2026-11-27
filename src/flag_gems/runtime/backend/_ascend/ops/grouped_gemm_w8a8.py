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
import triton.language.extra.cann.extension as al

from flag_gems.runtime import torch_device_fn

from .grouped_matmul import _get_aic_core_num

logger = logging.getLogger(__name__)


@triton.jit
def grouped_gemm_plan(
    Counts,
    Plan,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    expert = tl.program_id(0)
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(Counts + experts, experts < E, 0)
    tiles = tl.cdiv(counts, BLOCK_M)
    first_row = tl.sum(tl.where(experts < expert, counts, 0), 0)
    first_tile = tl.sum(tl.where(experts < expert, tiles, 0), 0)
    count = tl.load(Counts + expert)
    if expert == 0:
        tl.store(Plan, tl.sum(tiles, 0))
    for tile in range(tl.cdiv(count, BLOCK_M)):
        entry = Plan + 1 + 3 * (first_tile + tile)
        tl.store(entry, first_row + tile * BLOCK_M)
        tl.store(entry + 1, first_row + count)
        tl.store(entry + 2, expert)


@triton.jit(do_not_specialize=["M"])
def grouped_gemm_int8_kernel(
    A,
    As,
    B,
    Bs,
    C,
    Indices,
    Plan,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    tiles_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    if ROW_MODE:
        total_tiles = M * tiles_n
    else:
        total_tiles = tl.load(Plan) * tiles_n
    for tile in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        if ROW_MODE:
            row_start = tile // tiles_n
            row_end = row_start + 1
            expert = tl.load(Indices + row_start)
            tile_n = tile % tiles_n
        else:
            entry = Plan + 1 + 3 * (tile // tiles_n)
            row_start = tl.load(entry)
            row_end = tl.load(entry + 1)
            expert = tl.load(entry + 2)
            tile_n = tile % tiles_n

        rows = (row_start + tl.arange(0, BLOCK_M)).to(tl.int64)
        columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        reduction = tl.arange(0, BLOCK_K)
        weight_base = expert.to(tl.int64) * N * K
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
        for chunk in range(tl.cdiv(K, BLOCK_K)):
            offsets_k = chunk * BLOCK_K + reduction
            if ALIGNED:
                # Discarded output rows do not require zero-filled dot inputs.
                activation = tl.load(
                    A + rows[:, None] * K + offsets_k[None, :],
                    rows[:, None] < row_end,
                    care_padding=False,
                )
                weight = tl.load(
                    B
                    + weight_base
                    + columns[:, None].to(tl.int64) * K
                    + offsets_k[None, :]
                )
                al.compile_hint(activation, "dot_pad_only_k")
                al.compile_hint(weight, "dot_pad_only_k")
            else:
                activation = tl.load(
                    A + rows[:, None] * K + offsets_k[None, :],
                    (rows[:, None] < row_end) & (offsets_k[None, :] < K),
                    0,
                )
                weight = tl.load(
                    B
                    + weight_base
                    + columns[:, None].to(tl.int64) * K
                    + offsets_k[None, :],
                    (columns[:, None] < N) & (offsets_k[None, :] < K),
                    0,
                )
            accumulator = tl.dot(
                activation, tl.trans(weight), accumulator, out_dtype=tl.int32
            )
        row_scale = tl.load(As + rows, rows < row_end, 0)
        channel_scale = tl.load(Bs + expert.to(tl.int64) * N + columns, columns < N, 0)
        scaled = accumulator.to(tl.float32) * row_scale[:, None]
        scaled = scaled * channel_scale[None, :]
        tl.store(
            C + rows[:, None] * N + columns[None, :],
            scaled.to(tl.bfloat16),
            (rows[:, None] < row_end) & (columns[None, :] < N),
        )


def grouped_gemm_w8a8(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    """Write grouped INT8 NT GEMM with FP32 row/channel scales into BF16 C.

    Expert rows must be contiguous in ascending expert order; m_rows must be
    nonnegative, sum to M, and agree with m_indices. These device values are
    caller preconditions and are not synchronized to the host.
    """
    logger.debug("GEMS_ASCEND GROUPED_GEMM_W8A8")
    if A.ndim != 2 or B.ndim != 3:
        raise ValueError("A must be [M,K] and B must be [E,N,K]")
    M, K = A.shape
    E, N, weight_k = B.shape
    tensors = (A, As, B, Bs, C, m_indices, m_rows)
    dtypes = (
        torch.int8,
        torch.float32,
        torch.int8,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.int32,
    )
    if any(t.dtype != dtype for t, dtype in zip(tensors, dtypes)):
        raise NotImplementedError(
            "requires INT8 A/B, FP32 scales, BF16 C, INT32 metadata"
        )
    if A.device.type != "npu" or any(t.device != A.device for t in tensors):
        raise ValueError("all tensors must be on the same NPU device")
    if not all(t.is_contiguous() for t in tensors):
        raise NotImplementedError("only contiguous tensors are supported")
    shapes = ((M, 1), (E, N, 1), (M, N), (M,), (E,))
    if weight_k != K or any(
        tuple(t.shape) != shape
        for t, shape in zip((As, Bs, C, m_indices, m_rows), shapes)
    ):
        raise ValueError("incompatible scales, output or expert metadata shapes")
    if K > 131071 or (M > 0 and E == 0):
        raise ValueError("requires K <= 131071 and experts for nonempty input")
    output_start = C.data_ptr()
    output_end = output_start + C.numel() * C.element_size()
    for tensor in (A, As, B, Bs, m_indices, m_rows):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        if output_start < end and start < output_end:
            raise ValueError("C must not overlap an input")
    if M == 0 or N == 0:
        return

    row_mode = M <= E
    if K >= 1024 and N % 256 == 0:
        block_m, block_n = (16, 256) if M <= 16 * E else (128, 128)
        block_k = 512
    else:
        block_m = 16 if M <= 16 * E else 64
        block_n = (512 if K <= 128 else 128) if M <= 16 * E else 256
        block_k = 128 if K <= 128 else (512 if E < M <= 16 * E else 256)
    tiles_n = triton.cdiv(N, block_n)
    aligned = N % block_n == 0 and K % block_k == 0
    pipeline_options = {"unit_flag": True, "sync_solver": False} if aligned else {}
    with torch_device_fn.device(A.device):
        if row_mode:
            plan = None
            capacity = M * tiles_n
        else:
            row_tiles = triton.cdiv(M, block_m) + E - 1
            plan = torch.empty((1 + 3 * row_tiles,), dtype=torch.int32, device=A.device)
            capacity = row_tiles * tiles_n
            grouped_gemm_plan[(E,)](m_rows, plan, E, triton.next_power_of_2(E), block_m)
        grouped_gemm_int8_kernel[(min(_get_aic_core_num(A.device.index), capacity),)](
            A,
            As,
            B,
            Bs,
            C,
            m_indices,
            plan,
            M,
            N,
            K,
            row_mode,
            block_m,
            block_n,
            block_k,
            aligned,
            multibuffer=True,
            **pipeline_options,
        )
