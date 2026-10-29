# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock

import torch
import triton
import triton.language as tl
from triton.compiler import CompiledKernel
from triton.runtime import JITFunction

import flag_gems
from flag_gems.fused.FLA.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from flag_gems.fused.FLA.chunk_o import chunk_fwd_kernel_o
from flag_gems.fused.FLA.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
from flag_gems.fused.FLA.cumsum import chunk_local_cumsum_scalar_kernel
from flag_gems.fused.FLA.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel,
)
from flag_gems.fused.FLA.solve_tril import merge_16x16_to_64x64_inverse_kernel
from flag_gems.fused.FLA.triton_ops_helper import exp
from flag_gems.fused.FLA.wy_fast import recompute_w_u_fwd_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._thead.ops.gdn_w8a16_int8 import store_int8_state
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Reuse the floating-point chunk cores without their torch metadata builders.
chunk_prepare = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel.jit_function
chunk_wy = recompute_w_u_fwd_kernel.jit_function
chunk_state = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.jit_function
chunk_output = chunk_fwd_kernel_o.jit_function
chunk_cumsum = chunk_local_cumsum_scalar_kernel.jit_function
chunk_kkt = chunk_scaled_dot_kkt_fwd_kernel.jit_function
chunk_solve = merge_16x16_to_64x64_inverse_kernel.jit_function
CHUNK_SIZE = 64
PARALLEL_SOLVE_MAX_TOKENS = 4096
SHORT_PREFILL_CACHE_SIZE = 32


# The following three kernels specialize the FLA chunk recurrence for this
# operator's V-major state and pre-inverted diagonal blocks. Shared FLA cores
# keep their existing layouts and interfaces.
# Derived from flash-linear-attention (MIT), copyright 2023-2025 Songlin Yang,
# Yu Zhang; see the corresponding flag_gems.fused.FLA cores.
@triton.jit(do_not_specialize=["T"])
def update_prefill_v_major(
    k,
    u,
    w,
    v_new,
    g,
    history,
    initial,
    final,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    value_block, group = tl.program_id(0), tl.program_id(1)
    sequence, head = group // H, group % H
    if IS_VARLEN:
        begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        T = tl.load(cu_seqlens + sequence + 1).to(tl.int32) - begin
        history_begin = tl.load(chunk_offsets + sequence).to(tl.int32)
    else:
        begin, history_begin = 0, 0
    k += (begin * Hg + head // (H // Hg)).to(tl.int64) * 128
    u += (begin * H + head).to(tl.int64) * 128
    w += (begin * H + head).to(tl.int64) * 128
    v_new += (begin * H + head).to(tl.int64) * 128
    g += begin * H + head
    history += (history_begin * H + head).to(tl.int64) * 128 * 128
    initial += group.to(tl.int64) * 128 * 128
    final += group.to(tl.int64) * 128 * 128
    state_ptr = tl.make_block_ptr(
        initial, (128, 128), (128, 1), (value_block * 64, 0), (64, 128), (1, 0)
    )
    state = tl.load(state_ptr, boundary_check=(0, 1)).to(tl.float32)
    for chunk in range(tl.cdiv(T, 64)):
        history_ptr = tl.make_block_ptr(
            history + chunk * H * 128 * 128,
            (128, 128),
            (128, 1),
            (value_block * 64, 0),
            (64, 128),
            (1, 0),
        )
        tl.store(
            history_ptr, state.to(history_ptr.dtype.element_ty), boundary_check=(0, 1)
        )
        w_ptr = tl.make_block_ptr(
            w, (T, 128), (H * 128, 1), (chunk * 64, 0), (64, 128), (1, 0)
        )
        weights = tl.load(w_ptr, boundary_check=(0, 1))
        prediction = tl.dot(weights, tl.trans(state).to(weights.dtype))
        u_ptr = tl.make_block_ptr(
            u, (T, 128), (H * 128, 1), (chunk * 64, value_block * 64), (64, 64), (1, 0)
        )
        values = tl.load(u_ptr, boundary_check=(0, 1)) - prediction
        new_ptr = tl.make_block_ptr(
            v_new,
            (T, 128),
            (H * 128, 1),
            (chunk * 64, value_block * 64),
            (64, 64),
            (1, 0),
        )
        tl.store(new_ptr, values.to(new_ptr.dtype.element_ty), boundary_check=(0, 1))
        last_gate = tl.load(g + (min((chunk + 1) * 64, T) - 1) * H).to(tl.float32)
        gate_ptr = tl.make_block_ptr(g, (T,), (H,), (chunk * 64,), (64,), (0,))
        gates = tl.load(gate_ptr, boundary_check=(0,)).to(tl.float32)
        valid = chunk * 64 + tl.arange(0, 64) < T
        values *= tl.where(valid, exp(last_gate - gates), 0)[:, None]
        state *= exp(last_gate)
        values_t = tl.trans(values.to(k.dtype.element_ty))
        # Retain the pipelined load/dot schedule of the measured full-K tile.
        for reduction_block in tl.range(1, num_stages=2):
            key_ptr = tl.make_block_ptr(
                k,
                (T, 128),
                (Hg * 128, 1),
                (chunk * 64, reduction_block * 128),
                (64, 128),
                (1, 0),
            )
            keys = tl.load(key_ptr, boundary_check=(0, 1))
            state += tl.dot(values_t, keys)
    final_ptr = tl.make_block_ptr(
        final, (128, 128), (128, 1), (value_block * 64, 0), (64, 128), (1, 0)
    )
    tl.store(final_ptr, state, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def output_prefill_v_major(
    q,
    k,
    v,
    history,
    g,
    output,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    global_chunk, head = tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        sequence = tl.load(chunk_indices + global_chunk * 2).to(tl.int32)
        chunk = tl.load(chunk_indices + global_chunk * 2 + 1).to(tl.int32)
        begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        T = tl.load(cu_seqlens + sequence + 1).to(tl.int32) - begin
    else:
        begin, chunk = 0, global_chunk
    q += (begin * Hg + head // (H // Hg)) * 128
    k += (begin * Hg + head // (H // Hg)) * 128
    v += (begin * H + head) * 128
    output += (begin * H + head) * 128
    history += (global_chunk * H + head).to(tl.int64) * 128 * 128
    result = tl.zeros((64, 128), tl.float32)
    attention = tl.zeros((64, 64), tl.float32)
    for reduction_block in range(2):
        q_ptr = tl.make_block_ptr(
            q,
            (T, 128),
            (Hg * 128, 1),
            (chunk * 64, reduction_block * 64),
            (64, 64),
            (1, 0),
        )
        k_ptr = tl.make_block_ptr(
            k,
            (128, T),
            (1, Hg * 128),
            (reduction_block * 64, chunk * 64),
            (64, 64),
            (0, 1),
        )
        h_ptr = tl.make_block_ptr(
            history, (128, 128), (128, 1), (0, reduction_block * 64), (128, 64), (1, 0)
        )
        query = tl.load(q_ptr, boundary_check=(0, 1))
        key = tl.load(k_ptr, boundary_check=(0, 1))
        state = tl.load(h_ptr, boundary_check=(0, 1))
        result += tl.dot(query, tl.trans(state))
        attention += tl.dot(query, key)
    gate_ptr = tl.make_block_ptr(
        g + begin * H + head, (T,), (H,), (chunk * 64,), (64,), (0,)
    )
    gates = tl.load(gate_ptr, boundary_check=(0,)).to(tl.float32)
    result *= exp(gates)[:, None]
    attention *= exp(gates[:, None] - gates[None, :])
    tokens = chunk * 64 + tl.arange(0, 64)
    valid = tokens < T
    causal = (tokens[:, None] >= tokens[None, :]) & valid[:, None] & valid[None, :]
    attention = tl.where(causal, attention, 0)
    v_ptr = tl.make_block_ptr(
        v, (T, 128), (H * 128, 1), (chunk * 64, 0), (64, 128), (1, 0)
    )
    o_ptr = tl.make_block_ptr(
        output, (T, 128), (H * 128, 1), (chunk * 64, 0), (64, 128), (1, 0)
    )
    values = tl.load(v_ptr, boundary_check=(0, 1))
    result = result * scale + tl.dot(attention.to(values.dtype), values) * scale
    tl.store(o_ptr, result.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def prefill_solve_block_ptr(
    matrix, T, chunk, H: tl.constexpr, ROW: tl.constexpr, COL: tl.constexpr
):
    return tl.make_block_ptr(
        matrix,
        (T, 64),
        (H * 64, 1),
        (chunk * 64 + ROW * 16, COL * 16),
        (16, 16),
        (1, 0),
    )


@triton.jit(do_not_specialize=["T"])
def merge_prefill_diagonal(
    matrix,
    inverse,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    chunk, head = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        sequence = tl.load(chunk_indices + chunk * 2).to(tl.int32)
        chunk = tl.load(chunk_indices + chunk * 2 + 1).to(tl.int32)
        begin = tl.load(cu_seqlens + sequence).to(tl.int32)
        T = tl.load(cu_seqlens + sequence + 1).to(tl.int32) - begin
    else:
        begin = 0
    matrix += (begin * H + head) * 64
    inverse += (begin * H + head) * 64
    inv11 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 0, 0), boundary_check=(0, 1)
    )
    inv22 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 1, 1), boundary_check=(0, 1)
    )
    inv33 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 2, 2), boundary_check=(0, 1)
    )
    inv44 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 3, 3), boundary_check=(0, 1)
    )
    a21 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 1, 0), boundary_check=(0, 1)
    )
    a31 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 2, 0), boundary_check=(0, 1)
    )
    a32 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 2, 1), boundary_check=(0, 1)
    )
    a41 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 3, 0), boundary_check=(0, 1)
    )
    a42 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 3, 1), boundary_check=(0, 1)
    )
    a43 = tl.load(
        prefill_solve_block_ptr(matrix, T, chunk, H, 3, 2), boundary_check=(0, 1)
    )
    inv21 = -tl.dot(
        tl.dot(inv22, a21, input_precision="ieee"), inv11, input_precision="ieee"
    )
    inv32 = -tl.dot(
        tl.dot(inv33, a32, input_precision="ieee"), inv22, input_precision="ieee"
    )
    inv43 = -tl.dot(
        tl.dot(inv44, a43, input_precision="ieee"), inv33, input_precision="ieee"
    )
    inv31 = -tl.dot(
        inv33,
        tl.dot(a31, inv11, input_precision="ieee")
        + tl.dot(a32, inv21, input_precision="ieee"),
        input_precision="ieee",
    )
    inv42 = -tl.dot(
        inv44,
        tl.dot(a42, inv22, input_precision="ieee")
        + tl.dot(a43, inv32, input_precision="ieee"),
        input_precision="ieee",
    )
    inv41 = -tl.dot(
        inv44,
        tl.dot(a41, inv11, input_precision="ieee")
        + tl.dot(a42, inv21, input_precision="ieee")
        + tl.dot(a43, inv31, input_precision="ieee"),
        input_precision="ieee",
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 0, 0),
        inv11.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 1, 1),
        inv22.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 2, 2),
        inv33.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 3, 3),
        inv44.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 1, 0),
        inv21.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 2, 0),
        inv31.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 2, 1),
        inv32.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 3, 0),
        inv41.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 3, 1),
        inv42.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        prefill_solve_block_ptr(inverse, T, chunk, H, 3, 2),
        inv43.to(inverse.dtype.element_ty),
        boundary_check=(0, 1),
    )


class ShortPrefillKernel:
    """Keep only compiled launch metadata; every call supplies fresh buffers."""

    def __init__(self, kernel: JITFunction):
        self.kernel = kernel
        self.compiled: CompiledKernel | None = None
        self.tail: tuple[int | bool | None, ...] = ()

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        def launch(
            *args: torch.Tensor | int | float | bool | str | None,
            **options: int | bool,
        ) -> None:
            if self.compiled is None:
                self.compiled = self.kernel[grid](*args, **options)
                self.tail = tuple(
                    options[param.name] if param.name in options else param.default
                    for param in self.kernel.params[len(args) :]
                )
            else:
                self.compiled[(grid + (1, 1))[:3]](*args, *self.tail)

        return launch


ShortPrefillKey = tuple[
    int, int, int, int, int, int, bool, float, tuple[tuple[torch.dtype, bool], ...]
]
short_prefill_cache: OrderedDict[ShortPrefillKey, tuple[ShortPrefillKernel, ...]] = (
    OrderedDict()
)
short_prefill_cache_lock = Lock()


@triton.jit
def prepare_short_prefill(
    g_in,
    g_out,
    k,
    beta,
    matrix,
    inverse,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    USE_TMA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    chunk_prepare(
        g_in,
        g_out,
        k,
        beta,
        matrix,
        inverse,
        cu_seqlens,
        chunk_indices,
        T,
        H,
        Hg,
        K,
        BT,
        BK,
        IS_VARLEN,
        USE_G,
        USE_TMA,
        DOT_PRECISION,
    )
    chunk, head = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        sequence = tl.load(chunk_indices + chunk * 2)
        chunk = tl.load(chunk_indices + chunk * 2 + 1)
        begin = tl.load(cu_seqlens + sequence)
        length = tl.load(cu_seqlens + sequence + 1) - begin
    else:
        begin, length = 0, T
    rows, columns = tl.arange(0, BT), tl.arange(0, BT)
    tokens = chunk * BT + rows
    offsets = ((begin + tokens[:, None]) * H + head) * BT + columns[None, :]
    # The shared solve writes diagonal and lower 16x16 tiles only.
    # Initialize its untouched tiles here to avoid a separate zeros launch.
    mask = (tokens[:, None] < length) & (rows[:, None] // 16 < columns[None, :] // 16)
    tl.store(inverse + offsets, 0, mask)


@libentry()
@triton.jit
def invert_prefill_diagonal(
    matrix,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    chunk, head, diagonal = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        sequence = tl.load(chunk_indices + chunk * 2)
        chunk = tl.load(chunk_indices + chunk * 2 + 1)
        begin = tl.load(cu_seqlens + sequence)
        length = tl.load(cu_seqlens + sequence + 1) - begin
    else:
        begin, length = 0, T
    rows, columns = tl.arange(0, 16), tl.arange(0, 16)
    tokens = chunk * BT + diagonal * 16 + rows
    offsets = (
        ((begin + tokens[:, None]) * H + head) * BT + diagonal * 16 + columns[None, :]
    )
    values = tl.load(matrix + offsets, tokens[:, None] < length, 0)
    power = -tl.where(rows[:, None] > columns[None, :], values, 0).to(tl.float32)
    inverse = power + (rows[:, None] == columns[None, :]).to(tl.float32)
    # A strictly triangular 16x16 matrix has no powers beyond the fifteenth.
    for _ in range(3):
        power = tl.dot(power, power, input_precision="ieee")
        inverse += tl.dot(inverse, power, input_precision="ieee")
    tl.store(matrix + offsets, inverse, tokens[:, None] < length)


@libentry()
@triton.jit
def normalize_prefill_qk(
    q,
    k,
    q_out,
    k_out,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    row_block=None,
):
    if row_block is None:
        row_block = tl.program_id(0)
    rows = row_block * BM + tl.arange(0, BM)
    columns = tl.arange(0, BK)
    offsets = rows[:, None] * K + columns[None, :]
    mask = (rows[:, None] < ROWS) & (columns[None, :] < K)
    query = tl.load(q + offsets, mask, 0).to(tl.float32)
    key = tl.load(k + offsets, mask, 0).to(tl.float32)
    query *= tl.rsqrt(tl.sum(query * query, 1) + 1e-6)[:, None]
    key *= tl.rsqrt(tl.sum(key * key, 1) + 1e-6)[:, None]
    tl.store(q_out + offsets, query, mask)
    tl.store(k_out + offsets, key, mask)


@libentry()
@triton.jit
def dequantize_prefill_state(
    state,
    scales,
    output,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_MAJOR: tl.constexpr = False,
    value_block=None,
    group=None,
):
    if group is None:
        group = tl.program_id(1)
    if value_block is None:
        value_block = tl.program_id(0)
    columns = value_block * BV + tl.arange(0, BV)
    reduction = tl.arange(0, K)
    source = (group * V + columns[:, None]) * K + reduction[None, :]
    target = (
        source
        if STATE_V_MAJOR
        else (group * K + reduction[None, :]) * V + columns[:, None]
    )
    values = tl.load(state + source, columns[:, None] < V, 0).to(tl.float32)
    scale = tl.load(scales + group * V + columns, columns < V, 0)
    tl.store(output + target, values * scale[:, None], columns[:, None] < V)


normalize_short_qk = normalize_prefill_qk.jit_function
dequantize_short_state = dequantize_prefill_state.jit_function


@triton.jit
def initialize_short_prefill(
    q,
    k,
    q_out,
    k_out,
    state,
    scales,
    initial,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    block = tl.program_id(0)
    norm_blocks: tl.constexpr = tl.cdiv(ROWS, 16) if NORMALIZE else 0
    if block < norm_blocks:
        normalize_short_qk(q, k, q_out, k_out, ROWS, K, K, 16, block)
    else:
        state_block = block - norm_blocks
        value_blocks: tl.constexpr = tl.cdiv(V, 32)
        dequantize_short_state(
            state,
            scales,
            initial,
            K,
            V,
            32,
            False,
            state_block % value_blocks,
            state_block // value_blocks,
        )


@libentry()
@triton.jit
def quantize_prefill_state(
    source,
    state,
    scales,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_MAJOR: tl.constexpr = False,
):
    group = tl.program_id(1).to(tl.int64)
    columns = tl.program_id(0) * BV + tl.arange(0, BV)
    reduction = tl.arange(0, BK)
    source_offsets = (
        ((group * V + columns[:, None]) * K + reduction[None, :])
        if STATE_V_MAJOR
        else ((group * K + reduction[None, :]) * V + columns[:, None])
    )
    target_offsets = (group * V + columns[:, None]) * K + reduction[None, :]
    mask = (columns[:, None] < V) & (reduction[None, :] < K)
    values = tl.load(source + source_offsets, mask, 0).to(tl.float32)
    store_int8_state(
        values, state, scales, target_offsets, group * V + columns, mask, columns < V
    )


def chunk_gated_delta_rule_w8a16_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_int8: torch.Tensor,
    state_scale: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked prefill with INT8 cache at sequence boundaries.

    Q/K/V are contiguous sequence-first tensors; state is [N, HV, V, K].
    Variable-length calls supply consistent precomputed 64-token chunk metadata.
    No recurrent quantization is applied between tokens inside a prefill call.
    """
    logger.debug("GEMS_THEAD CHUNK_GATED_DELTA_RULE_W8A16_INT8")
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4:
        raise ValueError("expected Q/K[B,T,H,K] and V[B,T,HV,V]")
    batch, tokens, query_heads, reduction = q.shape
    heads, columns = v.shape[-2:]
    if batch != 1 or reduction != 128 or columns != 128:
        raise NotImplementedError("prefill currently supports packed B=1 and K=V=128")
    if query_heads < 1 or heads % query_heads or v.shape[:2] != q.shape[:2]:
        raise ValueError("invalid QKV head or token dimensions")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share a BF16 or FP16 dtype")
    if g.shape != (batch, tokens, heads) or beta.shape != g.shape:
        raise ValueError("g and beta must have shape [1,T,HV]")
    if g.dtype != torch.float32 or beta.dtype != q.dtype:
        raise TypeError("log gates must be FP32 and beta must match QKV dtype")
    sequences = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    if sequences < 1 or state_int8.shape != (sequences, heads, columns, reduction):
        raise ValueError("state must have shape [N,HV,V,K]")
    if state_int8.dtype != torch.int8 or state_scale.dtype != torch.float32:
        raise TypeError("state and scales must be INT8 and FP32")
    if state_scale.shape != (sequences, heads, columns):
        raise ValueError("state scales must have shape [N,HV,V]")
    if cu_seqlens is None:
        if chunk_indices is not None or chunk_offsets is not None:
            raise ValueError("chunk metadata requires cu_seqlens")
        chunks = triton.cdiv(tokens, CHUNK_SIZE)
        metadata = ()
    else:
        if chunk_indices is None or chunk_offsets is None:
            raise ValueError("varlen prefill requires chunk_indices and chunk_offsets")
        if (
            cu_seqlens.ndim != 1
            or chunk_indices.ndim != 2
            or chunk_indices.shape[1] != 2
            or chunk_offsets.shape != (sequences + 1,)
        ):
            raise ValueError("invalid variable-length metadata shapes")
        metadata = (cu_seqlens, chunk_indices, chunk_offsets)
        if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in metadata):
            raise TypeError("chunk metadata must be integer tensors")
        chunks = chunk_indices.shape[0]
    tensors = (q, k, v, g, beta, state_int8, state_scale, *metadata)
    if q.device.type != "cuda" or any(
        tensor.device != q.device or not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("all inputs must be contiguous and on one accelerator")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("INT8-state GDN is forward-only")
    output = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    if tokens == 0:
        return output, state_int8
    launch_scale = reduction**-0.5 if scale is None else float(scale)
    initialize, prepare = initialize_short_prefill, prepare_short_prefill
    wy, update_state, compute_output = chunk_wy, chunk_state, chunk_output
    quantize = quantize_prefill_state
    cache_miss = False
    if tokens <= 128:
        # Pointer alignment is a Triton specialization even for contiguous views.
        # Cache dtype/alignment, never tensor addresses, values, outputs or streams.
        cache_key = (
            q.device.index,
            tokens,
            query_heads,
            heads,
            sequences,
            chunks,
            use_qk_l2norm_in_kernel,
            launch_scale,
            tuple((tensor.dtype, tensor.data_ptr() % 16 == 0) for tensor in tensors),
        )
        with short_prefill_cache_lock:
            plan = short_prefill_cache.get(cache_key)
            if plan is not None:
                short_prefill_cache.move_to_end(cache_key)
        cache_miss = plan is None
        if cache_miss:
            plan = tuple(
                ShortPrefillKernel(kernel)
                for kernel in (
                    initialize_short_prefill,
                    prepare_short_prefill,
                    chunk_wy,
                    chunk_state,
                    chunk_output,
                    quantize_prefill_state.jit_function,
                )
            )
        initialize, prepare, wy, update_state, compute_output, quantize = plan
    chunk_size = 32 if tokens <= 32 else CHUNK_SIZE
    use_v_major = tokens > 128
    state_shape = (columns, reduction) if use_v_major else (reduction, columns)
    initial = torch.empty(
        (sequences, heads, *state_shape), device=q.device, dtype=torch.float32
    )
    final = torch.empty_like(initial)
    q_norm, k_norm = q, k
    if use_qk_l2norm_in_kernel:
        q_norm, k_norm = torch.empty_like(q), torch.empty_like(k)
    gate_sum = torch.empty_like(g)
    matrix = torch.empty(
        (batch, tokens, heads, chunk_size), device=q.device, dtype=torch.float32
    )
    inverse = (
        torch.empty(matrix.shape, device=q.device, dtype=q.dtype)
        if tokens <= 128
        else flag_gems.zeros(matrix.shape, device=q.device, dtype=q.dtype)
    )
    w = torch.empty((batch, tokens, heads, reduction), device=q.device, dtype=q.dtype)
    u, v_new = torch.empty_like(v), torch.empty_like(v)
    history = torch.empty((chunks, heads, *state_shape), device=q.device, dtype=q.dtype)
    variable = cu_seqlens is not None
    # Offline paired trials favor full-K state tiles with V-major history on long inputs.
    state_value_block = 64 if use_v_major else 32
    parallel_solve = 128 < tokens <= PARALLEL_SOLVE_MAX_TOKENS
    with torch_device_fn.device(q.device):
        if tokens <= 128:
            norm_blocks = (
                triton.cdiv(tokens * query_heads, 16) if use_qk_l2norm_in_kernel else 0
            )
            initialize[(norm_blocks + triton.cdiv(columns, 32) * sequences * heads,)](
                q,
                k,
                q_norm,
                k_norm,
                state_int8,
                state_scale,
                initial,
                tokens * query_heads,
                reduction,
                columns,
                use_qk_l2norm_in_kernel,
                num_warps=4,
            )
        else:
            dequantize_prefill_state[(triton.cdiv(columns, 32), sequences * heads)](
                state_int8,
                state_scale,
                initial,
                reduction,
                columns,
                32,
                STATE_V_MAJOR=use_v_major,
                num_warps=4,
            )
        if tokens > 128 and use_qk_l2norm_in_kernel:
            normalize_prefill_qk[(triton.cdiv(tokens * query_heads, 16),)](
                q,
                k,
                q_norm,
                k_norm,
                tokens * query_heads,
                reduction,
                reduction,
                16,
                num_warps=4,
            )
        # Measured long inputs benefit from separate preparation; short inputs save launches when fused.
        if tokens <= 128:
            prepare[(chunks, heads)](
                g,
                gate_sum,
                k_norm,
                beta,
                matrix,
                inverse,
                cu_seqlens,
                chunk_indices,
                tokens,
                heads,
                query_heads,
                reduction,
                chunk_size,
                128,
                variable,
                True,
                False,
                "ieee",
                num_warps=2,
                num_stages=2,
            )
        else:
            chunk_cumsum[(chunks, heads)](
                g,
                gate_sum,
                cu_seqlens,
                chunk_indices,
                tokens,
                batch,
                heads,
                chunk_size,
                False,
                variable,
                False,
                num_warps=1,
            )
            chunk_kkt[(chunks, heads)](
                k_norm,
                beta,
                gate_sum,
                matrix,
                cu_seqlens,
                chunk_indices,
                tokens,
                heads,
                query_heads,
                reduction,
                chunk_size,
                64,
                variable,
                True,
                num_warps=2,
                num_stages=3,
            )
            if parallel_solve:
                invert_prefill_diagonal[(chunks, heads, 4)](
                    matrix,
                    cu_seqlens,
                    chunk_indices,
                    tokens,
                    heads,
                    chunk_size,
                    variable,
                    num_warps=1,
                    num_stages=1,
                )
                merge_prefill_diagonal[(chunks, heads)](
                    matrix,
                    inverse,
                    cu_seqlens,
                    chunk_indices,
                    tokens,
                    heads,
                    variable,
                    num_warps=2,
                    num_stages=1,
                )
            else:
                chunk_solve[(chunks, heads)](
                    matrix,
                    inverse,
                    cu_seqlens,
                    chunk_indices,
                    tokens,
                    heads,
                    chunk_size,
                    False,
                    variable,
                    "ieee",
                    num_warps=1,
                    num_stages=2,
                )
        wy[(chunks, heads)](
            k_norm,
            v,
            beta,
            w,
            u,
            inverse,
            gate_sum,
            cu_seqlens,
            chunk_indices,
            tokens,
            heads,
            query_heads,
            reduction,
            columns,
            chunk_size,
            32 if tokens > 128 else 64,
            64,
            variable,
            num_warps=2,
            num_stages=3 if tokens > 128 else 2,
        )
        if use_v_major:
            update_prefill_v_major[(2, sequences * heads)](
                k_norm,
                u,
                w,
                v_new,
                gate_sum,
                history,
                initial,
                final,
                cu_seqlens,
                chunk_offsets,
                tokens,
                heads,
                query_heads,
                variable,
                num_warps=4,
                num_stages=2,
            )
            output_prefill_v_major[(1, chunks, heads)](
                q_norm,
                k_norm,
                v_new,
                history,
                gate_sum,
                output,
                cu_seqlens,
                chunk_indices,
                launch_scale,
                tokens,
                heads,
                query_heads,
                variable,
                num_warps=8,
                num_stages=3,
            )
        else:
            update_state[(triton.cdiv(columns, state_value_block), sequences * heads)](
                k_norm,
                u,
                w,
                v_new,
                gate_sum,
                None,
                history,
                initial,
                final,
                cu_seqlens,
                chunk_offsets,
                tokens,
                heads,
                query_heads,
                reduction,
                columns,
                chunk_size,
                state_value_block,
                True,
                False,
                True,
                True,
                True,
                variable,
                num_warps=4,
                num_stages=2,
            )
            compute_output[(triton.cdiv(columns, 128), chunks, heads)](
                q_norm,
                k_norm,
                v_new,
                history,
                gate_sum,
                output,
                cu_seqlens,
                chunk_indices,
                launch_scale,
                tokens,
                heads,
                query_heads,
                reduction,
                columns,
                chunk_size,
                64 if use_v_major else 128,
                128,
                True,
                variable,
                num_warps=8 if use_v_major else 4,
                num_stages=3 if use_v_major else 2,
            )
        quantize[(triton.cdiv(columns, 16), sequences * heads)](
            final,
            state_int8,
            state_scale,
            reduction,
            columns,
            reduction,
            16,
            STATE_V_MAJOR=use_v_major,
            num_warps=4,
        )
    if cache_miss:
        # Publish only complete plans. Concurrent cold calls compile independently;
        # cached kernels are read-only, and no GPU work runs under this lock.
        with short_prefill_cache_lock:
            short_prefill_cache.setdefault(cache_key, plan)
            if len(short_prefill_cache) > SHORT_PREFILL_CACHE_SIZE:
                short_prefill_cache.popitem(last=False)
    return output, state_int8
