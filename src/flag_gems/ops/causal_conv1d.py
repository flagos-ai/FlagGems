import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_device_capability
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

PAD_SLOT_ID = -1

# TLE `extract_tile` lowers to efficient sub-tile slicing only on Hopper+.
_TLE_MIN_CAPABILITY = 9

# Kernel widths for which the TLE forward kernel has been validated.
_TLE_FWD_SUPPORTED_WIDTHS = (2, 3, 4)
# The TLE update kernel additionally covers width 5 and 6.
_TLE_UPDATE_SUPPORTED_WIDTHS = (2, 3, 4, 5, 6)

_DEFAULT_BLOCK_M = 8
_DEFAULT_BLOCK_N = 256
_MAX_NUM_PROGRAMS = 1024


# ---------------------------------------------------------------------------
# Baseline kernels (no TLE)
# ---------------------------------------------------------------------------


@triton.jit()
def _causal_conv1d_fwd_kernel(
    x_ptr,  # (dim, cu_seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,  # (dim,)
    initial_states_ptr,  # conv_states, (num_cache_lines, dim, state_len)
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,
    block_idx_last_scheduled_token,
    initial_state_idx,
    num_computed_tokens,
    o_ptr,  # (dim, cu_seqlen)
    dim: tl.constexpr,
    seqlen: tl.int32,
    num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_block_m: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        n_block_to_fill = current_last_index - current_first_index
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim

    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
    ).to(tl.int64)

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return

    conv_states_base = (
        conv_states_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                col0 = tl.load(prior_tokens, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                col1 = tl.load(prior_tokens, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # Write back the tail of this sequence as the new conv_state.
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_last_index
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)
        else:
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            VAL = state_len - seqlen
            if load_init_state:
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_states_input_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                # NOTE: the barrier is required -- tl.where miscompiles when both
                # operands come straight out of a masked tl.load.
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
            else:
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

            conv_states_ptrs_target = (
                conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            col0 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            col1 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(
                prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
            )
        if KERNEL_WIDTH == 4:
            col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col1 = tl.load(
                prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
            )
            col0 = tl.load(
                prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
            )

        if (chunk_offset - 1) < n_block_to_fill:
            idx_tokens_last = (
                last_full_block_token_index
                - (n_block_to_fill - chunk_offset) * B_size
                - state_len
            ) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + (idx_tokens_last * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_first_index
                + (chunk_offset - 1)
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        acc_preload = tl.load(bias_ptr + idx_feats, mask=idx_feats < dim, other=0.0).to(
            tl.float32
        )
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)

    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d)


@triton.jit()
def _causal_conv1d_update_kernel(
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,
    block_idx_last_scheduled_token,
    initial_state_idx,
    o_ptr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    if IS_SPEC_DECODING:
        conv_state_token_offset = (
            tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
        )
    else:
        conv_state_token_offset = 0

    conv_states_base = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        col0 = tl.load(prior_tokens, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        col3 = tl.load(prior_tokens + 3 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        col4 = tl.load(prior_tokens + 4 * stride_conv_state_tok, mask_w, 0.0)

    idx_tokens = tl.arange(0, NP2_STATELEN)

    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )
    mask = (
        (conv_states_input_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)

    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

    # NOTE: required -- see the barrier comment in the forward kernel.
    tl.debug_barrier()
    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
    mask_out = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask_out)

    if HAS_BIAS:
        acc_preload = tl.load(bias_ptr + idx_feats, mask=idx_feats < dim, other=0.0).to(
            tl.float32
        )
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + (4 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + (5 * stride_w_width), mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    matrix_x = tl.load(
                        x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                    )

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d)


# ---------------------------------------------------------------------------
# TLE kernels (Triton >= 3.6, Hopper+)
#
# Same math as above; the sliding-window columns are sliced out of a single
# tile load with `tle.extract_tile` instead of issuing KERNEL_WIDTH-1
# separate strided loads per token.
# ---------------------------------------------------------------------------

if HAS_TLE:

    @triton.jit()
    def _causal_conv1d_fwd_kernel_tle(
        x_ptr,
        w_ptr,
        bias_ptr,
        initial_states_ptr,
        cache_indices_ptr,
        has_initial_states_ptr,
        query_start_loc_ptr,
        batch_ptr,
        token_chunk_offset_ptr,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        o_ptr,
        dim: tl.constexpr,
        seqlen: tl.int32,
        num_cache_lines: tl.constexpr,
        stride_x_dim: tl.constexpr,
        stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr,
        stride_w_width: tl.constexpr,
        stride_istate_seq: tl.constexpr,
        stride_istate_dim: tl.constexpr,
        stride_istate_token: tl.constexpr,
        stride_cache_indices: tl.constexpr,
        stride_o_dim: tl.constexpr,
        stride_o_token: tl.constexpr,
        stride_block_m: tl.constexpr,
        pad_slot_id: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        KERNEL_WIDTH: tl.constexpr,
        SILU_ACTIVATION: tl.constexpr,
        IS_APC_ENABLED: tl.constexpr,
        USE_PAD_SLOT: tl.constexpr,
        NP2_STATELEN: tl.constexpr,
        X_TILE_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        conv_states_ptr = initial_states_ptr
        conv_state_indices_ptr = cache_indices_ptr
        stride_conv_state_seq = stride_istate_seq
        stride_conv_state_dim = stride_istate_dim
        stride_conv_state_tok = stride_istate_token
        state_len = KERNEL_WIDTH - 1

        idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
        chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

        idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

        if idx_seq == pad_slot_id:
            return

        sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
        sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
        seqlen = sequence_end_index - sequence_start_index

        B_size: tl.constexpr = stride_block_m * BLOCK_M

        if IS_APC_ENABLED:
            current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
            current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
            sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

            sequence_completed_offset_token = sequence_completed_index % B_size
            seq_completed_offset = B_size - sequence_completed_offset_token
            seq_end_offset = (seqlen - seq_completed_offset) % B_size
            last_full_block_token_index = sequence_end_index - seq_end_offset
            if seq_end_offset == 0:
                last_full_block_token_index = last_full_block_token_index - B_size

            n_block_to_fill = current_last_index - current_first_index
            conv_state_init_index = tl.load(initial_state_idx + idx_seq)
        else:
            n_block_to_fill = 0
            current_last_index = 0
            conv_state_init_index = 0
            current_first_index = 0
            last_full_block_token_index = 0

        token_offset = BLOCK_M * chunk_offset
        segment_len = min(BLOCK_M, seqlen - token_offset)

        x_base = (
            x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
        )

        conv_states_input_coord = tl.load(
            conv_state_indices_ptr
            + idx_seq * stride_cache_indices
            + conv_state_init_index
        ).to(tl.int64)

        if USE_PAD_SLOT:
            if conv_states_input_coord == pad_slot_id:
                return

        conv_states_base = (
            conv_states_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )
        w_base = w_ptr + (idx_feats * stride_w_dim)

        idx_rows = tl.arange(0, X_TILE_M)

        if chunk_offset == 0:
            # First chunk: history comes from conv_state, so the x tile starts at
            # token_offset with no left halo.
            chunk = 0
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)

            offs_tok = token_offset + idx_rows
            tile_ptrs = (
                x_ptr
                + (sequence_start_index + offs_tok)[:, None] * stride_x_token
                + (idx_feats * stride_x_dim)[None, :]
            )
            tile_mask = (
                (offs_tok >= 0)[:, None]
                & (offs_tok < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
                & (idx_rows < (BLOCK_M + KERNEL_WIDTH - 1))[:, None]
            )
            tile_data = tl.load(
                tile_ptrs, mask=tile_mask, other=0.0, cache_modifier=".ca"
            )

            if load_init_state:
                # One tile load for the whole conv_state window, then slice.
                idx_state = tl.arange(0, NP2_STATELEN)
                conv_state_ptrs_full = (
                    conv_states_base[None, :]
                    + idx_state[:, None] * stride_conv_state_tok
                )
                mask_state_full = (idx_state[:, None] < state_len) & (
                    idx_feats[None, :] < dim
                )
                state_tile = tl.load(conv_state_ptrs_full, mask_state_full, other=0.0)

                if KERNEL_WIDTH == 2:
                    col0 = tle.extract_tile(
                        state_tile, index=[state_len - 1, 0], tile_shape=[1, BLOCK_N]
                    ).reshape((BLOCK_N,))
                if KERNEL_WIDTH == 3:
                    col1 = tle.extract_tile(
                        state_tile, index=[state_len - 1, 0], tile_shape=[1, BLOCK_N]
                    ).reshape((BLOCK_N,))
                    col0 = tle.extract_tile(
                        state_tile, index=[state_len - 2, 0], tile_shape=[1, BLOCK_N]
                    ).reshape((BLOCK_N,))
                if KERNEL_WIDTH == 4:
                    col2 = tle.extract_tile(
                        state_tile, index=[state_len - 1, 0], tile_shape=[1, BLOCK_N]
                    ).reshape((BLOCK_N,))
                    col1 = tle.extract_tile(
                        state_tile, index=[state_len - 2, 0], tile_shape=[1, BLOCK_N]
                    ).reshape((BLOCK_N,))
                    col0 = tle.extract_tile(
                        state_tile, index=[state_len - 3, 0], tile_shape=[1, BLOCK_N]
                    ).reshape((BLOCK_N,))
            else:
                if KERNEL_WIDTH >= 2:
                    col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 3:
                    col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                if KERNEL_WIDTH >= 4:
                    col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

            if state_len <= seqlen:
                idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
                x_ptrs = (
                    x_ptr
                    + ((sequence_start_index + idx_tokens_last) * stride_x_token)[
                        :, None
                    ]
                    + (idx_feats * stride_x_dim)[None, :]
                )
                mask_x = (
                    (idx_tokens_last >= 0)[:, None]
                    & (idx_tokens_last < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)

                conv_states_output_coord = tl.load(
                    conv_state_indices_ptr
                    + idx_seq * stride_cache_indices
                    + current_last_index
                ).to(tl.int64)

                conv_states_ptrs_target = (
                    conv_states_ptr
                    + (conv_states_output_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)
                )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.debug_barrier()
                tl.store(conv_states_ptrs_target, loaded_x, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                if load_init_state:
                    conv_states_ptrs_source = (
                        conv_states_ptr
                        + (conv_states_input_coord * stride_conv_state_seq)
                        + (idx_feats * stride_conv_state_dim)[None, :]
                        + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                    )
                    mask = (
                        (conv_states_input_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                    x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                    )
                    mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                    # NOTE: required -- tl.where miscompiles when both operands
                    # come straight out of a masked tl.load.
                    tl.debug_barrier()
                    new_conv_state = tl.where(mask, conv_state, loaded_x)
                else:
                    x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                    )
                    mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                    )
                    new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            # Later chunks: history is the KERNEL_WIDTH-1 tokens to the left of
            # this chunk, so shift the tile start left by the halo.
            chunk = 1
            x_tile_start = token_offset - (KERNEL_WIDTH - 1)
            offs_tok = x_tile_start + idx_rows
            tile_ptrs = (
                x_ptr
                + (sequence_start_index + offs_tok)[:, None] * stride_x_token
                + (idx_feats * stride_x_dim)[None, :]
            )
            tile_mask = (
                (offs_tok >= 0)[:, None]
                & (offs_tok < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
                & (idx_rows < (BLOCK_M + KERNEL_WIDTH - 1))[:, None]
            )
            tile_data = tl.load(
                tile_ptrs, mask=tile_mask, other=0.0, cache_modifier=".ca"
            )

            if KERNEL_WIDTH >= 2:
                col0 = tl.reshape(
                    tle.extract_tile(tile_data, index=[0, 0], tile_shape=[1, BLOCK_N]),
                    [BLOCK_N],
                )
            if KERNEL_WIDTH >= 3:
                col1 = tl.reshape(
                    tle.extract_tile(tile_data, index=[1, 0], tile_shape=[1, BLOCK_N]),
                    [BLOCK_N],
                )
            if KERNEL_WIDTH >= 4:
                col2 = tl.reshape(
                    tle.extract_tile(tile_data, index=[2, 0], tile_shape=[1, BLOCK_N]),
                    [BLOCK_N],
                )

            if (chunk_offset - 1) < n_block_to_fill:
                idx_tokens_last = (
                    last_full_block_token_index
                    - (n_block_to_fill - chunk_offset) * B_size
                    - state_len
                ) + tl.arange(0, NP2_STATELEN)
                x_ptrs = (
                    x_ptr
                    + (idx_tokens_last * stride_x_token)[:, None]
                    + (idx_feats * stride_x_dim)[None, :]
                )
                mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)

                conv_states_output_coord = tl.load(
                    conv_state_indices_ptr
                    + idx_seq * stride_cache_indices
                    + current_first_index
                    + (chunk_offset - 1)
                ).to(tl.int64)

                conv_states_ptrs_target = (
                    conv_states_ptr
                    + (conv_states_output_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)
                )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.debug_barrier()
                tl.store(conv_states_ptrs_target, loaded_x, mask)

        if HAS_BIAS:
            acc_preload = tl.load(
                bias_ptr + idx_feats, mask=idx_feats < dim, other=0.0
            ).to(tl.float32)
        else:
            acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

        mask_w = idx_feats < dim
        if KERNEL_WIDTH >= 2:
            w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
            w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
        if KERNEL_WIDTH >= 3:
            w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
        if KERNEL_WIDTH >= 4:
            w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)

        for idx_token in range(segment_len):
            acc = acc_preload

            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = tl.reshape(
                            tle.extract_tile(
                                tile_data,
                                index=[idx_token + chunk * (KERNEL_WIDTH - 1), 0],
                                tile_shape=[1, BLOCK_N],
                            ),
                            [BLOCK_N],
                        )
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = tl.reshape(
                            tle.extract_tile(
                                tile_data,
                                index=[idx_token + chunk * (KERNEL_WIDTH - 1), 0],
                                tile_shape=[1, BLOCK_N],
                            ),
                            [BLOCK_N],
                        )
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = tl.reshape(
                            tle.extract_tile(
                                tile_data,
                                index=[idx_token + chunk * (KERNEL_WIDTH - 1), 0],
                                tile_shape=[1, BLOCK_N],
                            ),
                            [BLOCK_N],
                        )

                acc += matrix_x * matrix_w

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            mask_1d = (idx_token < segment_len) & (idx_feats < dim)
            o_ptrs = (
                o_ptr
                + (sequence_start_index + token_offset + idx_token) * stride_o_token
                + (idx_feats * stride_o_dim)
            )
            tl.store(o_ptrs, acc, mask=mask_1d)

    @triton.jit()
    def _causal_conv1d_update_kernel_tle(
        x_ptr,
        w_ptr,
        bias_ptr,
        conv_state_ptr,
        conv_state_indices_ptr,
        num_accepted_tokens_ptr,
        query_start_loc_ptr,
        block_idx_last_scheduled_token,
        initial_state_idx,
        o_ptr,
        batch: int,
        dim: tl.constexpr,
        seqlen: tl.constexpr,
        state_len: tl.constexpr,
        num_cache_lines: tl.constexpr,
        stride_x_seq: tl.constexpr,
        stride_x_dim: tl.constexpr,
        stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr,
        stride_w_width: tl.constexpr,
        stride_conv_state_seq: tl.constexpr,
        stride_conv_state_dim: tl.constexpr,
        stride_conv_state_tok: tl.constexpr,
        stride_state_indices: tl.constexpr,
        stride_o_seq: tl.constexpr,
        stride_o_dim: tl.constexpr,
        stride_o_token: tl.constexpr,
        pad_slot_id: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        KERNEL_WIDTH: tl.constexpr,
        SILU_ACTIVATION: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        IS_APC_ENABLED: tl.constexpr,
        IS_SPEC_DECODING: tl.constexpr,
        NP2_STATELEN: tl.constexpr,
        USE_PAD_SLOT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        idx_seq = tl.program_id(0)
        if idx_seq >= batch:
            return

        idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

        if IS_APC_ENABLED:
            conv_state_init = tl.load(initial_state_idx + idx_seq)
            current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        else:
            conv_state_init = 0
            current_last_index = 0

        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
        ).to(tl.int64)

        if USE_PAD_SLOT:
            if conv_states_input_coord == pad_slot_id:
                return

        if IS_VARLEN:
            query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
            query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
            state_len = state_len - (seqlen - (query_end_index - query_start_index))
            seqlen = query_end_index - query_start_index
            x_offset = query_start_index * stride_x_token
            o_offset = query_start_index * stride_o_token
        else:
            query_start_index = idx_seq * seqlen
            query_end_index = query_start_index + seqlen
            x_offset = idx_seq * stride_x_seq
            o_offset = idx_seq * stride_o_seq

        if query_start_index == query_end_index:
            return

        if IS_SPEC_DECODING:
            conv_state_token_offset = (
                tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
            )
        else:
            conv_state_token_offset = 0

        # One tile load for the conv_state window, sliced with extract_tile.
        conv_states_base = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )

        idx_state = tl.arange(0, NP2_STATELEN)
        conv_state_ptrs_full = (
            conv_states_base[None, :]
            + (conv_state_token_offset + idx_state)[:, None] * stride_conv_state_tok
        )
        mask_state_full = (idx_state[:, None] < state_len) & (idx_feats[None, :] < dim)
        state_tile = tl.load(conv_state_ptrs_full, mask_state_full, other=0.0)

        if KERNEL_WIDTH >= 2:
            col0 = tle.extract_tile(
                state_tile, index=[0, 0], tile_shape=[1, BLOCK_N]
            ).reshape((BLOCK_N,))
        if KERNEL_WIDTH >= 3:
            col1 = tle.extract_tile(
                state_tile, index=[1, 0], tile_shape=[1, BLOCK_N]
            ).reshape((BLOCK_N,))
        if KERNEL_WIDTH >= 4:
            col2 = tle.extract_tile(
                state_tile, index=[2, 0], tile_shape=[1, BLOCK_N]
            ).reshape((BLOCK_N,))
        if KERNEL_WIDTH >= 5:
            col3 = tle.extract_tile(
                state_tile, index=[3, 0], tile_shape=[1, BLOCK_N]
            ).reshape((BLOCK_N,))
        if KERNEL_WIDTH >= 6:
            col4 = tle.extract_tile(
                state_tile, index=[4, 0], tile_shape=[1, BLOCK_N]
            ).reshape((BLOCK_N,))

        idx_tokens = tl.arange(0, NP2_STATELEN)

        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + conv_state_token_offset * stride_conv_state_tok
            + (idx_feats * stride_conv_state_dim)[None, :]
            + (
                (idx_tokens + (1 if IS_SPEC_DECODING else seqlen))
                * stride_conv_state_tok
            )[:, None]
        )
        mask = (
            (conv_states_input_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

        VAL = state_len - seqlen
        x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)
        x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
        mask_x = (
            (idx_tokens - VAL >= 0)[:, None]
            & (idx_tokens - VAL < seqlen)[:, None]
            & (idx_feats < dim)[None, :]
        )
        loaded_x = tl.load(x_ptrs, mask_x, 0.0)

        # NOTE: required -- see the barrier comment in the forward kernel.
        tl.debug_barrier()
        new_conv_state = tl.where(mask, conv_state, loaded_x)

        conv_states_offset = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
        ).to(tl.int64)
        conv_state_ptrs_target = (
            conv_state_ptr
            + (conv_states_offset * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
        )[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
        mask_out = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
        tl.store(conv_state_ptrs_target, new_conv_state, mask_out)

        if HAS_BIAS:
            acc_preload = tl.load(
                bias_ptr + idx_feats, mask=idx_feats < dim, other=0.0
            ).to(tl.float32)
        else:
            acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

        w_base = w_ptr + (idx_feats * stride_w_dim)
        mask_w = idx_feats < dim
        if KERNEL_WIDTH >= 2:
            w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
            w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
        if KERNEL_WIDTH >= 3:
            w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
        if KERNEL_WIDTH >= 4:
            w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
        if KERNEL_WIDTH >= 5:
            w_col4 = tl.load(w_base + (4 * stride_w_width), mask_w, other=0.0)
        if KERNEL_WIDTH >= 6:
            w_col5 = tl.load(w_base + (5 * stride_w_width), mask_w, other=0.0)

        x_base_1d = x_base
        mask_x_1d = idx_feats < dim

        for idx_token in tl.range(seqlen):
            acc = acc_preload

            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = tl.load(
                            x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                        )
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = tl.load(
                            x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                        )
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = tl.load(
                            x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                        )
                elif KERNEL_WIDTH == 5:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = tl.load(
                            x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                        )
                elif KERNEL_WIDTH == 6:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = col4
                    elif j == 5:
                        matrix_w = w_col5
                        matrix_x = tl.load(
                            x_base_1d + idx_token * stride_x_token, mask=mask_x_1d
                        )

                acc += matrix_x * matrix_w

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x
            elif KERNEL_WIDTH == 5:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = matrix_x
            elif KERNEL_WIDTH == 6:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = col4
                col4 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            mask_1d = (idx_token < seqlen) & (idx_feats < dim)
            o_ptrs = (
                o_ptr
                + o_offset
                + idx_token * stride_o_token
                + (idx_feats * stride_o_dim)
            )
            tl.store(o_ptrs, acc, mask=mask_1d)

else:
    _causal_conv1d_fwd_kernel_tle = None
    _causal_conv1d_update_kernel_tle = None


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def _tle_available(x: torch.Tensor) -> bool:
    if not HAS_TLE:
        return False
    if x.device.type != "cuda":
        return False
    capability = get_device_capability()
    return capability[0] >= _TLE_MIN_CAPABILITY


def _normalize_activation(activation):
    if isinstance(activation, bool):
        return "silu" if activation else None
    if activation is not None and activation not in ("silu", "swish"):
        raise ValueError(
            f"Unsupported activation: {activation!r}, expected None, 'silu' or 'swish'."
        )
    return activation


def _make_chunk_schedule(seqlens_cpu: torch.Tensor, block_m: int, device):
    """Flatten (sequence, chunk) pairs into a 1-D program schedule.

    Returns (batch_ptr, token_chunk_offset_ptr, num_programs). Each program
    handles up to ``block_m`` tokens of one sequence.
    """
    nums = -(-seqlens_cpu.numpy() // block_m)  # ceil-div per sequence
    total = int(nums.sum())

    seq_ids = np.repeat(np.arange(len(nums)), nums)
    chunk_ids = np.concatenate(
        [np.arange(n) for n in nums] if total > 0 else [np.empty(0, dtype=np.int64)]
    )

    size = max(total, _MAX_NUM_PROGRAMS)
    batch_ptr = torch.full((size,), PAD_SLOT_ID, dtype=torch.int32, device=device)
    token_chunk_offset_ptr = torch.full(
        (size,), PAD_SLOT_ID, dtype=torch.int32, device=device
    )
    if total > 0:
        batch_ptr[:total] = torch.from_numpy(seq_ids.astype(np.int32)).to(device)
        token_chunk_offset_ptr[:total] = torch.from_numpy(
            chunk_ids.astype(np.int32)
        ).to(device)
    return batch_ptr, token_chunk_offset_ptr, total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    conv_states=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,
):
    """Depthwise causal conv1d over a varlen (packed) batch, with state update.

    This is the Mamba/SSM-style stateful causal convolution, not a general
    ``torch.nn.functional.conv1d``. It is depthwise (one filter per channel),
    left-padded so that output token ``t`` depends only on tokens ``<= t``, and
    it reads/writes a persistent ``conv_states`` cache.

    Args:
        x: (dim, cu_seqlen) packed activations.
        weight: (dim, width) depthwise filters.
        bias: (dim,) or None.
        conv_states: (num_cache_lines, dim, width - 1). Updated in place with the
            trailing ``width - 1`` tokens of each sequence.
        query_start_loc: (batch + 1,) int32 cumulative sequence offsets.
        cache_indices: (batch,) int32 row of ``conv_states`` used by each sequence.
        has_initial_state: (batch,) bool; whether to seed from ``conv_states``.
        activation: "silu"/"swish", True (== "silu"), or None.
        pad_slot_id: sentinel marking padded cache slots; those are skipped.

    Returns:
        (dim, cu_seqlen) output with the same dtype as ``x``.
    """
    logger.debug("GEMS CAUSAL_CONV1D")
    assert conv_states is not None, "conv_states is required"
    assert query_start_loc is not None, "query_start_loc is required"
    assert x.ndim == 2, f"x must be (dim, cu_seqlen), got shape {tuple(x.shape)}"
    assert weight.ndim == 2, f"weight must be (dim, width), got {tuple(weight.shape)}"

    activation = _normalize_activation(activation)

    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    out = torch.empty_like(x)

    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    num_cache_lines = conv_states.size(0)

    block_m = _DEFAULT_BLOCK_M
    block_n = _DEFAULT_BLOCK_N

    use_tle = _tle_available(x) and width in _TLE_FWD_SUPPORTED_WIDTHS
    kernel = _causal_conv1d_fwd_kernel_tle if use_tle else _causal_conv1d_fwd_kernel

    # Chunk schedule is host-side and involves a D2H sync; compute it once here
    # rather than inside the grid lambda.
    seqlens_cpu = query_start_loc.diff().to("cpu")
    batch_ptr, token_chunk_offset_ptr, num_programs = _make_chunk_schedule(
        seqlens_cpu, block_m, x.device
    )

    grid = (num_programs, triton.cdiv(dim, block_n))

    stride_x_dim, stride_x_token = x.stride()
    stride_w_dim, stride_w_width = weight.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_states.stride()
    stride_o_dim, stride_o_token = out.stride()
    stride_cache_indices = cache_indices.stride(0) if cache_indices is not None else 0

    extra = {}
    if use_tle:
        extra["X_TILE_M"] = triton.next_power_of_2(block_m + width - 1)

    with torch_device_fn.device(x.device):
        kernel[grid](
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            has_initial_state,
            query_start_loc,
            batch_ptr,
            token_chunk_offset_ptr,
            None,  # block_idx_first_scheduled_token
            None,  # block_idx_last_scheduled_token
            None,  # initial_state_idx
            None,  # num_computed_tokens
            out,
            dim,
            cu_seqlen,
            num_cache_lines,
            stride_x_dim,
            stride_x_token,
            stride_w_dim,
            stride_w_width,
            stride_istate_seq,
            stride_istate_dim,
            stride_istate_token,
            stride_cache_indices,
            stride_o_dim,
            stride_o_token,
            1,  # stride_block_m
            pad_slot_id,
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ("silu", "swish"),
            IS_APC_ENABLED=False,
            USE_PAD_SLOT=pad_slot_id is not None,
            NP2_STATELEN=np2_statelen,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_stages=2,
            **extra,
        )

    return out.to(original_x_dtype)


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    pad_slot_id=PAD_SLOT_ID,
):
    """Single-step (decode) causal conv1d with in-place conv_state update.

    Args:
        x: (batch, dim, seqlen) or (batch, dim) for seqlen == 1.
        conv_state: (num_cache_lines, dim, width - 1), updated in place.
        weight: (dim, width).
        bias: (dim,) or None.
        activation: "silu"/"swish", True (== "silu"), or None.
        conv_state_indices: (batch,) int32 row of ``conv_state`` per sequence.
        pad_slot_id: sentinel marking padded cache slots; those are skipped.

    Returns:
        Output with the same shape and dtype as the input ``x``.
    """
    logger.debug("GEMS CAUSAL_CONV1D_UPDATE")
    assert conv_state_indices is not None, "conv_state_indices is required"

    activation = _normalize_activation(activation)

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines = conv_state.size(0)
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    # NOTE: the kernel writes the result over the input tile, so allocate a
    # separate output buffer rather than aliasing x.
    out = torch.empty_like(x)

    block_n = _DEFAULT_BLOCK_N

    use_tle = _tle_available(x) and width in _TLE_UPDATE_SUPPORTED_WIDTHS
    kernel = (
        _causal_conv1d_update_kernel_tle if use_tle else _causal_conv1d_update_kernel
    )

    stride_w_dim, stride_w_width = weight.stride()
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)

    grid = (batch, triton.cdiv(dim, block_n))

    with torch_device_fn.device(x.device):
        kernel[grid](
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            None,  # num_accepted_tokens
            None,  # query_start_loc
            None,  # block_idx_last_scheduled_token
            None,  # initial_state_idx
            out,
            batch,
            dim,
            seqlen,
            state_len,
            num_cache_lines,
            stride_x_seq,
            stride_x_dim,
            stride_x_token,
            stride_w_dim,
            stride_w_width,
            stride_istate_seq,
            stride_istate_dim,
            stride_istate_token,
            stride_state_indices,
            stride_o_seq,
            stride_o_dim,
            stride_o_token,
            pad_slot_id,
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ("silu", "swish"),
            IS_VARLEN=False,
            IS_APC_ENABLED=False,
            IS_SPEC_DECODING=False,
            NP2_STATELEN=np2_statelen,
            USE_PAD_SLOT=pad_slot_id is not None,
            BLOCK_N=block_n,
        )

    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)
