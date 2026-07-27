# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging

import torch

logger = logging.getLogger(__name__)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS_KUNLUNXIN FUSED RECURRENT GATED DELTA RULE FWD")
    batch, seq_len, heads, _ = q.shape
    value_heads = v.shape[2]
    output = torch.zeros_like(v)
    source_state = initial_state.clone() if inplace_final_state else initial_state
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = torch.zeros(
            seq_len,
            value_heads,
            k.shape[-1],
            v.shape[-1],
            dtype=initial_state.dtype,
            device=initial_state.device,
        )

    num_sequences = batch if cu_seqlens is None else len(cu_seqlens) - 1
    for sequence in range(num_sequences):
        if cu_seqlens is None:
            batch_idx, begin, end = sequence, 0, seq_len
        else:
            batch_idx = 0
            begin = cu_seqlens[sequence].item()
            end = cu_seqlens[sequence + 1].item()

        initial_idx = (
            sequence
            if ssm_state_indices is None
            else ssm_state_indices[begin].item()
        )
        for value_head in range(value_heads):
            query_head = value_head // (value_heads // heads)
            state = source_state[initial_idx, value_head].float().clone()
            for position in range(begin, end):
                query = q[batch_idx, position, query_head].float()
                key = k[batch_idx, position, query_head].float()
                value = v[batch_idx, position, value_head].float()
                if use_qk_l2norm_in_kernel:
                    query = query / (query.norm() + 1e-6)
                    key = key / (key.norm() + 1e-6)
                query = query * scale
                state = state * torch.exp(g[batch_idx, position, value_head].float())
                value = value - (state * key[:, None]).sum(0)
                value = value * beta[batch_idx, position, value_head].float()
                state = state + key[:, None] * value[None, :]
                output[batch_idx, position, value_head] = (
                    state * query[:, None]
                ).sum(0).to(output.dtype)

                state_idx = (
                    sequence
                    if ssm_state_indices is None
                    else ssm_state_indices[position].item()
                )
                if inplace_final_state:
                    final_state[state_idx, value_head] = state.to(final_state.dtype)
                else:
                    final_state[position, value_head] = state.to(final_state.dtype)

    return output, final_state
