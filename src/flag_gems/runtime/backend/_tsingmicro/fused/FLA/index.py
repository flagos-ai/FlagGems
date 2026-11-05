# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch
import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # The chunk counts size every downstream grid and buffer, so a single bad
    # value there is fatal (or silently wrong).  Derive them on the host from
    # the n_seq + 1 entry metadata instead of reading a device-computed scalar
    # back through ``torch.arange(chunk_offsets[-1])``: that read is a
    # synchronising 4-byte D2H copy of a device scalar, and one stale/clobbered
    # value there aborted a chunked prefill with
    # "Trying to create tensor with negative dimension -32".  cu_seqlens is
    # minuscule, so the host copy costs nothing.
    host = cu_seqlens.detach().to("cpu")
    chunk_counts = (host[1:] - host[:-1] + (chunk_size - 1)) // chunk_size
    if bool((chunk_counts < 0).any()):
        raise ValueError(
            f"prepare_chunk_indices: cu_seqlens is not monotonically "
            f"non-decreasing (chunk_size={chunk_size}): {host.tolist()}"
        )
    chunk_offsets = torch.cat([chunk_counts.new_zeros(1), chunk_counts]).cumsum(-1)
    chunk_arange = torch.arange(chunk_offsets[-1])
    seq_ids = torch.repeat_interleave(torch.arange(chunk_counts.numel()), chunk_counts)
    chunk_ids = chunk_arange - torch.repeat_interleave(chunk_offsets[:-1], chunk_counts)
    return torch.stack([seq_ids, chunk_ids], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
