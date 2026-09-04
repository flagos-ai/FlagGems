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

from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def _next_power_of_2_or_1(x: int) -> int:
    return 1 if x <= 1 else triton.next_power_of_2(x)


def _validate_index_tensor(
    name: str,
    tensor: torch.Tensor,
    ndim: int,
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on the same device as topk_indices")


def _validate_integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_ptr,
    combined_stride,
    lens_ptr,
    pair_metadata_ptr,
    topk_ptr,
    topk_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    PADDED_WINDOW_SIZE: tl.constexpr,
    COMBINED_TOPK: tl.constexpr,
    PADDED_COMBINED_TOPK: tl.constexpr,
    RETURN_PAIR_METADATA: tl.constexpr,
    ASSUME_ORDERED_TOPK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_idx, query_end, num_workers):
        token_in_query = token_idx - query_start
        pos = start_pos + token_in_query
        raw_topk_len = (pos + 1) // COMPRESS_RATIO
        topk_len = tl.minimum(raw_topk_len, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        # Write the -1 padding tail in-kernel so the caller can allocate with
        # torch.empty instead of torch.full: that drops a separate full-buffer
        # memset which this kernel would otherwise mostly overwrite. The upper
        # bound here is COMBINED_TOPK -- the full alignment-padded row width,
        # which is >= topk + window_size -- so [valid_len, COMBINED_TOPK) fills
        # the entire tail, including the alignment padding past topk+window_size.
        # Together with the index stores in [0, valid_len) below, every column
        # of the row is written exactly once (disjoint ranges), so no
        # uninitialized torch.empty value is ever left readable as a valid index.
        valid_len = topk_len + swa_len
        tail_offs = tl.arange(0, PADDED_COMBINED_TOPK)
        tl.store(
            combined_ptr + token_idx * combined_stride + tail_offs,
            -1,
            mask=(tail_offs >= valid_len) & (tail_offs < COMBINED_TOPK),
        )

        offs = tl.arange(0, PADDED_TOP_K)
        mask = offs < topk_len
        topk_vals = tl.load(
            topk_ptr + token_idx * topk_stride + offs, mask=mask, other=-1
        )

        if RETURN_PAIR_METADATA:
            # Each global even row owns exactly one metadata entry.  Compare
            # the source top-k rows directly, before their per-request offset
            # is added, so the fast attention path never has to assume that a
            # particular top-k producer emits candidates in a fixed order.
            if token_idx % 2 == 0:
                has_next = token_idx + 1 < query_end
                next_pos = pos + 1
                next_raw_topk_len = (next_pos + 1) // COMPRESS_RATIO
                next_topk_len = tl.minimum(next_raw_topk_len, TOP_K)
                if ASSUME_ORDERED_TOPK:
                    # The short-row producer writes every candidate in its
                    # natural order. Keep the raw-length bound so this hint is
                    # never extended into the learned top-k region.
                    topk_prefix_matches = (
                        has_next
                        & (
                            (next_raw_topk_len == raw_topk_len)
                            | (next_raw_topk_len == raw_topk_len + 1)
                        )
                        & (next_raw_topk_len <= TOP_K)
                    )
                else:
                    next_topk_vals = tl.load(
                        topk_ptr + (token_idx + 1) * topk_stride + offs,
                        mask=mask
                        & has_next
                        & (
                            (next_topk_len == topk_len)
                            | (next_topk_len == topk_len + 1)
                        ),
                        other=-1,
                    )
                    topk_mismatches = tl.sum(
                        (
                            (topk_vals != next_topk_vals)
                            & mask
                            & has_next
                            & (
                                (next_topk_len == topk_len)
                                | (next_topk_len == topk_len + 1)
                            )
                        ).to(tl.int32),
                        axis=0,
                    )
                    topk_prefix_matches = (
                        has_next
                        & (
                            (next_topk_len == topk_len)
                            | (next_topk_len == topk_len + 1)
                        )
                        & (topk_mismatches == 0)
                    )
                topk_grows = next_topk_len == topk_len + 1
                if WINDOW_SIZE == 0:
                    # The pair consumer uses SWA growth to distinguish modes
                    # 1 and 3. Without a window those descriptors are not
                    # consumable, so keep every pair on the regular path.
                    pair_mode = 0
                else:
                    next_swa_len = tl.minimum(next_pos + 1, WINDOW_SIZE)
                    prefix_pair = (
                        topk_prefix_matches & (valid_len > 0) & (swa_len < WINDOW_SIZE)
                    )
                    shift_pair = (
                        topk_prefix_matches
                        & (swa_len == WINDOW_SIZE)
                        & (next_swa_len == WINDOW_SIZE)
                    )
                    prefix_mode = tl.where(topk_grows, 3, 1)
                    shift_mode = tl.where(topk_grows, 4, 2)
                    pair_mode = tl.where(
                        prefix_pair,
                        prefix_mode,
                        tl.where(shift_pair, shift_mode, 0),
                    )
                encoded_metadata = tl.where(
                    pair_mode != 0,
                    (topk_len << 3) | pair_mode,
                    0,
                )
                tl.store(pair_metadata_ptr + token_idx // 2, encoded_metadata)

        tl.store(
            combined_ptr + token_idx * combined_stride + offs,
            topk_vals + M * batch_idx,
            mask=mask,
        )

        swa_offs = tl.arange(0, PADDED_WINDOW_SIZE)
        tl.store(
            combined_ptr + token_idx * combined_stride + topk_len + swa_offs,
            M * batch_idx + N + swa_offs + pos - swa_len + 1 - gather_start,
            mask=(swa_offs < swa_len) & (swa_offs < WINDOW_SIZE),
        )
        tl.store(lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    *,
    return_pair_metadata: bool = False,
    assume_ordered_topk: bool = False,
) -> (
    Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    if type(return_pair_metadata) is not bool:
        raise TypeError("return_pair_metadata must be a bool")
    if type(assume_ordered_topk) is not bool:
        raise TypeError("assume_ordered_topk must be a bool")
    if assume_ordered_topk and not return_pair_metadata:
        raise ValueError("assume_ordered_topk requires return_pair_metadata=True")

    _validate_integer("window_size", window_size, allow_zero=True)
    _validate_integer("compress_ratio", compress_ratio)
    _validate_integer("topk", topk, allow_zero=True)
    _validate_integer("M", M)
    _validate_integer("N", N, allow_zero=True)

    _validate_index_tensor("topk_indices", topk_indices, 2)
    device = topk_indices.device
    _validate_index_tensor("query_start_loc", query_start_loc, 1, device)
    _validate_index_tensor("seq_lens", seq_lens, 1, device)
    _validate_index_tensor("gather_lens", gather_lens, 1, device)

    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    if topk > topk_indices.shape[1]:
        raise ValueError("topk cannot exceed the row width of topk_indices")
    if query_start_loc.shape != (num_reqs + 1,):
        raise ValueError("query_start_loc must have num_reqs + 1 entries")
    if gather_lens.shape != (num_reqs,):
        raise ValueError("gather_lens must have the same length as seq_lens")
    if num_tokens > 0 and num_reqs == 0:
        raise ValueError("non-empty topk_indices requires at least one request")

    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    # Allocate uninitialized: the kernel writes the -1 padding tail itself, so
    # torch.full's separate full-buffer memset (mostly overwritten) is avoided.
    combined = torch.empty(
        (num_tokens, combined_topk), device=topk_indices.device, dtype=torch.int32
    )
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    if return_pair_metadata:
        pair_metadata = torch.empty(
            ((num_tokens + 1) // 2,),
            device=topk_indices.device,
            dtype=torch.int32,
        )
        pair_metadata_ptr = pair_metadata
    else:
        pair_metadata = None
        # Specialized away when RETURN_PAIR_METADATA is false.
        pair_metadata_ptr = combined
    if num_tokens == 0:
        if pair_metadata is not None:
            return combined, lens, pair_metadata
        return combined, lens
    with torch_device_fn.device(topk_indices.device):
        _combine_topk_swa_indices_kernel[(num_reqs, 128)](
            combined,
            combined.stride(0),
            lens,
            pair_metadata_ptr,
            topk_indices,
            topk_indices.stride(0),
            query_start_loc,
            seq_lens,
            gather_lens,
            M,
            N,
            TOP_K=topk,
            COMPRESS_RATIO=compress_ratio,
            WINDOW_SIZE=window_size,
            PADDED_TOP_K=_next_power_of_2_or_1(topk_indices.shape[-1]),
            PADDED_WINDOW_SIZE=_next_power_of_2_or_1(window_size),
            COMBINED_TOPK=combined_topk,
            PADDED_COMBINED_TOPK=_next_power_of_2_or_1(combined_topk),
            RETURN_PAIR_METADATA=return_pair_metadata,
            ASSUME_ORDERED_TOPK=assume_ordered_topk,
        )
    if pair_metadata is not None:
        return combined, lens, pair_metadata
    return combined, lens


__all__ = ["combine_topk_swa_indices"]
