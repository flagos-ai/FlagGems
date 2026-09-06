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
    quad_metadata_ptr=None,
    RETURN_QUAD_METADATA: tl.constexpr = False,
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

                if RETURN_QUAD_METADATA:
                    if token_idx % 4 == 0:
                        # Certify all three prefix boundaries while the source
                        # indices are available.  The four queries must belong
                        # to this request, whose SWA addresses are contiguous.
                        have_quad = (
                            (token_idx + 3 < query_end)
                            & (COMPRESS_RATIO == 4)
                            & (WINDOW_SIZE > 0)
                            & (swa_len > 0)
                            # The ordered hint suppresses pair descriptors
                            # outside the complete-candidate region. Quads
                            # need both pair descriptors to retain counts.
                            & (
                                (not ASSUME_ORDERED_TOPK)
                                | ((pos + 4) // COMPRESS_RATIO <= TOP_K)
                            )
                        )
                        third_topk_len = tl.minimum((pos + 3) // COMPRESS_RATIO, TOP_K)
                        row1 = tl.load(
                            topk_ptr + (token_idx + 1) * topk_stride + offs,
                            mask=have_quad & (offs < next_topk_len),
                            other=-1,
                        )
                        row2 = tl.load(
                            topk_ptr + (token_idx + 2) * topk_stride + offs,
                            mask=have_quad & (offs < third_topk_len),
                            other=-1,
                        )
                        row3 = tl.load(
                            topk_ptr + (token_idx + 3) * topk_stride + offs,
                            mask=have_quad & (offs < third_topk_len),
                            other=-1,
                        )
                        mismatches = (
                            ((topk_vals != row1) & (offs < topk_len))
                            | ((row1 != row2) & (offs < next_topk_len))
                            | ((row2 != row3) & (offs < third_topk_len))
                        )
                        quad_matches = tl.sum(mismatches.to(tl.int32), axis=0) == 0
                        tl.store(
                            quad_metadata_ptr + token_idx // 4,
                            (have_quad & quad_matches).to(tl.int32),
                        )

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


@triton.jit
def _store_combined_quad_row(
    combined_ptr,
    lens_ptr,
    row,
    owned,
    values,
    topk_len,
    swa_len,
    pos,
    request_offset,
    gather_start,
    N,
    TOP_K: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    # Fixed disjoint spans cover compressed IDs, SWA IDs and all -1 padding.
    # TOP_K is also the source row width; the final window span cannot contain
    # compressed IDs. This avoids a separate wide, masked padding store.
    width: tl.constexpr = TOP_K + WINDOW_SIZE
    cols = tl.arange(0, TOP_K)
    valid_len = topk_len + swa_len
    sliding = request_offset + N + cols - topk_len + pos - swa_len + 1 - gather_start
    first = tl.where(
        cols < topk_len,
        values + request_offset,
        tl.where(cols < valid_len, sliding, -1),
    )
    tl.store(combined_ptr + row * width + cols, first, mask=owned)
    end_cols = TOP_K + tl.arange(0, WINDOW_SIZE)
    last = tl.where(
        end_cols < valid_len,
        request_offset + N + end_cols - topk_len + pos - swa_len + 1 - gather_start,
        -1,
    )
    tl.store(combined_ptr + row * width + end_cols, last, mask=owned)
    tl.store(lens_ptr + row, valid_len, mask=owned)


@triton.jit
def _encode_quad_pair(match, l0, l1, s0, s1, WINDOW_SIZE: tl.constexpr):
    grows = l1 == l0 + 1
    prefix = match & (l0 + s0 > 0) & (s0 < WINDOW_SIZE)
    shifted = match & (s0 == WINDOW_SIZE) & (s1 == WINDOW_SIZE)
    mode = tl.where(
        prefix, tl.where(grows, 3, 1), tl.where(shifted, tl.where(grows, 4, 2), 0)
    )
    return tl.where(mode != 0, (l0 << 3) | mode, 0)


@triton.jit
def _combine_topk_swa_indices_quad_kernel(
    topk_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    combined_ptr,
    lens_ptr,
    pair_metadata_ptr,
    quad_metadata_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    # Specialization for compression ratio 4. Reuse four source rows for both
    # pair descriptors and all three strict quad-prefix comparisons.
    req = tl.program_id(0)
    worker = tl.program_id(1)
    workers = tl.num_programs(1)
    base = tl.load(query_start_loc_ptr)
    qs = tl.load(query_start_loc_ptr + req) - base
    qe = tl.load(query_start_loc_ptr + req + 1) - base
    seq = tl.load(seq_lens_ptr + req)
    gathered = tl.load(gather_lens_ptr + req)
    start_pos = seq - (qe - qs)
    gather_start = seq - gathered
    offsets = tl.arange(0, TOP_K)
    for group in range(qs // 4 + worker, tl.cdiv(qe, 4), workers):
        r0 = group * 4
        r1 = r0 + 1
        r2 = r0 + 2
        r3 = r0 + 3
        # A group may cross a request boundary. Only the owning request writes
        # each row, pair and quad; descriptors never certify cross-request reuse.
        ok0 = (r0 >= qs) & (r0 < qe)
        ok1 = (r1 >= qs) & (r1 < qe)
        ok2 = (r2 >= qs) & (r2 < qe)
        ok3 = (r3 >= qs) & (r3 < qe)
        p0 = start_pos + r0 - qs
        p1 = p0 + 1
        p2 = p0 + 2
        p3 = p0 + 3
        l0 = tl.minimum((p0 + 1) // 4, TOP_K)
        l1 = tl.minimum((p1 + 1) // 4, TOP_K)
        l2 = tl.minimum((p2 + 1) // 4, TOP_K)
        l3 = tl.minimum((p3 + 1) // 4, TOP_K)
        s0 = tl.minimum(p0 + 1, WINDOW_SIZE)
        s1 = tl.minimum(p1 + 1, WINDOW_SIZE)
        s2 = tl.minimum(p2 + 1, WINDOW_SIZE)
        s3 = tl.minimum(p3 + 1, WINDOW_SIZE)
        v0 = tl.load(
            topk_ptr + r0 * TOP_K + offsets,
            mask=ok0 & (offsets < l0),
            other=-1,
        )
        v1 = tl.load(
            topk_ptr + r1 * TOP_K + offsets,
            mask=ok1 & (offsets < l1),
            other=-1,
        )
        v2 = tl.load(
            topk_ptr + r2 * TOP_K + offsets,
            mask=ok2 & (offsets < l2),
            other=-1,
        )
        v3 = tl.load(
            topk_ptr + r3 * TOP_K + offsets,
            mask=ok3 & (offsets < l3),
            other=-1,
        )
        eq01 = tl.sum(((v0 != v1) & (offsets < l0)).to(tl.int32), 0) == 0
        eq12 = tl.sum(((v1 != v2) & (offsets < l1)).to(tl.int32), 0) == 0
        eq23 = tl.sum(((v2 != v3) & (offsets < l2)).to(tl.int32), 0) == 0
        match01 = ok0 & ok1 & ((l1 == l0) | (l1 == l0 + 1)) & eq01
        match23 = ok2 & ok3 & ((l3 == l2) | (l3 == l2 + 1)) & eq23
        tl.store(
            pair_metadata_ptr + group * 2,
            _encode_quad_pair(match01, l0, l1, s0, s1, WINDOW_SIZE),
            mask=ok0,
        )
        tl.store(
            pair_metadata_ptr + group * 2 + 1,
            _encode_quad_pair(match23, l2, l3, s2, s3, WINDOW_SIZE),
            mask=ok2,
        )
        quad = ok0 & ok1 & ok2 & ok3 & (s0 > 0) & eq01 & eq12 & eq23
        tl.store(quad_metadata_ptr + group, quad.to(tl.int32), mask=ok0)
        _store_combined_quad_row(
            combined_ptr,
            lens_ptr,
            r0,
            ok0,
            v0,
            l0,
            s0,
            p0,
            req * M,
            gather_start,
            N,
            TOP_K,
            WINDOW_SIZE,
        )
        _store_combined_quad_row(
            combined_ptr,
            lens_ptr,
            r1,
            ok1,
            v1,
            l1,
            s1,
            p1,
            req * M,
            gather_start,
            N,
            TOP_K,
            WINDOW_SIZE,
        )
        _store_combined_quad_row(
            combined_ptr,
            lens_ptr,
            r2,
            ok2,
            v2,
            l2,
            s2,
            p2,
            req * M,
            gather_start,
            N,
            TOP_K,
            WINDOW_SIZE,
        )
        _store_combined_quad_row(
            combined_ptr,
            lens_ptr,
            r3,
            ok3,
            v3,
            l3,
            s3,
            p3,
            req * M,
            gather_start,
            N,
            TOP_K,
            WINDOW_SIZE,
        )


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
    return_quad_metadata: bool = False,
) -> (
    Tuple[torch.Tensor, torch.Tensor]
    | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Combine compressed and window IDs, optionally certifying shared prefixes.

    Quad flags require pair metadata and describe four rows of the same
    request. Every compressed-prefix boundary is compared exactly, including
    when assume_ordered_topk is enabled for the existing pair path.
    """
    if type(return_pair_metadata) is not bool:
        raise TypeError("return_pair_metadata must be a bool")
    if type(assume_ordered_topk) is not bool:
        raise TypeError("assume_ordered_topk must be a bool")
    if type(return_quad_metadata) is not bool:
        raise TypeError("return_quad_metadata must be a bool")
    if return_quad_metadata and not return_pair_metadata:
        raise ValueError("return_quad_metadata requires return_pair_metadata=True")
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
    quad_metadata = (
        torch.empty(
            ((num_tokens + 3) // 4,),
            device=topk_indices.device,
            dtype=torch.int32,
        )
        if return_quad_metadata
        else None
    )
    if num_tokens == 0:
        if quad_metadata is not None:
            return combined, lens, pair_metadata, quad_metadata
        if pair_metadata is not None:
            return combined, lens, pair_metadata
        return combined, lens
    with torch_device_fn.device(topk_indices.device):
        if (
            return_quad_metadata
            and not assume_ordered_topk
            and compress_ratio == 4
            and window_size == 128
            and topk == topk_indices.shape[1] == 2048
            and num_tokens >= 1024
        ):
            _combine_topk_swa_indices_quad_kernel[(num_reqs, 256)](
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                combined,
                lens,
                pair_metadata,
                quad_metadata,
                M,
                N,
                TOP_K=topk,
                WINDOW_SIZE=window_size,
                num_warps=4,
            )
            return combined, lens, pair_metadata, quad_metadata
        # Quad comparisons increase each worker's live state. More workers
        # avoid serializing many rows behind each comparison in prefill.
        workers = (1024 if num_reqs == 1 else 512) if return_quad_metadata else 128
        _combine_topk_swa_indices_kernel[(num_reqs, workers)](
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
            quad_metadata_ptr=quad_metadata,
            RETURN_QUAD_METADATA=return_quad_metadata,
        )
    if quad_metadata is not None:
        return combined, lens, pair_metadata, quad_metadata
    if pair_metadata is not None:
        return combined, lens, pair_metadata
    return combined, lens


__all__ = ["combine_topk_swa_indices"]
