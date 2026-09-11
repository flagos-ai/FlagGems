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

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)
_USE_DEVICE_ASSERT = runtime_device.vendor_name in ("nvidia", "hygon")


@libentry()
@triton.jit(debug=_USE_DEVICE_ASSERT)
def _embedding_bag_forward_kernel(
    weight,
    indices,
    offsets,
    per_sample_weights,
    output,
    offset_to_bag,
    bag_size,
    max_indices,  # maximum-value embedding indices
    error,
    num_indices: tl.constexpr,
    num_offsets: tl.constexpr,
    num_bags: tl.constexpr,
    embedding_dim: tl.constexpr,
    num_weights: tl.constexpr,
    stride_w0: tl.constexpr,  # weight row stride
    stride_w1: tl.constexpr,  # weight feature stride
    stride_i: tl.constexpr,  # indices stride
    stride_o: tl.constexpr,  # offsets stride
    stride_p: tl.constexpr,  # per-sample-weight stride
    padding: tl.constexpr,
    mode: tl.constexpr,
    has_per_sample: tl.constexpr,
    block_l: tl.constexpr,  # bag-length block size
    block_d: tl.constexpr,  # embedding-dimension block size
    ascend_abi: tl.constexpr = False,  # Ascend auxiliary-output application binary interface
    bag_size_b: tl.constexpr = False,  # bag-size output uses num_bags entries
    search_steps: tl.constexpr = 0,
    index_block: tl.constexpr = 128,
    async_assert: tl.constexpr = False,  # asynchronous device assertion
    serial_max: tl.constexpr = False,  # serial maximum reduction
    key_max: tl.constexpr = False,  # maximum reduction using ordered keys
    reciprocal_mean: tl.constexpr = False,
    skip_pad_scale: tl.constexpr = False,  # skip padding-scale adjustment
):
    pid = tle.program_id(0)
    programs = tle.num_programs(0)
    dim_tiles = tl.maximum(tl.cdiv(embedding_dim, block_d), 1)
    acc_dtype: tl.constexpr = (
        tl.float64 if weight.dtype.element_ty == tl.float64 else tl.float32
    )
    rows = tl.arange(0, block_l)
    cols = tl.arange(0, block_d)
    total_work = num_bags * dim_tiles
    for work in range(pid, total_work, programs):
        bag = work // dim_tiles
        dim_tile = work % dim_tiles
        dim = dim_tile * block_d + cols
        begin = tl.load(offsets + bag * stride_o).to(tl.int64)
        end = tl.load(
            offsets + (bag + 1) * stride_o, bag + 1 < num_offsets, other=0
        ).to(tl.int64)
        end = tl.where(bag + 1 < num_offsets, end, num_indices)
        valid_offsets = (begin >= 0) & (end >= begin) & (end <= num_indices)
        valid_offsets &= (bag != 0) | (begin == 0)
        if num_offsets > num_bags:
            valid_offsets &= (bag != num_bags - 1) | (end == num_indices)
        if dim_tile == 0:
            code = tl.where(valid_offsets, 0, 2)
            index_lane = tl.arange(0, index_block)
            # Fixed partitions keep mapping writes disjoint even when a
            # malformed offset sequence makes different bags overlap.
            if num_indices > 0:
                for input_base in range(
                    bag * index_block,
                    num_indices,
                    tl.maximum(num_bags, 1) * index_block,
                ):
                    pos = input_base + index_lane
                    in_bounds = pos < num_indices
                    idx = tl.load(indices + pos * stride_i, in_bounds, other=0)
                    invalid = in_bounds & ((idx < 0) | (idx >= num_weights))
                    code |= tl.max(invalid.to(tl.int32), 0)
                    low = tl.full((index_block,), 0, tl.int64)
                    high = tl.full((index_block,), num_bags, tl.int64)
                    for _ in range(search_steps):
                        mid = (low + high) // 2
                        off = tl.load(
                            offsets + mid * stride_o,
                            in_bounds & (mid < num_bags),
                            other=num_indices,
                        )
                        right = (mid < num_bags) & (off <= pos)
                        low = tl.where(right, mid + 1, low)
                        high = tl.where(right, high, mid)
                    tl.store(offset_to_bag + pos, tl.maximum(low - 1, 0), in_bounds)
            if async_assert:
                tl.device_assert(code == 0, "embedding_bag: invalid index or offset")
            else:
                tl.store(error + bag, code)
        end = tl.where(valid_offsets, end, begin)
        begin = tl.minimum(tl.maximum(begin, 0), num_indices)
        end = tl.minimum(tl.maximum(end, begin), num_indices)
        if mode == 2 and serial_max:
            count = tl.full((), 0, tl.int64)
            best = tl.full((block_d,), 0, acc_dtype)
            winner = tl.full((block_d,), -1, tl.int64)
            for pos in range(begin, end):
                idx = tl.load(indices + pos * stride_i).to(tl.int64)
                valid = (idx >= 0) & (idx < num_weights) & (idx != padding)
                values = tl.load(
                    weight + idx * stride_w0 + dim * stride_w1,
                    valid & (dim < embedding_dim),
                    other=0.0,
                ).to(acc_dtype)
                replace = valid & ((count == 0) | (values > best))
                best = tl.where(replace, values, best)
                winner = tl.where(replace, idx, winner)
                count += valid.to(tl.int64)
            result = best
            tl.store(
                max_indices + bag * embedding_dim + dim, winner, dim < embedding_dim
            )
        else:
            count = tl.full((), 0, tl.int64)
            if mode == 2:
                if key_max:
                    best = tl.full((block_d,), -9223372036854775808, tl.int64)
                else:
                    best = tl.full((block_d,), float("-inf"), acc_dtype)
                best_pos = tl.full(
                    (block_d,),
                    num_indices,
                    tl.int32 if key_max and num_indices < 2147483647 else tl.int64,
                )
                first = tl.full((), num_indices, tl.int64)
            else:
                acc = tl.zeros((block_l, block_d), acc_dtype)
            if num_indices > 0:
                for base in range(begin, end, block_l):
                    pos = base + rows
                    active = pos < end
                    idx = tl.load(indices + pos * stride_i, active, other=0).to(
                        tl.int64
                    )
                    valid = active & (idx >= 0) & (idx < num_weights) & (idx != padding)
                    if embedding_dim > 0:
                        values = tl.load(
                            weight
                            + idx[:, None] * stride_w0
                            + dim[None, :] * stride_w1,
                            valid[:, None] & (dim[None, :] < embedding_dim),
                            other=0.0,
                        ).to(acc_dtype)
                    else:
                        values = tl.zeros((block_l, block_d), acc_dtype)
                    count += tl.sum(valid.to(tl.int32), 0)
                    if mode == 2:
                        first = tl.minimum(
                            first, tl.min(tl.where(valid, pos, num_indices), 0)
                        )
                        if key_max:
                            bits = values.to(tl.int64, bitcast=True)
                            magnitude = bits & 0x7FFFFFFFFFFFFFFF
                            numeric = valid[:, None] & (magnitude <= 0x7FF0000000000000)
                            ordered = bits ^ tl.where(bits < 0, 0x7FFFFFFFFFFFFFFF, 0)
                            ordered = tl.where(magnitude == 0, 0, ordered)
                            candidate = tl.where(numeric, ordered, -9223372036854775808)
                            # Ordered IEEE-754 bits permit an integer argmax.
                            # Canonical zero keys preserve the first signed-zero tie;
                            # the original winning value is loaded below.
                            tile_best, relative = tl.max(
                                candidate, 0, return_indices=True
                            )
                            tile_pos = tl.where(
                                tile_best > -9223372036854775808,
                                base + relative,
                                num_indices,
                            ).to(tl.int32 if num_indices < 2147483647 else tl.int64)
                        else:
                            numeric = valid[:, None] & (values == values)
                            candidate = tl.where(numeric, values, float("-inf"))
                            tile_best = tl.max(candidate, 0)
                            tile_pos = tl.min(
                                tl.where(
                                    numeric & (values == tile_best[None, :]),
                                    pos[:, None],
                                    num_indices,
                                ),
                                0,
                            )
                        replace = (tile_best > best) | (
                            (tile_best == best) & (tile_pos < best_pos)
                        )
                        best = tl.where(replace, tile_best, best)
                        best_pos = tl.where(replace, tile_pos, best_pos)
                    else:
                        if has_per_sample:
                            # CUDA ATen multiplies zero padding values by their sample
                            # weights, preserving NaN from a nonfinite padding weight.
                            scale = tl.load(
                                per_sample_weights + pos * stride_p,
                                valid if skip_pad_scale else active,
                                other=0.0,
                            )
                            values *= scale[:, None].to(acc_dtype)
                        acc += values
            if mode == 2:
                if num_indices == 0 or embedding_dim == 0:
                    winner = tl.full((block_d,), -1, tl.int64)
                    result = tl.zeros((block_d,), acc_dtype)
                else:
                    first_idx = tl.load(
                        indices + first * stride_i, first < num_indices, other=0
                    )
                    first_value = tl.load(
                        weight + first_idx.to(tl.int64) * stride_w0 + dim * stride_w1,
                        (first < num_indices)
                        & (first_idx >= 0)
                        & (first_idx < num_weights)
                        & (dim < embedding_dim),
                        other=0.0,
                    )
                    if key_max:
                        first_nan = (
                            first_value.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF
                        ) > 0x7FF0000000000000
                        best_pos = tl.where(
                            first_nan, first.to(best_pos.dtype), best_pos
                        )
                    else:
                        best_pos = tl.where(first_value != first_value, first, best_pos)
                    winner = tl.load(
                        indices + best_pos.to(tl.int64) * stride_i,
                        best_pos < num_indices,
                        other=-1,
                    )
                    result = tl.load(
                        weight + winner.to(tl.int64) * stride_w0 + dim * stride_w1,
                        (winner >= 0) & (winner < num_weights) & (dim < embedding_dim),
                        other=0.0,
                    )
                if ascend_abi:
                    winner = tl.where(count == 0, 0, winner)
                if embedding_dim > 0:
                    tl.store(
                        max_indices + bag * embedding_dim + dim,
                        winner,
                        dim < embedding_dim,
                    )
            else:
                result = tl.sum(acc, 0)
                if mode == 1:
                    if reciprocal_mean:
                        reciprocal = 1.0 / tl.maximum(count, 1).to(acc_dtype)
                        result = result * reciprocal
                    else:
                        result = result / tl.maximum(count, 1).to(acc_dtype)
        if embedding_dim > 0:
            tl.store(output + bag * embedding_dim + dim, result, dim < embedding_dim)
        if dim_tile == 0:
            if ascend_abi and mode == 0:
                tl.store(bag_size + bag, 0)
            else:
                tl.store(bag_size + bag, count)
    if num_offsets > num_bags and not bag_size_b:
        if pid == 0:
            tl.store(bag_size + num_bags, 0)
    if num_bags == 0:
        if pid == 0:
            only_offset = tl.load(offsets)
            if async_assert:
                tl.device_assert(
                    only_offset == 0, "embedding_bag: invalid terminal offset"
                )
            else:
                tl.store(error, tl.where(only_offset == 0, 0, 2))


def _embedding_bag_impl(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
    forward_only=False,
    *,
    _kernel=None,
    _async_assert=None,
    _launch_kwargs=None,
    _check_error=None,
    _block_config=None,
):
    """Embedding-bag forward with target ATen auxiliary tensor conventions.

    The low-level ``-1`` padding sentinel means no padding; other legal negative
    padding indices are normalized. Unlike legacy CUDA, a supplied terminal
    offset must equal ``indices.numel()``. NVIDIA reports invalid tensor data
    asynchronously through device assertions, matching CUDA ATen. Backend
    implementations may provide their own launch and error checks. Other vendors
    read error flags on the host and raise a recoverable exception; that path
    synchronizes and cannot be captured in a device graph. All indirect memory
    accesses remain masked regardless of the error-reporting mechanism.
    """
    logger.debug("GEMS _EMBEDDING_BAG")
    if weight.ndim != 2 or indices.ndim != 1 or offsets.ndim != 1:
        raise RuntimeError("weight must be 2D and indices and offsets must be 1D")
    if weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(
            "weight must have float16, bfloat16, float32 or float64 dtype"
        )
    if indices.dtype not in (torch.int32, torch.int64) or offsets.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise RuntimeError("indices and offsets must have int32 or int64 dtype")
    if indices.device != weight.device or offsets.device != weight.device:
        raise RuntimeError("weight, indices and offsets must be on the same device")
    if mode not in (0, 1, 2):
        raise RuntimeError("mode must be 0 (sum), 1 (mean), or 2 (max)")
    if per_sample_weights is not None:
        if mode != 0:
            raise RuntimeError("per_sample_weights is only supported for mode='sum'")
        if (
            per_sample_weights.ndim != 1
            or per_sample_weights.numel() != indices.numel()
        ):
            raise RuntimeError(
                "per_sample_weights must be 1D with the same size as indices"
            )
        if (
            per_sample_weights.device != weight.device
            or per_sample_weights.dtype != weight.dtype
        ):
            raise RuntimeError(
                "per_sample_weights must have the same device and dtype as weight"
            )
    n, o = indices.numel(), offsets.numel()
    v, d = weight.shape
    if padding_idx != -1:
        if padding_idx < -v or padding_idx >= v:
            raise RuntimeError("padding_idx must be within the number of embeddings")
        if padding_idx < 0:
            padding_idx += v
    if include_last_offset and o == 0:
        raise RuntimeError("include_last_offset requires at least one offset")
    bags = o - int(include_last_offset)
    if bags == 0 and n != 0:
        raise RuntimeError("nonempty indices require at least one bag")
    index_dtype = (
        torch.int64 if torch.int64 in (indices.dtype, offsets.dtype) else torch.int32
    )
    out = torch.empty((bags, d), device=weight.device, dtype=weight.dtype)
    offset2bag = torch.empty((n,), device=weight.device, dtype=index_dtype)
    ascend_abi = runtime_device.vendor_name == "ascend"
    async_assert = _USE_DEVICE_ASSERT if _async_assert is None else _async_assert
    kernel = _embedding_bag_forward_kernel if _kernel is None else _kernel
    bag_size_b = ascend_abi or (
        runtime_device.vendor_name == "mthreads" and not forward_only
    )
    bag_size = torch.empty(
        (bags if bag_size_b else o,), device=weight.device, dtype=index_dtype
    )
    if ascend_abi and mode != 2:
        max_indices = bag_size
    else:
        max_indices = torch.empty(
            (bags, d) if mode == 2 else (0,), device=weight.device, dtype=index_dtype
        )
    with torch_device_fn.device(weight.device):
        error = None
        error_blocks = max(bags, 1)
        if n or o:
            if not async_assert:
                error = torch.empty(
                    (error_blocks,), device=weight.device, dtype=torch.int32
                )
        if o:
            # One masked feature tile also defines helpers for zero-width weights.
            block_d = min(64, triton.next_power_of_2(max(d, 1)))
            block_l = 16 if n <= bags * 16 else 32
            num_warps = 4
            serial_max = key_max = reciprocal_mean = False
            # Wide or long FP64 bags are limited by double-precision reduction
            # work and register pressure. Keep the ordinary path for small bags.
            if (
                runtime_device.vendor_name == "nvidia"
                and forward_only
                and weight.dtype == torch.float64
                and bags >= 128
            ):
                if mode == 2:
                    if n > bags * 16 and d <= 256:
                        serial_max = True
                        block_d = 64
                        num_warps = 1 if d <= 64 else 2
                    else:
                        key_max = True
                        block_d = 128
                        block_l = 8 if n <= bags * 8 else 16
                else:
                    reciprocal_mean = mode == 1
                    if d <= 64:
                        block_d = 64
                        block_l = 4 if mode == 1 else 8
                    elif d <= 128:
                        block_d = 128 if mode == 1 else 64
                        block_l = 16 if per_sample_weights is not None else 8
                    else:
                        block_d = 128 if d <= 256 or mode == 1 else 64
                        block_l = 4
            index_block = 128
            if _block_config is not None:
                block_l, block_d, index_block, num_warps = _block_config
            launch_kwargs = {"debug": True} if async_assert else {}
            if _launch_kwargs is not None:
                launch_kwargs.update(_launch_kwargs)
            grid = (min(max(bags * max(triton.cdiv(d, block_d), 1), 1), 65535),)
            kernel[grid](
                weight,
                indices,
                offsets,
                per_sample_weights,
                out,
                offset2bag,
                bag_size,
                max_indices,
                error,
                n,
                o,
                bags,
                d,
                v,
                weight.stride(0),
                weight.stride(1),
                indices.stride(0),
                offsets.stride(0),
                per_sample_weights.stride(0) if per_sample_weights is not None else 0,
                padding_idx,
                mode,
                per_sample_weights is not None,
                block_l,
                block_d,
                ascend_abi=ascend_abi,
                bag_size_b=bag_size_b,
                # A single bag maps every position to zero. Avoid a redundant
                # vector gather from one scalar offset on Ascend's compiler.
                search_steps=bags.bit_length() if bags > 1 else 0,
                index_block=index_block,
                async_assert=async_assert,
                serial_max=serial_max,
                key_max=key_max,
                reciprocal_mean=reciprocal_mean,
                num_warps=num_warps,
                # Keep assertions enabled even with TRITON_DEBUG=0.
                **launch_kwargs,
            )
        if (n or o) and not async_assert and _check_error is not None:
            _check_error(error, error_blocks)
        elif (n or o) and not async_assert:
            code = max(error.tolist())
            if code & 1:
                raise RuntimeError(
                    "embedding_bag: indices must be in [0, weight.size(0))"
                )
            if code & 2:
                raise RuntimeError(
                    "embedding_bag: offsets must start at 0, be nondecreasing and in [0, indices.numel()]; "
                    "the terminal offset must equal indices.numel() when include_last_offset=True"
                )
    return out, offset2bag, bag_size, max_indices


def _embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    return _embedding_bag_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


def _embedding_bag_forward_only(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS _EMBEDDING_BAG_FORWARD_ONLY")
    return _embedding_bag_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        forward_only=True,
    )
