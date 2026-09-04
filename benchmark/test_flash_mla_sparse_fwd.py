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

import dataclasses
import math
import random
from typing import List, Optional

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import base

try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd as vllm_flash_mla_sparse_fwd,
    )

    HAS_VLLM_FLASHMLA_SPARSE = True
except ImportError:
    HAS_VLLM_FLASHMLA_SPARSE = False


# Keep a strong reference to the source tensor so an id cannot be reused while
# its padded value remains cached.
_PADDED_ATTN_SINK_CACHE = {}


# Match the prefill gather workspace used by the 8K reproduction setup:
# N = ceil(max_model_len / compress_ratio) = 2048 and
# M = N + window_size + max_num_batched_tokens = 34944.  The workspace keeps
# four request slots even when the current chunk contains fewer requests.
_PAIR_WORKSPACE_REQUESTS = 4
_PAIR_COMPRESSED_POOL_SIZE = 2048
_PAIR_WINDOW_SIZE = 128
_PAIR_MAX_NUM_BATCHED_TOKENS = 32768
_PAIR_REQUEST_STRIDE = (
    _PAIR_COMPRESSED_POOL_SIZE + _PAIR_WINDOW_SIZE + _PAIR_MAX_NUM_BATCHED_TOKENS
)


def vllm_flash_mla_sparse_fwd_with_head_padding(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    attn_sink=None,
    topk_length=None,
    *,
    pair_metadata=None,
    pair_window_size=0,
    return_stats=True,
):
    """Run the reference through its legacy minimum-head padding path.

    The reference kernel only accepts 64 or 128 query heads.  Keeping the input
    at its logical head count lets the benchmark compare the full legacy path
    (pad, reference call, and output slice) with a direct low-head call.
    """
    # For pair-work-item cases, compare against the same low-head kernel with
    # pairing disabled.  The vLLM reference requires padding to 64 heads,
    # which would measure a different workload instead of the optimization.
    if pair_metadata is not None:
        return flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            d_v,
            attn_sink,
            topk_length,
            return_stats=return_stats,
            pair_metadata=None,
            pair_window_size=0,
        )

    # Pair metadata is a FlagGems execution strategy and must not leak into
    # the unmodified vLLM reference provider.
    del pair_window_size, return_stats
    logical_h_q = q.shape[1]
    if logical_h_q in (64, 128):
        return vllm_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length
        )

    assert 0 < logical_h_q < 64
    padded_h_q = 64
    q_padded = F.pad(q, (0, 0, 0, padded_h_q - logical_h_q))
    if attn_sink is not None:
        cache_key = (id(attn_sink), logical_h_q, padded_h_q)
        if cache_key not in _PADDED_ATTN_SINK_CACHE:
            _PADDED_ATTN_SINK_CACHE[cache_key] = (
                attn_sink,
                F.pad(
                    attn_sink,
                    (0, padded_h_q - logical_h_q),
                    value=float("-inf"),
                ),
            )
        attn_sink = _PADDED_ATTN_SINK_CACHE[cache_key][1]

    output, max_logits, lse = vllm_flash_mla_sparse_fwd(
        q_padded, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    return (
        output[:, :logical_h_q, :],
        max_logits[:, :logical_h_q],
        lse[:, :logical_h_q],
    )


@dataclasses.dataclass
class TestParam:
    # Instruct pytest to ignore this class
    __test__ = False

    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    is_all_indices_invalid: bool = False
    num_warmup: int = 5
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    fixed_topk_length: Optional[int] = None
    kv_tokens_per_query: Optional[int] = None
    queries_per_kv_region: Optional[int] = None
    kv_tokens_per_region: Optional[int] = None
    compressed_pool_size: int = 0
    kv_workspace_regions: int = 0
    use_pair_work_items: bool = False
    pair_window_size: int = 0
    compressed_topk: int = 0
    sm_scale: float = 0.5
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flag_gems.device


# used by make_input_flashmla
_flashmla_sparse_counter = 0


class FlashmlaSparseBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd",
            vllm_flash_mla_sparse_fwd_with_head_padding,
            [torch.bfloat16],
        )
        self.set_gems(flag_gems.flash_mla_sparse_fwd)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        for param in FlashmlaSparseBenchmark.get_performance_test_params_flashmla():
            yield from FlashmlaSparseBenchmark.make_input_flashmla(param)

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def get_performance_test_params_flashmla():
        cases = (
            [
                # Production-layout prefill: C4 compression, a 2048-entry
                # compressed top-k capacity, and a 128-token sliding window.
                # Pair metadata covers append-only, window-shift, and both
                # compression-growth cases.
                TestParam(
                    s_q,
                    _PAIR_WORKSPACE_REQUESTS * _PAIR_REQUEST_STRIDE,
                    2048 + _PAIR_WINDOW_SIZE,
                    h_q=4,
                    d_qk=512,
                    have_attn_sink=have_attn_sink,
                    have_topk_length=True,
                    queries_per_kv_region=1024,
                    kv_tokens_per_region=_PAIR_REQUEST_STRIDE,
                    compressed_pool_size=_PAIR_COMPRESSED_POOL_SIZE,
                    kv_workspace_regions=_PAIR_WORKSPACE_REQUESTS,
                    use_pair_work_items=True,
                    pair_window_size=_PAIR_WINDOW_SIZE,
                    compressed_topk=2048,
                    sm_scale=512**-0.5,
                )
                for s_q in [1024, 2048, 4096]
                for have_attn_sink in [False, True]
            ]
            + [
                # Each query owns a disjoint 1280-token KV region.  The top-k
                # tensor stays at 2048 entries, so the tail remains invalid.
                TestParam(
                    s_q,
                    s_q * 1280,
                    2048,
                    h_q=4,
                    d_qk=512,
                    have_topk_length=True,
                    fixed_topk_length=1280,
                    kv_tokens_per_query=1280,
                    sm_scale=512**-0.5,
                )
                for s_q in [64, 96, 128]
            ]
            + [
                # Keep sink-enabled coverage as a secondary variant of the
                # same disjoint-KV workload.
                TestParam(
                    s_q,
                    s_q * 1280,
                    2048,
                    h_q=4,
                    d_qk=512,
                    have_attn_sink=True,
                    have_topk_length=True,
                    fixed_topk_length=1280,
                    kv_tokens_per_query=1280,
                    sm_scale=512**-0.5,
                )
                for s_q in [64, 96, 128]
            ]
            + [
                TestParam(4096, s_kv, 2048, h_q=128, d_qk=576, have_attn_sink=True)
                for s_kv in [8192, 32768, 65536, 98304, 131072]
            ]
            + [
                TestParam(4096, s_kv, 512, h_q=64, d_qk=512, have_attn_sink=True)
                for s_kv in [8192, 32768, 49152, 65536]
            ]
            + [
                TestParam(4096, s_kv, 1024, h_q=128, d_qk=512, have_attn_sink=True)
                for s_kv in [8192, 32768, 49152, 65536]
            ]
        )
        return cases

    @staticmethod
    def _randperm_batch(
        batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]
    ) -> torch.Tensor:
        """
        Generate random permutations in batch
        The return tensor, denoted as `res`, has a shape of [batch_size, perm_size].
        `0 <= res[i, :] < perm_range[i]` holds.
        Values within each row are unique.
        If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains
        values in `[0, perm_range[i])` as many as possible, and the rest are filled with `padding`.
        """

        assert not torch.are_deterministic_algorithms_enabled()

        torch.use_deterministic_algorithms(True)
        perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
        rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32)
        rand[
            torch.arange(0, perm_range_max).broadcast_to(batch_size, perm_range_max)
            >= perm_range.view(batch_size, 1)
        ] = float("-inf")
        res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
        if len(paddings) == 1:
            res[res >= perm_range.view(batch_size, 1)] = paddings[0]
        else:
            fillers = torch.tensor(paddings, dtype=torch.int32).index_select(
                0, torch.randint(0, len(paddings), (res.numel(),), dtype=torch.int32)
            )
            res.masked_scatter_(res >= perm_range.view(batch_size, 1), fillers)
        torch.use_deterministic_algorithms(False)
        return res

    @staticmethod
    def _make_pair_work_item_inputs(
        s_q: int,
        s_kv: int,
        combined_topk: int,
        queries_per_kv_region: int,
        kv_tokens_per_region: int,
        compressed_pool_size: int,
        kv_workspace_regions: int,
        compressed_topk: int,
        window_size: int,
    ):
        """Build C4 + SWA inputs and packed metadata emitted by the producer.

        Metadata uses the low three bits for the pair mode and stores the
        first row's compressed-prefix length in the remaining bits:

        1. unchanged compression, growing SWA;
        2. unchanged compression, sliding SWA;
        3. compression growth, growing SWA;
        4. compression growth, sliding SWA.
        """
        assert s_q > 0 and combined_topk > 0
        assert queries_per_kv_region > 0
        assert queries_per_kv_region % 2 == 0
        assert kv_tokens_per_region > 0
        assert compressed_pool_size > 0
        assert kv_workspace_regions > 0
        assert compressed_topk > 0 and window_size > 0
        assert combined_topk == compressed_topk + window_size

        num_regions = math.ceil(s_q / queries_per_kv_region)
        assert num_regions <= kv_workspace_regions
        assert s_kv == kv_workspace_regions * kv_tokens_per_region
        max_compressed_length = min(queries_per_kv_region // 4, compressed_topk)
        assert max_compressed_length <= compressed_pool_size
        assert compressed_topk <= compressed_pool_size
        assert (
            compressed_pool_size + window_size + queries_per_kv_region
            <= kv_tokens_per_region
        )

        rows = torch.arange(s_q, dtype=torch.int64)
        region = rows // queries_per_kv_region
        position = rows % queries_per_kv_region
        compressed_length = torch.minimum(
            (position + 1) // 4,
            torch.tensor(compressed_topk, dtype=torch.int64),
        )
        swa_length = torch.minimum(
            position + 1,
            torch.tensor(window_size, dtype=torch.int64),
        )
        topk_lengths = (compressed_length + swa_length).to(torch.int32)

        columns = torch.arange(combined_topk, dtype=torch.int64)[None, :]
        region_base = (region * kv_tokens_per_region)[:, None]
        compressed_indices = region_base + columns
        swa_column = columns - compressed_length[:, None]
        swa_start = position - swa_length + 1
        swa_indices = (
            region_base + compressed_pool_size + swa_start[:, None] + swa_column
        )
        indices = torch.where(
            columns < compressed_length[:, None],
            compressed_indices,
            torch.where(
                columns < topk_lengths.to(torch.int64)[:, None],
                swa_indices,
                -1,
            ),
        ).to(torch.int32)

        first_rows = torch.arange(0, s_q, 2, dtype=torch.int64)
        first_position = first_rows % queries_per_kv_region
        first_compressed = torch.minimum(
            (first_position + 1) // 4,
            torch.tensor(compressed_topk, dtype=torch.int64),
        )
        second_compressed = torch.minimum(
            (first_position + 2) // 4,
            torch.tensor(compressed_topk, dtype=torch.int64),
        )
        first_swa = torch.minimum(
            first_position + 1,
            torch.tensor(window_size, dtype=torch.int64),
        )
        same_compression = first_compressed == second_compressed
        growing_swa = first_swa < window_size
        pair_mode = torch.where(
            same_compression,
            torch.where(growing_swa, 1, 2),
            torch.where(growing_swa, 3, 4),
        )

        has_second = first_rows + 1 < s_q
        same_region = (
            first_rows // queries_per_kv_region
            == (first_rows + 1) // queries_per_kv_region
        )
        pair_mode = torch.where(has_second & same_region, pair_mode, 0)
        pair_metadata = (pair_mode | (first_compressed << 3)).to(torch.int32)
        return indices, topk_lengths, pair_metadata

    @staticmethod
    def make_input_flashmla(param: TestParam):
        """Create input data for sparse MLA operator by referring to the FlashMLA examples"""
        s_q = param.s_q
        s_kv = param.s_kv
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        topk = param.topk
        kv_tokens_per_query = param.kv_tokens_per_query
        queries_per_kv_region = param.queries_per_kv_region
        kv_tokens_per_region = param.kv_tokens_per_region
        compressed_pool_size = param.compressed_pool_size
        kv_workspace_regions = param.kv_workspace_regions
        use_pair_work_items = param.use_pair_work_items
        pair_window_size = param.pair_window_size
        compressed_topk = param.compressed_topk
        have_attn_sink = param.have_attn_sink
        have_topk_length = param.have_topk_length
        is_all_indices_invalid = param.is_all_indices_invalid
        dtype = param.dtype
        device = param.device

        global _flashmla_sparse_counter
        FlashmlaSparseBenchmark._init_seed(_flashmla_sparse_counter)
        _flashmla_sparse_counter = _flashmla_sparse_counter + 1

        q = (
            torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        kv = (
            torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        q = q.clamp_(-10, 10)
        kv = kv.clamp_(-10, 10)
        invalid_indices_candidate = [
            -2147483648,
            -123456,
            -1,
            s_kv,
            114514,
            1919810,
            2147480000,
            2147483647,
        ]

        generated_topk_length = None
        pair_metadata = None
        if use_pair_work_items:
            assert kv_tokens_per_query is None
            assert queries_per_kv_region is not None
            assert kv_tokens_per_region is not None
            assert compressed_pool_size > 0
            assert kv_workspace_regions > 0
            assert pair_window_size > 0
            assert compressed_topk > 0
            indices, generated_topk_length, pair_metadata = (
                FlashmlaSparseBenchmark._make_pair_work_item_inputs(
                    s_q,
                    s_kv,
                    topk,
                    queries_per_kv_region,
                    kv_tokens_per_region,
                    compressed_pool_size,
                    kv_workspace_regions,
                    compressed_topk,
                    pair_window_size,
                )
            )
        elif kv_tokens_per_query is not None:
            assert kv_tokens_per_query > 0
            assert s_kv == s_q * kv_tokens_per_query
            perm_range = torch.full((s_q,), kv_tokens_per_query, dtype=torch.int32)
            invalid_indices_candidate = [
                index
                for index in invalid_indices_candidate
                if index < 0 or index >= s_kv
            ]
        else:
            perm_range = torch.full((s_q,), s_kv, dtype=torch.int32)

        if not use_pair_work_items:
            indices = FlashmlaSparseBenchmark._randperm_batch(
                s_q,
                perm_range,
                topk,
                invalid_indices_candidate,
            )
            if kv_tokens_per_query is not None:
                valid_indices = (indices >= 0) & (indices < kv_tokens_per_query)
                row_offsets = (
                    torch.arange(s_q, dtype=torch.int32) * kv_tokens_per_query
                ).view(s_q, 1)
                indices = torch.where(valid_indices, indices + row_offsets, indices)
        indices = indices.view(s_q, h_kv, topk)
        if is_all_indices_invalid:
            all_indices_invalid_mask = torch.randn(s_q, device="cpu") < -2
            indices[
                all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)
            ] = random.choice(invalid_indices_candidate)
        indices = indices.to(device)
        if pair_metadata is not None:
            pair_metadata = pair_metadata.to(device)

        attn_sink = None
        if have_attn_sink:
            attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
            mask = torch.randn((h_q,), dtype=torch.float32, device=device)
            attn_sink[mask < -0.5] = float("-inf")
            attn_sink[mask > +0.5] = float("+inf")

        topk_length = None
        if have_topk_length:
            if generated_topk_length is not None:
                topk_length = generated_topk_length.to(device)
            elif param.fixed_topk_length is not None:
                assert 0 <= param.fixed_topk_length <= topk
                topk_length = torch.full(
                    (s_q,),
                    param.fixed_topk_length,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                topk_length = torch.randint(
                    0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
                ).clamp_max(topk)

        kwargs = {}
        if use_pair_work_items:
            # The production caller consumes only the output, and the pair
            # work-item specialization intentionally has no statistics
            # path.  The baseline shim drops metadata while Gems consumes it.
            kwargs["pair_metadata"] = pair_metadata
            kwargs["pair_window_size"] = pair_window_size
            kwargs["return_stats"] = False
        yield (
            q,
            kv,
            indices,
            param.sm_scale,
            param.d_v,
            attn_sink,
            topk_length,
            kwargs,
        )


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not HAS_VLLM_FLASHMLA_SPARSE, reason="vLLM not installed")
def test_flash_mla_sparse_fwd():
    bench = FlashmlaSparseBenchmark()
    bench.run()
