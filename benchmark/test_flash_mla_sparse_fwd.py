import dataclasses
import random
from typing import List

import pytest
import torch

import flag_gems

from . import performance_utils as base

try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd as vllm_flash_mla_sparse_fwd,
    )

    HAS_VLLM_FLASHMLA_SPARSE = True
except ImportError:
    HAS_VLLM_FLASHMLA_SPARSE = False


@dataclasses.dataclass
class TestParam:
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
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flag_gems.device


# used by make_input_flashmla
_flashmla_sparse_counter = 0


class FlashmlaSparseBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd", vllm_flash_mla_sparse_fwd, [torch.bfloat16]
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
    def make_input_flashmla(param: TestParam):
        """Create input data for sparse MLA operator by referring to the FlashMLA examples"""
        s_q = param.s_q
        s_kv = param.s_kv
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        topk = param.topk
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

        indices = FlashmlaSparseBenchmark._randperm_batch(
            s_q,
            torch.full((s_q,), s_kv, dtype=torch.int32),
            topk,
            invalid_indices_candidate,
        ).view(s_q, h_kv, topk)
        if is_all_indices_invalid:
            all_indices_invalid_mask = torch.randn(s_q, device="cpu") < -2
            indices[
                all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)
            ] = random.choice(invalid_indices_candidate)
        indices = indices.to(device)

        attn_sink = None
        if have_attn_sink:
            attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
            mask = torch.randn((h_q,), dtype=torch.float32, device=device)
            attn_sink[mask < -0.5] = float("-inf")
            attn_sink[mask > +0.5] = float("+inf")

        topk_length = None
        if have_topk_length:
            topk_length = torch.randint(
                0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
            ).clamp_max(topk)

        yield (q, kv, indices, 0.5, param.d_v, attn_sink, topk_length)


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not HAS_VLLM_FLASHMLA_SPARSE, reason="vLLM not installed")
def test_flash_mla_sparse_fwd():
    bench = FlashmlaSparseBenchmark()
    bench.run()
