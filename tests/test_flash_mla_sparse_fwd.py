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
import random
from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE

random.seed(42)

try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd as vllm_flash_mla_sparse_fwd,
    )

    HAS_VLLM_FLASHMLA_SPARSE = True
except ImportError:
    HAS_VLLM_FLASHMLA_SPARSE = False
    print(
        "vLLM not installed, the native pytorch implementation of FlashMLA for comparison"
    )
    torch.set_float32_matmul_precision("high")


@dataclasses.dataclass
class Flashmla_Sparse_Test_Param:
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


class FlashmlaSparseTestKit:
    # used by torch vertion flashmla_sparse
    @staticmethod
    def _merge_two_lse(
        lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int
    ) -> torch.Tensor:
        if lse1 is None:
            return lse0

        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )

    # torch version flashmla_sparse
    @staticmethod
    def torch_flash_mla_sparse_fwd(
        s_q: int,
        s_kv: int,
        h_q: int,
        h_kv: int,
        d_qk: int,
        topk: int,
        q: torch.Tensor,  # [s_q, h_q, d_qk]
        kv: torch.Tensor,  # [s_q, 1, d_qk]
        indices: torch.Tensor,  # [s_q, 1, topk]
        sm_scale: float,
        d_v: int,
        attn_sink: Optional[torch.Tensor],  # [h_q]
        topk_length: Optional[torch.Tensor],  # [s_q]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - o: [s_q, h_q, dv]
        - o_fp32: [s_q, h_q, dv]
        - max_logits: [s_q, h_q]
        - lse: [s_q, h_q]
        """
        indices = indices.clone().squeeze(1)
        if topk_length is not None:
            mask = torch.arange(topk, device=topk_length.device).unsqueeze(
                0
            ).broadcast_to(s_q, topk) >= topk_length.unsqueeze(1)
            indices[mask] = -1
        invalid_mask = (indices < 0) | (indices >= s_kv)
        indices[invalid_mask] = 0
        q = q.float()
        gathered_kv = (
            kv.index_select(dim=0, index=indices.flatten())
            .reshape(s_q, topk, d_qk)
            .float()
        )
        P = q @ gathered_kv.transpose(1, 2)
        P *= sm_scale
        P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

        orig_lse = torch.logsumexp(P, dim=-1)
        max_logits = P.max(dim=-1).values

        lse_for_o = FlashmlaSparseTestKit._merge_two_lse(orig_lse, attn_sink, s_q, h_q)
        if not torch.is_inference_mode_enabled():
            lse_for_o = lse_for_o.clone()
        lse_for_o[lse_for_o == float("-inf")] = float(
            "+inf"
        )  # So that corresponding O will be 0
        s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
        out = s_for_o @ gathered_kv[..., :d_v]

        lonely_q_mask = orig_lse == float("-inf")
        orig_lse[lonely_q_mask] = float("+inf")
        return (out.to(torch.bfloat16), max_logits, orig_lse)

    @staticmethod
    def get_correctness_test_params():
        if QUICK_MODE:
            cases = [Flashmla_Sparse_Test_Param(64, 1024, 128, 128, 1, 576, 512)]
        else:
            cases = [
                Flashmla_Sparse_Test_Param(s_q, s_kv, topk, h_q, h_kv, d_qk, d_v)
                for s_q in [64, 128, 512]
                for s_kv in [1024, 2048, 4096]
                for h_q in [64, 128, 256]
                for h_kv in [1]
                for d_qk in [576]
                for d_v in [512]
                for topk in [64, 128, 256]
            ]
        return cases

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def make_input(param: Flashmla_Sparse_Test_Param):
        """Create input data for sparse MLA operator"""
        S = param.s_q
        H = param.h_q
        DQK = param.d_qk
        SKV = param.s_kv
        HKV = param.h_kv
        topk = param.topk
        dtype = param.dtype
        device = param.device
        requires_grad = False

        FlashmlaSparseTestKit._init_seed(42)

        q = torch.randn((S, H, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )
        kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )

        indices = torch.full((S, HKV, topk), SKV, dtype=torch.int32, device=device)
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[t, h, : len(i_i)] = i_i

        return q, kv, indices

    @staticmethod
    def get_correctness_test_params_flashmla():
        if QUICK_MODE:
            cases = [
                Flashmla_Sparse_Test_Param(
                    s_q=62,
                    s_kv=592,
                    topk=128,
                    h_q=128,
                    d_qk=512,
                    have_attn_sink=True,
                    have_topk_length=False,
                )
            ]
        else:
            cases = [
                Flashmla_Sparse_Test_Param(
                    s_q,
                    s_kv,
                    topk,
                    h_q,
                    d_qk=d_qk,
                    have_attn_sink=have_attn_sink,
                    have_topk_length=have_topk_length,
                )
                for s_q in [1, 62, 213]
                for h_q in [128, 64]
                for d_qk in [512, 576]
                for s_kv, topk in [
                    (592, 128),
                    (1840, 256),
                    (1592, 384),
                    (1521, 512),
                    (95, 128),
                    (153, 256),
                    (114, 384),
                ]
                for have_attn_sink in [True, False]
                for have_topk_length in [True, False]
            ]
        return cases

    @staticmethod
    def _randperm_batch(
        batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]
    ) -> torch.Tensor:
        """
        Generate random permutations in batch
        The return tensor, denoted as `res`, has a shape of [batch_size, perm_size]. `0 <= res[i, :] < perm_range[i]`
        holds.
        Values within each row are unique.
        If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains values in `[0, perm_range[i])`
        as many as possible, and the rest are filled with `padding`.
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
    def make_input_flashmla(param: Flashmla_Sparse_Test_Param):
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
        FlashmlaSparseTestKit._init_seed(_flashmla_sparse_counter)
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
        indices = FlashmlaSparseTestKit._randperm_batch(
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
        return q, kv, indices, attn_sink, topk_length


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize("param", FlashmlaSparseTestKit.get_correctness_test_params())
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flashmla_sparse(param):
    """Sparse MLA forward propagation test"""
    # Skip FlashMLA unsupported cases
    if param.h_q != 64 and param.h_q != 128:
        # RuntimeError: Unsupported h_q: 256
        # FlashMLA csrc/api/sparse_fwd.h:197
        # FlashMLA requires that h_q is 64 or 128
        return

    if param.topk % 128 != 0:
        # Assertion `params.topk % (2*B_TOPK) == 0` failed
        # FlashMLA csrc/sm90/prefill/sparse/phase1.cuh:577
        # FlashMLA csrc/sm90/prefill/sparse/config.h:27 "B_TOPK = 64"
        # topk not divisible by 128, not supported by FlashMLA
        return

    # Create input
    q, kv, indices = FlashmlaSparseTestKit.make_input(param)
    sm_scale = param.d_qk**-0.5

    if HAS_VLLM_FLASHMLA_SPARSE:
        ref_output, ref_max_logbits, ref_lse = vllm_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, param.d_v
        )
    else:
        (
            ref_output,
            ref_max_logbits,
            ref_lse,
        ) = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            param.s_q,
            param.s_kv,
            param.h_q,
            param.h_kv,
            param.d_qk,
            param.topk,
            q,
            kv,
            indices,
            sm_scale,
            param.d_v,
            None,
            None,
        )

    # Your operator implementation
    your_output, your_max_logbits, your_lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        param.d_v,
    )

    # Accuracy comparison
    flag_gems.testing.assert_close(your_output, ref_output, param.dtype, atol=1e-2)
    flag_gems.testing.assert_close(
        your_max_logbits, ref_max_logbits, torch.float32, atol=1e-4
    )
    flag_gems.testing.assert_close(your_lse, ref_lse, torch.float32, atol=1e-4)


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("have_attn_sink", [False, True])
@pytest.mark.parametrize("have_topk_length", [False, True])
@pytest.mark.parametrize("d_qk", [512, 576])
def test_flash_mla_sparse_hq4_torch_reference_and_strided_out(
    have_attn_sink, have_topk_length, d_qk
):
    """Cover the unpadded four-head path without relying on native FlashMLA."""
    s_q, s_kv, topk = 4, 19, 32
    h_q, h_kv, d_v = 4, 1, 512
    dtype = torch.bfloat16
    device = flag_gems.device

    FlashmlaSparseTestKit._init_seed(2026)
    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10

    indices = torch.arange(topk, dtype=torch.int32).remainder(s_kv).repeat(s_q, 1)
    # Both kinds of invalid indices must be ignored inside the effective top-k range.
    indices[0, 1] = -1
    indices[0, 3] = -123456
    indices[0, 5] = s_kv
    indices[0, 7] = s_kv + 114514
    # Entries after topk_length are valid on purpose: using one would change the result.
    indices[1, 7:] = torch.arange(topk - 7, dtype=torch.int32).remainder(s_kv)
    # The final query has no valid index at all.
    indices[3, 0::2] = -7
    indices[3, 1::2] = s_kv + 3
    indices = indices.view(s_q, h_kv, topk).to(device)

    topk_length = None
    if have_topk_length:
        topk_length = torch.tensor([topk, 7, 0, topk], dtype=torch.int32, device=device)
    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.tensor(
            [0.25, float("-inf"), float("+inf"), -0.75],
            dtype=torch.float32,
            device=device,
        )
    sm_scale = d_qk**-0.5

    ref_output, ref_max_logits, ref_lse = (
        FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            s_q,
            s_kv,
            h_q,
            h_kv,
            d_qk,
            topk,
            q,
            kv,
            indices,
            sm_scale,
            d_v,
            attn_sink,
            topk_length,
        )
    )

    sentinel = torch.tensor(17.0, dtype=dtype, device=device)
    output_storage = torch.full(
        (s_q, 64, d_v), sentinel.item(), dtype=dtype, device=device
    )
    out = output_storage[:, :h_q, :]
    untouched_heads = output_storage[:, h_q:, :].clone()
    assert not out.is_contiguous()
    assert out.stride(-1) == 1

    output, max_logits, lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        out=out,
    )

    assert output is out
    assert output.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        output, ref_output, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    torch.testing.assert_close(
        max_logits,
        ref_max_logits,
        atol=1e-6,
        rtol=2.01 / 65536,
        equal_nan=False,
    )
    torch.testing.assert_close(
        lse, ref_lse, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )

    # A +inf sink and queries with no effective valid index produce zero output.
    if have_attn_sink:
        assert torch.count_nonzero(output[:, 2, :]).item() == 0
    invalid_rows = output[2:] if have_topk_length else output[3:]
    invalid_max_logits = max_logits[2:] if have_topk_length else max_logits[3:]
    invalid_lse = lse[2:] if have_topk_length else lse[3:]
    assert torch.count_nonzero(invalid_rows).item() == 0
    assert torch.isneginf(invalid_max_logits).all()
    assert torch.isposinf(invalid_lse).all()
    # The four-head view must not allow a padded-head kernel to write the backing tail.
    assert torch.equal(output_storage[:, h_q:, :], untouched_heads)

    no_stats_storage = torch.full(
        (s_q, 64, d_v), sentinel.item(), dtype=dtype, device=device
    )
    no_stats_out = no_stats_storage[:, :h_q, :]
    no_stats_output, no_stats_max_logits, no_stats_lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        out=no_stats_out,
        return_stats=False,
    )
    assert no_stats_output is no_stats_out
    assert no_stats_max_logits is None
    assert no_stats_lse is None
    torch.testing.assert_close(no_stats_output, output, atol=0, rtol=0)
    assert torch.equal(no_stats_storage[:, h_q:, :], untouched_heads)


def _make_pair_metadata_case():
    """Build all four pair modes, a fallback pair, and an odd tail."""
    s_q, s_kv, topk = 11, 64, 32
    h_q, h_kv, d_qk = 4, 1, 512
    dtype = torch.bfloat16
    device = flag_gems.device
    pair_window_size = 4

    FlashmlaSparseTestKit._init_seed(3901)
    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10

    # Each list is the complete effective index row. Modes 2 and 4 use a
    # shifted four-token window; modes 3 and 4 insert one new top-k token.
    active_rows = [
        [0, -1, s_kv + 7, 20, 21],
        [0, -1, s_kv + 7, 20, 21, 22],  # mode 1
        [3, 4, 5, 30, 31, 32, 33],
        [3, 4, 5, 31, 32, 33, 34],  # mode 2
        [6, 7, 40, 41],
        [6, 7, 8, 40, 41, 42],  # mode 3
        [9, 10, 50, 51, 52, 53],
        [9, 10, 11, 51, 52, 53, 54],  # mode 4
        [-2147483648, 2147483647, -1, s_kv],
        [2, 4, 6, 8, 10],  # descriptor mode 0: both rows use fallback
        [-2147483648, -7, s_kv, s_kv + 3],  # odd, all-invalid tail
    ]
    topk_length = torch.tensor(
        [len(row) for row in active_rows], dtype=torch.int32, device=device
    )
    indices = torch.empty((s_q, h_kv, topk), dtype=torch.int32)
    poison = torch.arange(topk, dtype=torch.int32)
    for row_idx, active in enumerate(active_rows):
        indices[row_idx, 0] = (poison * 17 + row_idx * 19 + 13).remainder(s_kv)
        indices[row_idx, 0, : len(active)] = torch.tensor(active, dtype=torch.int32)

    pair_metadata = torch.tensor(
        [
            (3 << 3) | 1,
            (3 << 3) | 2,
            (2 << 3) | 3,
            (2 << 3) | 4,
            0,
            0,
        ],
        dtype=torch.int32,
        device=device,
    )
    return (
        q,
        kv,
        indices.to(device),
        topk_length,
        pair_metadata,
        pair_window_size,
    )


def _pair_metadata_reference(q, kv, indices, topk_length, attn_sink):
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    return FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        indices.shape[-1],
        q,
        kv,
        indices,
        d_qk**-0.5,
        512,
        attn_sink,
        topk_length,
    )


def _pair_metadata_sink(have_attn_sink):
    if not have_attn_sink:
        return None
    return torch.tensor(
        [0.25, float("-inf"), float("+inf"), -0.75],
        dtype=torch.float32,
        device=flag_gems.device,
    )


def _make_host_validation_inputs():
    q = torch.zeros((2, 4, 512), dtype=torch.bfloat16)
    kv = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
    indices = torch.zeros((2, 1, 1), dtype=torch.int32)
    topk_length = torch.ones(2, dtype=torch.int32)
    attn_sink = torch.zeros(4, dtype=torch.float32)
    return q, kv, indices, topk_length, attn_sink


@pytest.mark.flash_mla_sparse_fwd
def test_flash_mla_sparse_rejects_empty_kv_before_kernel_launch():
    q, _, indices, topk_length, _ = _make_host_validation_inputs()
    empty_kv = torch.empty((0, 1, 512), dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="kv must contain at least one row"):
        flag_gems.flash_mla_sparse_fwd(
            q,
            empty_kv,
            indices,
            q.shape[-1] ** -0.5,
            topk_length=topk_length,
            return_stats=False,
        )


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize("tensor_name", ["kv", "indices", "topk_length", "attn_sink"])
def test_flash_mla_sparse_rejects_mixed_devices_before_kernel_launch(tensor_name):
    q, kv, indices, topk_length, attn_sink = _make_host_validation_inputs()
    inputs = {
        "kv": kv,
        "indices": indices,
        "topk_length": topk_length,
        "attn_sink": attn_sink,
    }
    inputs[tensor_name] = torch.empty_like(inputs[tensor_name], device="meta")

    with pytest.raises(AssertionError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            inputs["kv"],
            inputs["indices"],
            q.shape[-1] ** -0.5,
            attn_sink=inputs["attn_sink"],
            topk_length=inputs["topk_length"],
            return_stats=False,
        )


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("h_q", [4, 64])
def test_flash_mla_sparse_clamps_topk_length_to_row_bounds(h_q):
    """Malformed lengths must not make either Triton path read another row."""

    device = flag_gems.device
    q = torch.randn((2, h_q, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((64, 1, 512), dtype=torch.bfloat16, device=device)
    indices = torch.arange(64, dtype=torch.int32, device=device).view(1, 1, 64)
    indices = indices.expand(2, -1, -1).contiguous()
    raw_lengths = torch.tensor([-7, 71], dtype=torch.int32, device=device)
    clamped_lengths = raw_lengths.clamp(0, indices.shape[-1])

    output, _, _ = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        512**-0.5,
        topk_length=raw_lengths,
        return_stats=False,
    )
    reference, _, _ = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        512**-0.5,
        topk_length=clamped_lengths,
        return_stats=False,
    )

    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("have_attn_sink", [False, True])
def test_flash_mla_sparse_hq4_pair_metadata_modes_and_fallback(have_attn_sink):
    """Exercise modes 1-4, mode 0, invalid IDs, and the odd-tail path."""
    q, kv, indices, topk_length, pair_metadata, window = _make_pair_metadata_case()
    attn_sink = _pair_metadata_sink(have_attn_sink)
    sm_scale = q.shape[-1] ** -0.5
    reference, _, _ = _pair_metadata_reference(q, kv, indices, topk_length, attn_sink)
    baseline, baseline_max_logits, baseline_lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        attn_sink,
        topk_length,
        return_stats=False,
    )
    assert baseline_max_logits is None and baseline_lse is None

    output_storage = torch.full(
        (q.shape[0], 9, 512), 17.0, dtype=q.dtype, device=q.device
    )
    out = output_storage[:, : q.shape[1], :]
    untouched_tail = output_storage[:, q.shape[1] :, :].clone()
    assert not out.is_contiguous()
    output, max_logits, lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        attn_sink,
        topk_length,
        out=out,
        return_stats=False,
        pair_metadata=pair_metadata,
        pair_window_size=window,
    )

    assert output is out
    assert output.data_ptr() == out.data_ptr()
    assert max_logits is None and lse is None
    torch.testing.assert_close(
        output, baseline, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    assert torch.equal(output_storage[:, q.shape[1] :, :], untouched_tail)
    assert torch.count_nonzero(output[[8, 10]]).item() == 0

    if have_attn_sink:
        # -inf is a no-op, while +inf forces the corresponding head to zero.
        no_sink_output, _, _ = flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            512,
            topk_length=topk_length,
            return_stats=False,
        )
        torch.testing.assert_close(
            output[:, 1],
            no_sink_output[:, 1],
            atol=8e-4,
            rtol=3.01 / 128,
            equal_nan=False,
        )
        assert torch.count_nonzero(output[:, 2]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flash_mla_sparse_pair_metadata_cuda_graph_replay():
    """Replay reads updated q values and writes the caller-owned output."""
    q, kv, indices, topk_length, pair_metadata, window = _make_pair_metadata_case()
    sm_scale = q.shape[-1] ** -0.5
    graph_out = torch.empty_like(q)
    warmup_out = torch.empty_like(q)

    capture_stream = torch.cuda.Stream(device=q.device)
    capture_stream.wait_stream(torch.cuda.current_stream(q.device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            warmup_output, warmup_max_logits, warmup_lse = (
                flag_gems.flash_mla_sparse_fwd(
                    q,
                    kv,
                    indices,
                    sm_scale,
                    512,
                    topk_length=topk_length,
                    out=warmup_out,
                    return_stats=False,
                    pair_metadata=pair_metadata,
                    pair_window_size=window,
                )
            )
            assert warmup_output is warmup_out
            assert warmup_max_logits is None and warmup_lse is None
    torch.cuda.current_stream(q.device).wait_stream(capture_stream)
    torch.cuda.synchronize(q.device)
    original_eager = warmup_out.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_max_logits, captured_lse = (
            flag_gems.flash_mla_sparse_fwd(
                q,
                kv,
                indices,
                sm_scale,
                512,
                topk_length=topk_length,
                out=graph_out,
                return_stats=False,
                pair_metadata=pair_metadata,
                pair_window_size=window,
            )
        )
    assert captured_output is graph_out
    assert captured_max_logits is None and captured_lse is None

    graph.replay()
    torch.cuda.synchronize(q.device)
    original_replay = graph_out.clone()
    torch.testing.assert_close(original_replay, original_eager, atol=0, rtol=0)

    original_q = q.clone()
    q.copy_(original_q * -1.75 + 0.03125)
    torch.cuda.synchronize(q.device)
    graph.replay()
    torch.cuda.synchronize(q.device)
    updated_replay = graph_out.clone()

    eager_out = torch.empty_like(graph_out)
    eager_output, eager_max_logits, eager_lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        topk_length=topk_length,
        out=eager_out,
        return_stats=False,
        pair_metadata=pair_metadata,
        pair_window_size=window,
    )
    assert eager_output is eager_out
    assert eager_max_logits is None and eager_lse is None
    torch.testing.assert_close(updated_replay, eager_output, atol=0, rtol=0)
    assert not torch.equal(updated_replay[:8], original_replay[:8])


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flash_mla_sparse_hq4_mode4_virtual_union_exceeds_row_stride():
    """Mode 4 must map its 2177th union item back into a 2176-wide row."""
    s_q, s_kv, physical_width = 2, 2304, 2176
    h_q, h_kv, d_qk = 4, 1, 512
    window = 128
    first_topk_length = 2047
    device = flag_gems.device

    FlashmlaSparseTestKit._init_seed(3903)
    q = torch.randn((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=torch.bfloat16, device=device) / 10
    indices = torch.full(
        (s_q, h_kv, physical_width), -1, dtype=torch.int32, device=device
    )
    prefix = torch.arange(first_topk_length, dtype=torch.int32, device=device)
    indices[0, 0, :first_topk_length] = prefix
    indices[1, 0, :first_topk_length] = prefix
    indices[1, 0, first_topk_length] = first_topk_length
    indices[0, 0, first_topk_length:2175] = torch.arange(
        2048, 2176, dtype=torch.int32, device=device
    )
    indices[1, 0, 2048:physical_width] = torch.arange(
        2049, 2177, dtype=torch.int32, device=device
    )
    topk_length = torch.tensor([2175, 2176], dtype=torch.int32, device=device)
    pair_metadata = torch.tensor(
        [(first_topk_length << 3) | 4], dtype=torch.int32, device=device
    )
    sm_scale = d_qk**-0.5

    assert indices.is_contiguous()
    assert indices.stride(0) == physical_width
    assert int(topk_length[1].item()) + 1 == 2177
    reference, _, _ = _pair_metadata_reference(q, kv, indices, topk_length, None)
    baseline, _, _ = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        topk_length=topk_length,
        return_stats=False,
    )
    output, max_logits, lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        topk_length=topk_length,
        return_stats=False,
        pair_metadata=pair_metadata,
        pair_window_size=window,
    )

    assert max_logits is None and lse is None
    torch.testing.assert_close(
        output, baseline, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flash_mla_sparse_hq4_damaged_pair_descriptors_fall_back():
    """Structurally inconsistent descriptors must execute two single rows."""
    q, kv, indices, topk_length, pair_metadata, window = _make_pair_metadata_case()
    sm_scale = q.shape[-1] ** -0.5
    reference, _, _ = _pair_metadata_reference(q, kv, indices, topk_length, None)
    baseline, _, _ = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        topk_length=topk_length,
        return_stats=False,
    )

    damaged = pair_metadata.clone()
    damaged[0] = (5 << 3) | 1  # mode 1 claims an empty window
    damaged[1] = (2 << 3) | 2  # mode 2 claims a five-token window
    damaged[2] = (2 << 3) | 1  # mode 1 cannot grow by two total entries
    damaged[3] = (3 << 3) | 4  # mode 4 claims a three-token window
    damaged[4] = 7  # unsupported mode
    damaged[5] = 1  # valid-looking mode, but the pair has no second row

    output, max_logits, lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        topk_length=topk_length,
        return_stats=False,
        pair_metadata=damaged,
        pair_window_size=window,
    )

    assert max_logits is None and lse is None
    torch.testing.assert_close(
        output, baseline, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flash_mla_sparse_pair_metadata_stats_bypass_and_default():
    """The default and statistics paths retain the regular implementation."""
    q, kv, indices, topk_length, pair_metadata, window = _make_pair_metadata_case()
    attn_sink = _pair_metadata_sink(True)
    sm_scale = q.shape[-1] ** -0.5

    default_output, default_max_logits, default_lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        attn_sink,
        topk_length,
    )
    output, max_logits, lse = flag_gems.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        512,
        attn_sink,
        topk_length,
        pair_metadata=pair_metadata,
        pair_window_size=window,
    )

    assert max_logits is not None and lse is not None
    assert torch.equal(output, default_output)
    assert torch.equal(max_logits, default_max_logits)
    assert torch.equal(lse, default_lse)

    reference, ref_max_logits, ref_lse = _pair_metadata_reference(
        q, kv, indices, topk_length, attn_sink
    )
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    torch.testing.assert_close(
        max_logits,
        ref_max_logits,
        atol=1e-6,
        rtol=2.01 / 65536,
        equal_nan=False,
    )
    torch.testing.assert_close(
        lse, ref_lse, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flash_mla_sparse_pair_metadata_parameter_validation():
    q, kv, indices, topk_length, pair_metadata, window = _make_pair_metadata_case()
    sm_scale = q.shape[-1] ** -0.5

    for bad_value in (None, 0, 1, "false"):
        with pytest.raises(TypeError):
            flag_gems.flash_mla_sparse_fwd(
                q,
                kv,
                indices,
                sm_scale,
                topk_length=topk_length,
                return_stats=bad_value,
            )
    for bad_value in (None, True, 4.0, "4"):
        with pytest.raises(TypeError):
            flag_gems.flash_mla_sparse_fwd(
                q,
                kv,
                indices,
                sm_scale,
                topk_length=topk_length,
                pair_metadata=pair_metadata,
                pair_window_size=bad_value,
            )
    with pytest.raises(TypeError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            topk_length=topk_length,
            pair_metadata="not a tensor",
            pair_window_size=window,
        )
    with pytest.raises(ValueError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            topk_length=topk_length,
            pair_window_size=window,
        )
    for bad_window in (0, -1):
        with pytest.raises(ValueError):
            flag_gems.flash_mla_sparse_fwd(
                q,
                kv,
                indices,
                sm_scale,
                topk_length=topk_length,
                pair_metadata=pair_metadata,
                pair_window_size=bad_window,
            )
    with pytest.raises(ValueError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            pair_metadata=pair_metadata,
            pair_window_size=window,
        )

    wrong_shape = pair_metadata[:-1]
    with pytest.raises(AssertionError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            topk_length=topk_length,
            pair_metadata=wrong_shape,
            pair_window_size=window,
        )
    wrong_dtype = pair_metadata.to(torch.int64)
    with pytest.raises(AssertionError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            topk_length=topk_length,
            pair_metadata=wrong_dtype,
            pair_window_size=window,
        )
    noncontiguous = torch.empty(
        (pair_metadata.numel(), 2), dtype=torch.int32, device=q.device
    )[:, 0]
    assert not noncontiguous.is_contiguous()
    with pytest.raises(AssertionError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            topk_length=topk_length,
            pair_metadata=noncontiguous,
            pair_window_size=window,
        )
    if q.device.type != "cpu":
        wrong_device = pair_metadata.cpu()
        with pytest.raises(AssertionError):
            flag_gems.flash_mla_sparse_fwd(
                q,
                kv,
                indices,
                sm_scale,
                topk_length=topk_length,
                pair_metadata=wrong_device,
                pair_window_size=window,
            )


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3833: Precision & Compile Error.",
)
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flash_mla_sparse_pair_metadata_is_keyword_only():
    q, kv, indices, topk_length, pair_metadata, window = _make_pair_metadata_case()
    with pytest.raises(TypeError):
        flag_gems.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            q.shape[-1] ** -0.5,
            512,
            None,
            topk_length,
            None,
            False,
            pair_metadata,
            window,
        )


@pytest.mark.skip(reason="Issue #3691: operator not working")
@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize(
    "param", FlashmlaSparseTestKit.get_correctness_test_params_flashmla()
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_flash_mla_sparse_flashmla(param: Flashmla_Sparse_Test_Param):
    """Sparse MLA forward propagation test from FlashMLA"""
    # Create input
    q, kv, indices, attn_sink, topk_length = FlashmlaSparseTestKit.make_input_flashmla(
        param
    )
    sm_scale = 0.5

    if HAS_VLLM_FLASHMLA_SPARSE:
        ref_output, ref_max_logbits, ref_lse = vllm_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
        )
    else:
        (
            ref_output,
            ref_max_logbits,
            ref_lse,
        ) = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            param.s_q,
            param.s_kv,
            param.h_q,
            param.h_kv,
            param.d_qk,
            param.topk,
            q,
            kv,
            indices,
            sm_scale,
            param.d_v,
            attn_sink,
            topk_length,
        )

    # Your operator implementation
    your_output, your_max_logbits, your_lse = flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
    )

    # Accuracy comparison
    torch.testing.assert_close(
        your_output, ref_output, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )  # cos_diff_tol=7e-6
    torch.testing.assert_close(
        your_max_logbits, ref_max_logbits, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )
    torch.testing.assert_close(
        your_lse, ref_lse, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )
