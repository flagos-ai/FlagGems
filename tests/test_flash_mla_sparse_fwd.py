import dataclasses
import random
from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems

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


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize("param", FlashmlaSparseTestKit.get_correctness_test_params())
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


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize(
    "param", FlashmlaSparseTestKit.get_correctness_test_params_flashmla()
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
