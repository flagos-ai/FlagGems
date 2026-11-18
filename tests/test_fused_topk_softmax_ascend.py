import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from flag_gems.fused.topk_softmax import topk_softmax


if not torch_npu.npu.is_available():
    pytest.skip("Ascend NPU is required", allow_module_level=True)


DEVICE = "npu:0"
NUM_TOKENS = 64
NUM_EXPERTS = 256
TOPK = 8
REPEATS = 20


def reference(logits: torch.Tensor, renormalize: bool):
    probs = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(probs, TOPK, dim=-1)

    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    return weights, ids.to(torch.int32)


def run_topk_softmax(logits: torch.Tensor, renormalize: bool):
    num_tokens = logits.size(0)

    weights = torch.empty(
        num_tokens,
        TOPK,
        dtype=torch.float32,
        device=logits.device,
    )
    ids = torch.empty(
        num_tokens,
        TOPK,
        dtype=torch.int32,
        device=logits.device,
    )
    token_expert_indices = torch.empty(
        num_tokens,
        TOPK,
        dtype=torch.int32,
        device=logits.device,
    )

    topk_softmax(
        weights,
        ids,
        token_expert_indices,
        logits,
        renormalize,
    )
    torch.npu.synchronize()
    return weights, ids, token_expert_indices


@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax_ascend_bf16_is_accurate_and_deterministic(
    renormalize: bool,
):
    torch.npu.set_device(0)
    torch.manual_seed(20260731)

    logits = torch.randn(
        NUM_TOKENS,
        NUM_EXPERTS,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    ref_weights, ref_ids = reference(logits, renormalize)

    expected_slots = torch.arange(
        NUM_TOKENS * TOPK,
        dtype=torch.int32,
        device=DEVICE,
    ).view(TOPK, NUM_TOKENS).transpose(0, 1).contiguous()

    baseline_weights = None
    baseline_ids = None

    for _ in range(REPEATS):
        weights, ids, slots = run_topk_softmax(logits, renormalize)

        assert torch.isfinite(weights).all()
        assert torch.equal(
            torch.sort(ids, dim=-1).values,
            torch.sort(ref_ids, dim=-1).values,
        )
        assert torch.equal(slots, expected_slots)

        reference_in_id_order = torch.gather(
            torch.softmax(logits.float(), dim=-1),
            1,
            ids.long(),
        )
        if renormalize:
            reference_in_id_order = reference_in_id_order / (
                reference_in_id_order.sum(dim=-1, keepdim=True)
            )

        torch.testing.assert_close(
            weights.float(),
            reference_in_id_order,
            rtol=1e-5,
            atol=1e-6,
        )

        if renormalize:
            torch.testing.assert_close(
                weights.float().sum(dim=-1),
                torch.ones(NUM_TOKENS, device=DEVICE),
                rtol=1e-5,
                atol=1e-6,
            )

        if baseline_weights is not None:
            torch.testing.assert_close(
                weights.float(),
                baseline_weights.float(),
                rtol=0,
                atol=1e-6,
            )
            assert torch.equal(ids, baseline_ids)
        else:
            baseline_weights = weights.clone()
            baseline_ids = ids.clone()
