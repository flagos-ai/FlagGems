import pytest
import torch

import flag_gems
import flag_gems.fused as fused

from . import base, utils

vendor_name = flag_gems.vendor_name


class TopKSoftmaxBenchmark(base.Benchmark):
    """
    Benchmark for comparing topk_softmax between vLLM (CUDA kernel) and FlagGems (Triton kernel).
    """

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, topk)
        topk_softmax_configs = [
            (256, 128, 16),
            (1024, 256, 32),
            (4096, 64, 8),
            (8192, 128, 8),
            (16384, 256, 8),
        ]
        self.shapes = topk_softmax_configs

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from self.topk_softmax_input_fn(config, dtype, self.device)

    def topk_softmax_input_fn(self, config, dtype, device):
        """
        config: (num_tokens, num_experts, topk)
        """
        num_tokens, num_experts, k = config

        gating_output = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)

        for renormalize in (False, True):
            topk_weights = torch.empty(
                num_tokens, k, device=device, dtype=torch.float32
            )
            topk_indices = torch.empty(num_tokens, k, device=device, dtype=torch.int32)
            token_expert_indices = torch.empty(
                num_tokens, k, device=device, dtype=torch.int32
            )

            yield (
                topk_weights,
                topk_indices,
                token_expert_indices,
                gating_output,
                renormalize,
            )


@pytest.mark.topk_softmax
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM prior to 0.9 does not include the topk_softmax kernel in vllm._custom_ops.",
)
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "metax", reason="#2857: Not working")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2880: Not working")
@pytest.mark.skipif(vendor_name == "mthreads", reason="#2881: Not working")
@pytest.mark.skipif(vendor_name == "hygon", reason="#2882: RuntimeError")
@pytest.mark.skipif(vendor_name == "cambricon", reason="#2883: TypeError")
def test_topk_softmax(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    try:
        from vllm._custom_ops import topk_softmax as vllm_topk_softmax
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Skipped due to missing vLLM topk_softmax: {e}")

    bench = TopKSoftmaxBenchmark(
        op_name="topk_softmax",
        torch_op=vllm_topk_softmax,
        gems_op=fused.topk_softmax,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    bench.run()
