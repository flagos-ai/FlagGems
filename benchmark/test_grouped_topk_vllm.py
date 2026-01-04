import pytest
import torch

import flag_gems

from .performance_utils import Benchmark, SkipVersion, vendor_name


class GroupedTopKBenchmark(Benchmark):
    def __init__(
        self,
        op_name,
        torch_op,
        dtypes,
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func=0,
    ):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func

    def set_shapes(self, shape_file_path=None):
        grouped_topk_configs = [
            (1, 64, 8, 2, 8),
            (8, 64, 8, 2, 8),
            (32, 64, 8, 2, 8),
            (64, 64, 8, 2, 8),
            (128, 64, 8, 2, 8),
            (256, 64, 8, 2, 8),
            (32, 128, 8, 2, 8),
            (64, 128, 8, 2, 8),
            (128, 128, 8, 2, 8),
            (64, 64, 4, 2, 4),
            (64, 128, 16, 2, 8),
            (512, 64, 8, 2, 8),
            (1024, 64, 8, 2, 8),
            (2048, 64, 8, 2, 8),
        ]
        self.shapes = grouped_topk_configs

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.grouped_topk_input_fn(config, cur_dtype, self.device)

    def grouped_topk_input_fn(self, config, dtype, device):
        num_tokens, num_experts, n_group, topk_group, topk = config

        scores = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
        bias = torch.randn(num_experts, device=device, dtype=dtype)

        yield (
            scores,
            n_group,
            topk_group,
            topk,
            self.renormalize,
            self.routed_scaling_factor,
            bias,
            self.scoring_func,
        )


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the grouped_topk kernel.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.grouped_topk
def test_perf_grouped_topk():
    try:
        from vllm._custom_ops import grouped_topk as vllm_grouped_topk
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Skipped due to missing vLLM grouped_topk: {e}")

    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=vllm_grouped_topk,
        dtypes=[torch.float32, torch.bfloat16],
        renormalize=True,
        scoring_func=0,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the grouped_topk kernel.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.grouped_topk
def test_perf_grouped_topk_no_renorm():
    try:
        from vllm._custom_ops import grouped_topk as vllm_grouped_topk
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Skipped due to missing vLLM grouped_topk: {e}")

    bench = GroupedTopKBenchmark(
        op_name="grouped_topk_no_renorm",
        torch_op=vllm_grouped_topk,
        dtypes=[torch.float32, torch.bfloat16],
        renormalize=False,
        scoring_func=0,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the grouped_topk kernel.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.grouped_topk_sigmoid
def test_perf_grouped_topk_sigmoid():
    try:
        from vllm._custom_ops import grouped_topk as vllm_grouped_topk
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Skipped due to missing vLLM grouped_topk: {e}")

    bench = GroupedTopKBenchmark(
        op_name="grouped_topk_sigmoid",
        torch_op=vllm_grouped_topk,
        dtypes=[torch.float32, torch.bfloat16],
        renormalize=True,
        scoring_func=1,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()
