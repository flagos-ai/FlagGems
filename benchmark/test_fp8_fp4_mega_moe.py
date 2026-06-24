import pytest
import torch

try:
    from vllm.platforms import current_platform
    from vllm.third_party import deep_gemm

    VLLM_AVAILABLE = True
    SM100_AVAILABLE = current_platform.has_device_capability(100)
except ImportError:
    deep_gemm = None
    VLLM_AVAILABLE = False
    SM100_AVAILABLE = False

from . import base
from tests.test_fp8_fp4_mega_moe import _build_inputs


class FP8FP4MegaMoEBenchmark(base.Benchmark):
    """Benchmark for fp8_fp4_mega_moe: FlagGems Triton vs vLLM DeepGEMM."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 32, 32, 2, 1),
            (4, 64, 32, 4, 2),
        ]
        self.shape_desc = "num_tokens, hidden, intermediate, num_experts, top_k"

    def get_input_iter(self, dtype):
        for num_tokens, hidden, intermediate, num_experts, top_k in self.shapes:
            case = _build_inputs(
                num_tokens,
                hidden,
                intermediate,
                num_experts,
                top_k,
                self.device,
            )
            yield case + (dtype,)


def _vllm_wrapper(
    x_fp8,
    x_scale,
    topk_idx,
    topk_weights,
    l1_weights,
    l1_scales,
    l2_weights,
    l2_scales,
    dtype,
):
    raise RuntimeError(
        "vLLM/DeepGEMM fp8_fp4_mega_moe requires a torch.distributed "
        "symmetric-memory group. Use DeepGEMM tests/test_mega_moe.py for the "
        "production benchmark."
    )


def _gems_wrapper(
    x_fp8,
    x_scale,
    topk_idx,
    topk_weights,
    l1_weights,
    l1_scales,
    l2_weights,
    l2_scales,
    dtype,
):
    from flag_gems.fused import fp8_fp4_mega_moe

    return fp8_fp4_mega_moe(
        x_fp8,
        x_scale,
        topk_idx,
        topk_weights,
        l1_weights,
        l1_scales,
        l2_weights,
        l2_scales,
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and SM100_AVAILABLE),
    reason="requires CUDA with Blackwell architecture (SM100+)",
)
@pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="requires vLLM with DeepGEMM MegaMoE support",
)
@pytest.mark.fp8_fp4_mega_moe
def test_fp8_fp4_mega_moe():
    pytest.skip(
        "vLLM/DeepGEMM fp8_fp4_mega_moe requires a torch.distributed "
        "symmetric-memory group, so it cannot be invoked through the generic "
        "single-process Benchmark harness."
    )

    bench = FP8FP4MegaMoEBenchmark(
        op_name="fp8_fp4_mega_moe",
        torch_op=_vllm_wrapper,
        gems_op=_gems_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.run()
