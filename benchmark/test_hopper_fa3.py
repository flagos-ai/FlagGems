import pytest
import torch

from tests.hopper_fa3_utils import (
    HAS_VLLM_FA,
    VLLM_FA_HAS_BLOCK_TABLE,
    VLLM_FA_HAS_SEQUSED_K,
    Shape,
    Tensors,
    attn_flops,
    benchmark_shapes,
    is_fa3_supported,
    make_varlen,
    run_flag_gems,
    run_vllm_fa,
)

from . import base, consts


def _gems_wrapper(tensors: Tensors, shape: Shape, fa_version: int):
    return run_flag_gems(tensors, shape, fa_version=fa_version)


def _vllm_wrapper(tensors: Tensors, shape: Shape, fa_version: int):
    return run_vllm_fa(tensors, shape, fa_version=fa_version)


class HopperFA3Benchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_SHAPE_DESC = (
        "name, seq_lens, num_query_heads, num_kv_heads, head_dim, causal, paged"
    )

    def set_shapes(self, shape_file_path=None):
        shapes = benchmark_shapes()
        if not (VLLM_FA_HAS_BLOCK_TABLE and VLLM_FA_HAS_SEQUSED_K):
            shapes = [shape for shape in shapes if not shape.paged]
        self.shapes = shapes

    def get_input_iter(self, dtype):
        for idx, shape in enumerate(self.shapes):
            tensors = make_varlen(shape, dtype, self.device, seed=2026 + idx)
            yield tensors, shape, 3

    def unpack_to_args_kwargs(self, input_tuple):
        return list(input_tuple), {}

    def record_shapes(self, tensors: Tensors, shape: Shape, fa_version: int):
        return {
            "name": shape.name,
            "seq_lens": shape.seq_lens,
            "num_query_heads": shape.nh_q,
            "num_kv_heads": shape.nh_k,
            "head_dim": shape.head_dim,
            "causal": shape.causal,
            "paged": shape.paged,
            "max_seqlen_q": tensors.max_seqlen_q,
            "max_seqlen_k": tensors.max_seqlen_k,
            "fa_version": fa_version,
        }

    def get_tflops(self, op, tensors: Tensors, shape: Shape, fa_version: int):
        return attn_flops(shape)


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(
    not is_fa3_supported(),
    reason="requires CUDA Hopper with Triton FA3 support",
)
@pytest.mark.skipif(
    not HAS_VLLM_FA,
    reason="requires vLLM flash-attention as the benchmark baseline",
)
def test_hopper_fa3():
    bench = HopperFA3Benchmark(
        op_name="hopper_fa3",
        torch_op=_vllm_wrapper,
        gems_op=_gems_wrapper,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
