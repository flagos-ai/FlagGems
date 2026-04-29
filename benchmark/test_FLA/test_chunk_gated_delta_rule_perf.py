import pytest
import torch
import torch.nn.functional as F

import flag_gems
from benchmark import base


def _chunk_wrapper(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    cu_seqlens,
):
    return flag_gems.chunk_gated_delta_rule(
        q=q.transpose(1, 2).contiguous(),
        k=k.transpose(1, 2).contiguous(),
        v=v.transpose(1, 2).contiguous(),
        beta=beta.transpose(1, 2).contiguous(),
        g=g.transpose(1, 2).contiguous(),
        BT=64,
        scale=scale,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=True,
    )


def _recurrent_wrapper(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    cu_seqlens,
):
    ssm_state_indices = torch.zeros(q.shape[1], device=q.device, dtype=torch.long)
    return flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )


class ChunkGatedDeltaRuleBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "T"
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"

    def get_input_iter(self, cur_dtype):
        for (T,) in self.shapes:
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T: int, dtype: torch.dtype):
        device = flag_gems.device
        Hg, H, K, V = 4, 8, 64, 64

        q = torch.randn((1, T, Hg, K), device=device, dtype=dtype)
        k = torch.randn((1, T, Hg, K), device=device, dtype=dtype)
        v = torch.randn((1, T, H, V), device=device, dtype=dtype)
        g = F.logsigmoid(torch.randn((1, T, H), device=device, dtype=dtype))
        beta = torch.rand((1, T, H), device=device, dtype=dtype).sigmoid()
        initial_state = torch.zeros((1, H, K, V), device=device, dtype=dtype)
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)
        scale = K**-0.5
        return q, k, v, g, beta, float(scale), initial_state, cu_seqlens


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_perf_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule",
        torch_op=_recurrent_wrapper,
    )
    bench.set_gems(_chunk_wrapper)
    bench.run()
