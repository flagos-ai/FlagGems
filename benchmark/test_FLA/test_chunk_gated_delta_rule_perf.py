import pytest
import torch
import torch.nn.functional as F
import triton

import flag_gems
from benchmark.base import Benchmark

_TRITON_ALLOCATOR_READY = False


@pytest.fixture(autouse=True)
def _install_triton_allocator():
    global _TRITON_ALLOCATOR_READY
    if (
        _TRITON_ALLOCATOR_READY
        or not torch.cuda.is_available()
        or not hasattr(triton, "set_allocator")
    ):
        return

    def _alloc(size: int, _alignment: int, _stream: int | None):
        return torch.empty((size,), dtype=torch.uint8, device=flag_gems.device)

    triton.set_allocator(_alloc)
    _TRITON_ALLOCATOR_READY = True


def _recurrent_wrapper(
    q,
    k,
    v,
    beta,
    g,
    BT=64,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    head_first=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
):
    if BT != 64:
        raise ValueError("chunk gated delta rule benchmark supports only BT=64")
    if head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)
        g = g.transpose(1, 2)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    B, T, _, _ = q.shape
    if cu_seqlens is None:
        q_recurrent = q.reshape(1, B * T, q.shape[2], q.shape[3])
        k_recurrent = k.reshape(1, B * T, k.shape[2], k.shape[3])
        v_recurrent = v.reshape(1, B * T, v.shape[2], v.shape[3])
        beta_recurrent = beta.reshape(1, B * T, beta.shape[2])
        g_recurrent = g.reshape(1, B * T, g.shape[2])
        cu_seqlens_recurrent = torch.arange(
            0, (B + 1) * T, T, device=q.device, dtype=torch.long
        )
        ssm_state_indices = (
            torch.arange(B, device=q.device, dtype=torch.long)
            .view(B, 1)
            .expand(B, T)
            .contiguous()
        )
    else:
        q_recurrent = q
        k_recurrent = k
        v_recurrent = v
        beta_recurrent = beta
        g_recurrent = g
        cu_seqlens_recurrent = cu_seqlens
        ssm_state_indices = None
    if initial_state is None:
        initial_state = q.new_zeros(v.shape[0], v.shape[2], k.shape[-1], v.shape[-1])
    o, final_state = flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=q_recurrent,
        k=k_recurrent,
        v=v_recurrent,
        g=g_recurrent,
        beta=beta_recurrent,
        scale=float(scale),
        initial_state=initial_state.clone(),
        inplace_final_state=output_final_state,
        cu_seqlens=cu_seqlens_recurrent,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if cu_seqlens is None:
        o = o.reshape(B, T, v.shape[2], v.shape[3])
    if head_first:
        o = o.transpose(1, 2)
    return o, final_state if output_final_state else None


class ChunkGatedDeltaRuleBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
    DEFAULT_SHAPE_DESC = "B, T, Hg, H, K, V"

    def get_input_iter(self, cur_dtype):
        for B, T, Hg, H, K, V in self.shapes:
            yield self._build_inputs(B, T, Hg, H, K, V, cur_dtype)

    def _build_inputs(self, B, T, Hg, H, K, V, dtype):
        device = flag_gems.device
        q = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
        k = F.normalize(
            torch.randn(B, T, Hg, K, device=device, dtype=torch.float32),
            p=2.0,
            dim=-1,
            eps=1e-6,
        ).to(dtype)
        v = (0.125 * torch.randn(B, T, H, V, device=device, dtype=torch.float32)).to(
            dtype
        )
        beta = (
            torch.empty(B, T, H, device=device, dtype=torch.float32)
            .uniform_(-2.0, 2.0)
            .sigmoid()
            .to(dtype)
        )
        decay = (
            torch.empty(B, T, H, device=device, dtype=torch.float32)
            .uniform_(-4.605170185988091, -3.506557897319982)
            .exp()
        )
        g = torch.log1p(-decay).to(dtype)
        initial_state = (
            0.125 * torch.randn(B, H, K, V, device=device, dtype=torch.float32)
        ).to(dtype)
        return (
            q,
            k,
            v,
            beta,
            g,
            {
                "BT": 64,
                "initial_state": initial_state,
                "output_final_state": True,
                "cu_seqlens": None,
                "head_first": False,
                "scale": K**-0.5,
                "use_qk_l2norm_in_kernel": False,
            },
        )


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_perf_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule",
        torch_op=_recurrent_wrapper,
    )
    bench.set_gems(flag_gems.chunk_gated_delta_rule)
    bench.run()
