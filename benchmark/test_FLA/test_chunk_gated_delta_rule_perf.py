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


def torch_int8_state_gdn_prefill(
    q, k, v, g, beta, initial_state, cu_seqlens, raw_cu_seqlens
):
    """PyTorch BF16/FP16-activation reference with FP32 recurrent state.

    The chunk algebra follows FlagGems PR 7279's PyTorch reference. Sequence
    boundaries are supplied on the host so timing contains no device reads.
    """
    dtype = q.dtype
    q = (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(
        dtype
    )
    k = (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(
        dtype
    )
    state, bounds = initial_state, raw_cu_seqlens
    group = v.shape[2] // q.shape[2]
    q = q[0].float().repeat_interleave(group, 1).transpose(0, 1)
    k = k[0].float().repeat_interleave(group, 1).transpose(0, 1)
    v = v[0].float().transpose(0, 1)
    g = g[0].transpose(0, 1)
    beta = beta[0].float().transpose(0, 1)
    outputs, states = [], []
    for seq, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        length = stop - start
        chunks = (length + 63) // 64
        padding = chunks * 64 - length
        qq = torch.nn.functional.pad(q[:, start:stop], (0, 0, 0, padding)).reshape(
            q.shape[0], chunks, 64, q.shape[-1]
        )
        kk = torch.nn.functional.pad(k[:, start:stop], (0, 0, 0, padding)).reshape(
            k.shape[0], chunks, 64, k.shape[-1]
        )
        vv = torch.nn.functional.pad(v[:, start:stop], (0, 0, 0, padding)).reshape(
            v.shape[0], chunks, 64, v.shape[-1]
        )
        gg = (
            torch.nn.functional.pad(g[:, start:stop], (0, padding))
            .reshape(g.shape[0], chunks, 64)
            .cumsum(-1)
        )
        bb = torch.nn.functional.pad(beta[:, start:stop], (0, padding)).reshape(
            beta.shape[0], chunks, 64
        )
        decay = (gg[..., :, None] - gg[..., None, :]).exp().tril()
        lhs = (kk @ kk.transpose(-1, -2) * decay * bb[..., None]).tril(-1)
        # Strictly lower 64x64 L is nilpotent: the finite geometric series
        # gives (I+L)^-1 exactly, using only PyTorch batched matmuls.
        power = -lhs
        inverse = torch.eye(64, device=q.device) + power
        for _ in range(5):
            power = power @ power
            inverse = inverse + inverse @ power
        rhs = bb[..., None] * torch.cat((vv, gg.exp()[..., None] * kk), dim=-1)
        solved = inverse @ rhs
        u, w = solved.split((vv.shape[-1], kk.shape[-1]), dim=-1)
        q_exp = qq * gg.exp()[..., None]
        qk = (qq @ kk.transpose(-1, -2)) * decay
        weighted_k = kk * (gg[..., -1:] - gg).exp()[..., None]
        last_decay = gg[..., -1].exp()
        s = state[seq].transpose(-1, -2)
        chunk_outputs = []
        for i in range(chunks):
            update = u[:, i] - w[:, i] @ s
            chunk_outputs.append(q_exp[:, i] @ s + qk[:, i] @ update)
            s = (
                last_decay[:, i, None, None] * s
                + weighted_k[:, i].transpose(-1, -2) @ update
            )
        outputs.append(
            torch.stack(chunk_outputs, dim=1).reshape(
                q.shape[0], chunks * 64, v.shape[-1]
            )[:, :length]
        )
        states.append(s.transpose(-1, -2))
    output = torch.cat(outputs, dim=1).transpose(0, 1).unsqueeze(0) * (
        q.shape[-1] ** -0.5
    )
    return output.to(dtype), torch.stack(states).contiguous()


def build_int8_gdn_prefill_inputs(shape, dtype):
    batch, tokens, query_heads, reduction, heads, columns, sequences = shape
    torch.manual_seed(42)
    device = flag_gems.device
    q = torch.randn((batch, tokens, query_heads, reduction), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn((batch, tokens, heads, columns), device=device, dtype=dtype) * 0.125
    g = -torch.rand((batch, tokens, heads), device=device, dtype=torch.float32) * 0.1
    beta = torch.rand((batch, tokens, heads), device=device, dtype=dtype)
    original = (
        torch.randn((sequences, heads, columns, reduction), device=device, dtype=dtype)
        * 0.1
    )
    state, scales = flag_gems.quantize_gdn_state_int8(original)
    initial = state.float() * scales[..., None]
    # The exports omit sequence boundaries; the user authorized uniform splits.
    lengths = [
        tokens // sequences + (index < tokens % sequences) for index in range(sequences)
    ]
    boundaries, pairs, starts = [0], [], [0]
    for sequence, length in enumerate(lengths):
        boundaries.append(boundaries[-1] + length)
        pairs.extend((sequence, chunk) for chunk in range(triton.cdiv(length, 64)))
        starts.append(len(pairs))
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    indices = torch.tensor(pairs, device=device, dtype=torch.int32)
    offsets = torch.tensor(starts, device=device, dtype=torch.int32)
    reference = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial,
        cu_seqlens=cu,
        raw_cu_seqlens=tuple(boundaries),
    )
    return (
        torch_int8_state_gdn_prefill,
        reference,
        (q, k, v, g, beta, state, scales, cu, indices, offsets),
    )


class Int8StateGdnBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = [
        (1, 8192, 32, 128),
        (3, 8192, 32, 128),
        ("update", 1, 3, 16, 128, 32, 128, 3),
        (1, 17, 16, 128, 32, 128, 1),
        (1, 1826, 16, 128, 32, 128, 2),
    ]
    DEFAULT_SHAPE_DESC = (
        "decode: T,packed,HV,V; prefill: B,T,H,K,HV,V,N; update: update,B,T,H,K,HV,V,N"
    )

    def get_input_iter(self, dtype):
        from benchmark.test_FLA.test_fused_recurrent_gated_delta_rule_perf import (
            build_int8_gdn_decode_inputs,
        )

        for shape in self.shapes:
            if shape[0] == "update":
                _, batch, tokens, query_heads, reduction, heads, columns, sequences = (
                    shape
                )
                if batch != 1 or sequences != tokens:
                    raise ValueError(
                        "update shapes require B=1 and one token per sequence"
                    )
                packed = 2 * query_heads * reduction + heads * columns
                _, reference, args = build_int8_gdn_decode_inputs(
                    (tokens, packed, heads, columns), dtype, reduction
                )
                q, k, v = torch.split(
                    args[0],
                    (query_heads * reduction, query_heads * reduction, heads * columns),
                    dim=-1,
                )
                qkv = (
                    q.reshape(1, tokens, query_heads, reduction).contiguous(),
                    k.reshape(1, tokens, query_heads, reduction).contiguous(),
                    v.reshape(1, tokens, heads, columns).contiguous(),
                )
                yield ("update", (qkv, *reference[1:])), (qkv, *args[1:])
            elif len(shape) == 7:
                _, reference, args = build_int8_gdn_prefill_inputs(shape, dtype)
                yield ("prefill", reference), args
            else:
                _, reference, args = build_int8_gdn_decode_inputs(shape, dtype)
                yield ("decode", reference), args


def torch_int8_state_gdn_baseline(reference, args):
    from benchmark.test_FLA.test_fused_recurrent_gated_delta_rule_perf import (
        torch_int8_state_gdn_decode,
    )

    if reference[0] == "prefill":
        return torch_int8_state_gdn_prefill(**reference[1])
    else:
        return torch_int8_state_gdn_decode(*reference[1])


def run_int8_state_gdn(reference, args):
    if reference[0] == "prefill":
        return flag_gems.chunk_gated_delta_rule_w8a16_int8(*args)
    elif reference[0] == "update":
        tokens = args[0][0].shape[1]
        mixed = flag_gems.cat(
            tuple(tensor.view(tokens, -1) for tensor in args[0]), dim=1
        )
        return flag_gems.gdn_w8a16_int8(mixed, *args[1:])
    else:
        return flag_gems.gdn_w8a16_int8(*args)


@pytest.mark.gdn_w8a16_int8
@pytest.mark.skipif(
    "gdn_w8a16_int8" not in vars(flag_gems),
    reason="backend does not export INT8-state GDN",
)
def test_perf_gdn_w8a16_int8():
    bench = Int8StateGdnBenchmark(
        op_name="gdn_w8a16_int8", torch_op=torch_int8_state_gdn_baseline
    )
    bench.set_gems(run_int8_state_gdn)
    bench.run()
