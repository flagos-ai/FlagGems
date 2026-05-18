import math

import pytest
import torch

from flag_gems.fused.fused_qnorm_rope_kv import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
)

from . import base


def _torch_fused_qnorm_rope_kv_ref(
    q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
    eps=1e-6, cache_block_size=16,
):
    """Pure-PyTorch reference: QNorm + RoPE + FP8 KV cache insert."""
    N, H, D = q.shape
    N_ins = slot_mapping.shape[0]
    cache_stride = k_cache.stride(0)

    for tok in range(N):
        pos = position_ids[tok].item()
        cos = cos_sin_cache[pos, :32]
        sin = cos_sin_cache[pos, 32:]

        for head in range(H):
            x = q[tok, head, :].float()
            var = (x * x).mean()
            rsqrt_val = torch.rsqrt(var + torch.tensor(eps, dtype=torch.float32, device=x.device))
            x_norm = x * rsqrt_val

            q[tok, head, :448] = x_norm[:448].to(torch.bfloat16)

            re_f = x_norm[448::2].to(torch.bfloat16).float()
            ro_f = x_norm[449::2].to(torch.bfloat16).float()
            q[tok, head, 448::2] = (re_f * cos - ro_f * sin).to(torch.bfloat16)
            q[tok, head, 449::2] = (re_f * sin + ro_f * cos).to(torch.bfloat16)

    for tok in range(N_ins):
        slot_id = slot_mapping[tok].item()
        if slot_id < 0:
            continue

        pos = position_ids[tok].item()
        cos = cos_sin_cache[pos, :32]
        sin = cos_sin_cache[pos, 32:]

        kv_f = kv[tok].to(torch.bfloat16).float()

        x_e = kv_f[448::2][:32]
        x_o = kv_f[449::2][:32]
        out_e = x_e * cos - x_o * sin
        out_o = x_e * sin + x_o * cos

        block_idx = slot_id // cache_block_size
        pos_in_block = slot_id % cache_block_size
        byte_off_tok = block_idx * cache_stride + pos_in_block * 576
        byte_off_scale = (block_idx * cache_stride
                          + cache_block_size * 576 + pos_in_block * 8)

        flat = k_cache.view(-1)

        for b in range(7):
            bdata = kv_f[b * 64:(b + 1) * 64]
            absmax = bdata.abs().max().clamp(min=1e-4)
            exponent = math.ceil(math.log2(absmax.item() / 448.0))
            inv_scale = 2.0 ** (-exponent)
            scaled = (bdata * inv_scale).clamp(-448.0, 448.0)
            fp8_vals = scaled.to(torch.float8_e4m3fn)
            fp8_i8 = fp8_vals.view(torch.int8).view(torch.uint8)
            flat[byte_off_tok + b * 64:byte_off_tok + (b + 1) * 64] = fp8_i8
            enc_scale = max(0, min(255, int(exponent + 127)))
            flat[byte_off_scale + b] = enc_scale

        flat[byte_off_scale + 7] = 0

        bf16_bytes = torch.zeros(128, dtype=torch.uint8, device=kv.device)
        rope_bf16 = torch.zeros(64, dtype=torch.bfloat16, device=kv.device)
        rope_bf16[0::2] = out_e.to(torch.bfloat16)
        rope_bf16[1::2] = out_o.to(torch.bfloat16)
        bf16_bytes[:] = rope_bf16.view(torch.uint8)
        flat[byte_off_tok + 448:byte_off_tok + 576] = bf16_bytes


class FusedQNormRopeKVBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, H"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 128),
            (4, 128),
            (17, 128),
            (64, 128),
            (1024, 128),
            (2048, 128),
        ]

    def get_input_iter(self, dtype):
        cache_block_size = 16
        max_pos = 8192

        for N, H in self.shapes:
            torch.manual_seed(42)
            q = torch.randn((N, H, 512), dtype=torch.bfloat16, device=self.device)
            kv = torch.randn((N, 576), dtype=torch.bfloat16, device=self.device)

            num_blocks = (N + cache_block_size - 1) // cache_block_size + 1
            cache_stride = cache_block_size * 576 + cache_block_size * 8
            k_cache = torch.zeros(
                (num_blocks, cache_stride), dtype=torch.uint8, device=self.device
            )

            slot_mapping = torch.arange(N, dtype=torch.int32, device=self.device)
            position_ids = torch.randint(
                0, max_pos, (N,), dtype=torch.int64, device=self.device
            )
            cos_sin_cache = torch.randn(
                (max_pos, 64), dtype=torch.float32, device=self.device
            )

            yield (
                q, kv, k_cache, slot_mapping,
                position_ids, cos_sin_cache,
                1e-6, cache_block_size,
            )


@pytest.mark.fused_qnorm_rope_kv
def test_fused_qnorm_rope_kv():
    bench = FusedQNormRopeKVBenchmark(
        op_name="fused_qnorm_rope_kv",
        torch_op=_torch_fused_qnorm_rope_kv_ref,
        gems_op=fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
        dtypes=[torch.bfloat16],
    )
    bench.run()
