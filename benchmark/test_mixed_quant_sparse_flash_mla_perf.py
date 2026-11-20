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

import flag_gems

from . import base


def _reference():
    import cann_ops_transformer  # noqa: F401

    return torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla.default


def _inputs(shape, dtype, device):
    import cann_ops_transformer  # noqa: F401

    (tokens,) = shape
    heads = 64
    length = 8192 if tokens == 1 else tokens + 256
    pages = (length + 31) // 32

    def integers(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    def packed(count):
        raw = torch.zeros((count, 32, 1, 608), dtype=torch.uint8)
        rope = torch.randn((count, 32, 1, 64), dtype=torch.float32).bfloat16()
        raw[..., :128] = rope.view(torch.uint8)
        raw[..., 128:576] = (
            torch.randn((count, 32, 1, 448)).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        raw[..., 576:590] = (
            (torch.rand((count, 32, 1, 7)) + 0.125).bfloat16().view(torch.uint8)
        )
        return raw.to(device).view(torch.float8_e4m3fn)

    cu = integers([0, tokens])
    used = integers([length])
    md = dict(
        cu_seqlens_q=cu,
        seqused_ori_kv=used,
        batch_size=1,
        max_seqlen_q=tokens,
        max_seqlen_ori_kv=length,
        rope_head_dim=64,
        ori_mask_mode=4,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=tokens == 1,
    )
    kw = dict(
        ori_kv=packed(pages),
        ori_block_table=integers(list(range(pages)))[None, :],
        cu_seqlens_q=cu,
        seqused_ori_kv=used,
        sinks=torch.randn((heads,), device=device),
        rope_head_dim=64,
        softmax_scale=512**-0.5,
        ori_mask_mode=4,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    if tokens == 1:
        clen = length // 4
        cpages = (clen + 31) // 32
        cused = integers([clen])
        residual = integers([length % 4])
        md.update(
            seqused_cmp_kv=cused,
            cmp_residual_kv=residual,
            max_seqlen_cmp_kv=clen,
            ori_topk=0,
            cmp_topk=512,
            cmp_ratio=4,
            cmp_mask_mode=3,
        )
        kw.update(
            cmp_kv=packed(cpages),
            cmp_block_table=integers(list(range(cpages)))[None, :],
            cmp_sparse_indices=torch.randperm(clen)[:512]
            .sort()
            .values.to(device=device, dtype=torch.int32)
            .view(1, 1, 512),
            seqused_cmp_kv=cused,
            cmp_residual_kv=residual,
            cmp_ratio=4,
            cmp_mask_mode=3,
        )
    kw["metadata"] = (
        torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata.default(
            heads, 1, 512, 1, **md
        ).to(device)
    )
    return (torch.randn((tokens, heads, 512), device=device, dtype=dtype), 1), kw


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.mixed_quant_sparse_flash_mla
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="Atlas A5 specialization",
)
def test_mixed_quant_sparse_flash_mla_perf():
    bench = A5Benchmark(
        op_name="mixed_quant_sparse_flash_mla",
        input_fn=_input_fn,
        torch_op=_reference(),
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.mixed_quant_sparse_flash_mla)
    bench.run()
