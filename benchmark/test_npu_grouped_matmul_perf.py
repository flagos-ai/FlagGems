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
    return torch.ops.npu.npu_grouped_matmul.default


def _inputs(shape, dtype, device):
    import torch_npu

    m, k, n, e, input_bits, weight_bits = shape

    def random_fp8(dims):
        return (torch.randn(dims, dtype=torch.bfloat16, device=device) * 0.25).to(
            torch.float8_e4m3fn
        )

    def scales(dims):
        return torch.randint(124, 128, dims, device=device, dtype=torch.uint8)

    counts = torch.full((e,), m // e, device=device, dtype=torch.int64)
    counts[: m % e] += 1
    if input_bits == 4:
        x = torch.randint(0, 256, (m, k // 2), device=device, dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
    else:
        x = random_fp8((m, k))
    if weight_bits == 4:
        raw = torch.randint(0, 256, (e, n, k // 2), device=device, dtype=torch.uint8)
        if input_bits == 8:
            raw = torch_npu.npu_format_cast(
                raw,
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
        w = raw.transpose(1, 2)
        ws = scales((e, n, k // 64, 2)).transpose(1, 2)
    else:
        w = random_fp8((e, k, n))
        ws = scales((e, k // 64, n, 2))
    sx = scales((m, k // 64, 2)).view(torch.float8_e8m0fnu)

    kwargs = dict(
        per_token_scale=[sx],
        group_list=counts if weight_bits == 4 else counts.cumsum(0),
        split_item=2,
        group_type=0,
        group_list_type=int(weight_bits == 4),
        output_dtype=torch.bfloat16,
        per_token_scale_dtype=293,
    )
    if input_bits == 8 and weight_bits == 4:
        kwargs.update(
            antiquant_scale=[ws.view(torch.float8_e8m0fnu)],
            x_dtype=24,
            weight_dtype=296,
        )
    else:
        kwargs.update(
            scale=[ws],
            scale_dtype=293,
            x_dtype=296 if input_bits == 4 else None,
            weight_dtype=296 if weight_bits == 4 else None,
        )
    return ([x], [w]), kwargs


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.npu_grouped_matmul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="Atlas A5 specialization",
)
def test_npu_grouped_matmul_perf():
    bench = A5Benchmark(
        op_name="npu_grouped_matmul",
        input_fn=_input_fn,
        torch_op=_reference(),
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.npu_grouped_matmul)
    bench.run()
