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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = utils.to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


@pytest.mark.rms_norm
@pytest.mark.parametrize(
    "rows, columns, row_stride, offset",
    [
        (1, 2048, 2624, 0),
        (4, 2048, 2624, 0),
        (4, 512, 2624, 2048),
        (64, 2048, 2624, 0),
        (256, 512, 2624, 2048),
        (4, 513, 769, 3),
        (4, 6144, 6208, 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.no_grad()
def test_rms_norm_row_padding_and_ordinary_graph(
    rows, columns, row_stride, offset, dtype
):
    if flag_gems.vendor_name != "nvidia" or not torch.cuda.is_available():
        pytest.skip("CUDA row-stride forward path")
    from flag_gems.ops.rms_norm import rms_norm_forward

    generator = torch.Generator().manual_seed(719 + rows + columns)
    size = offset + (rows - 1) * row_stride + columns
    cpu_a = torch.randn(size, generator=generator).to(dtype)
    cpu_b = torch.randn(size, generator=generator).to(dtype)
    cpu_weight = torch.randn(columns, generator=generator).to(dtype)
    storage = cpu_a.to(flag_gems.device)
    x = storage.as_strided((rows, columns), (row_stride, 1), offset)
    weight = cpu_weight.to(flag_gems.device)
    for _ in range(2):
        rms_norm_forward(x, (columns,), weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_y, graph_inv = rms_norm_forward(x, (columns,), weight)
    for cpu in (cpu_a, cpu_b, cpu_a):
        storage.copy_(cpu)
        eager_y, eager_inv = rms_norm_forward(x, (columns,), weight)
        packed_y, packed_inv = rms_norm_forward(x.contiguous(), (columns,), weight)
        graph_y.fill_(torch.nan)
        graph_inv.fill_(torch.nan)
        graph.replay()
        torch.testing.assert_close(graph_y, eager_y, atol=0, rtol=0)
        torch.testing.assert_close(graph_inv, eager_inv, atol=0, rtol=0)
        torch.testing.assert_close(graph_y, packed_y, atol=0, rtol=0)
        torch.testing.assert_close(graph_inv, packed_inv, atol=0, rtol=0)
        cpu_x = cpu.as_strided((rows, columns), (row_stride, 1), offset).double()
        inv = 1 / torch.sqrt((cpu_x * cpu_x).mean(-1, keepdim=True) + 1e-5)
        expected = ((cpu_x * inv).to(dtype) * cpu_weight).to(dtype).float()
        tolerance = {
            torch.bfloat16: 0.015625,
            torch.float16: 0.002,
            torch.float32: 1e-5,
        }[dtype]
        torch.testing.assert_close(
            graph_y.cpu().float(), expected, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            graph_inv.cpu().double(), inv[:, 0], atol=2e-6, rtol=2e-6
        )
