# Copyright 2026 FlagOS Contributors.
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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base

# quantized_max_pool1d pools over the last dimension of a 2D (N, L) or 3D
# (N, C, L) quantized input. These shapes mirror typical 1D-conv / pooling
# workloads.
QUANT_POOL_SHAPES = [
    (4, 1024),
    (16, 4096),
    (32, 8192),
    (64, 16384),
    (8, 3, 1024),
    (16, 8, 2048),
]


def _make_quantized(shape, device):
    fp_tensor = torch.randn(shape, device="cpu").clamp_(-2, 2)
    return torch.quantize_per_tensor(
        fp_tensor, scale=0.1, zero_point=0, dtype=torch.quint8
    ).to(device)


def _torch_baseline(q_tensor, kernel_size, stride, padding, dilation, ceil_mode):
    # The native quantized_max_pool1d has no CUDA kernel, so the baseline
    # dequantizes, runs a plain max_pool1d on fp32, and requantizes back.
    # This keeps the comparison fair and fully on-device.
    fp = q_tensor.dequantize()
    pooled = torch.nn.functional.max_pool1d(
        fp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    return torch.quantize_per_tensor(
        pooled,
        scale=float(q_tensor.q_scale()),
        zero_point=int(q_tensor.q_zero_point()),
        dtype=q_tensor.dtype,
    )


class QuantizedMaxPool1dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in QUANT_POOL_SHAPES:
            q_tensor = _make_quantized(shape, self.device)
            yield q_tensor, {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "ceil_mode": False,
            }


class QuantizedMaxPool1dOutBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in QUANT_POOL_SHAPES:
            q_tensor = _make_quantized(shape, self.device)
            params = {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "ceil_mode": False,
            }
            in_l = q_tensor.shape[-1]
            ks, st, pd, dl = (
                params["kernel_size"],
                params["stride"],
                params["padding"],
                params["dilation"],
            )
            out_l = (in_l + 2 * pd - (ks - 1) * dl - 1) // st + 1
            out_shape = q_tensor.shape[:-1] + (out_l,)
            out = torch.quantize_per_tensor(
                torch.zeros(out_shape),
                scale=float(q_tensor.q_scale()),
                zero_point=int(q_tensor.q_zero_point()),
                dtype=q_tensor.dtype,
            ).to(self.device)
            yield q_tensor, {**params, "out": out}


def _torch_out(q_tensor, kernel_size, stride, padding, dilation, ceil_mode, *, out):
    result = _torch_baseline(
        q_tensor, kernel_size, stride, padding, dilation, ceil_mode
    )
    out.int_repr().copy_(result.int_repr())
    return out


@pytest.mark.quantized_max_pool1d
def test_quantized_max_pool1d():
    bench = QuantizedMaxPool1dBenchmark(
        op_name="quantized_max_pool1d",
        input_fn=None,
        torch_op=_torch_baseline,
        dtypes=[torch.quint8],
    )
    bench.set_gems(flag_gems.quantized_max_pool1d)
    bench.run()


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out():
    bench = QuantizedMaxPool1dOutBenchmark(
        op_name="quantized_max_pool1d_out",
        input_fn=None,
        torch_op=_torch_out,
        dtypes=[torch.quint8],
    )
    bench.set_gems(flag_gems.quantized_max_pool1d_out)
    bench.run()
