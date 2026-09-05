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

from . import base, consts

# quantized_max_pool3d operates on per-tensor quint8 tensors. PyTorch ships no
# native QuantizedCUDA kernel for it, so the baseline runs on the CPU
# (QuantizedCPU) while the FlagGems kernel runs on the GPU.
QDTYPE = torch.quint8
SCALE = 0.1
ZERO_POINT = 128

# Representative 3D CNN feature volumes (N, C, D, H, W).
QUANTIZED_MAX_POOL3D_SHAPES = [
    (4, 3, 16, 56, 56),
    (8, 64, 8, 28, 28),
    (16, 128, 4, 14, 14),
    (32, 256, 2, 7, 7),
]


def _make_qinput(shape, device):
    x = torch.randn(shape, device=device)
    return torch.quantize_per_tensor(
        x, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE
    )


def quantized_max_pool3d_input_fn(shape, dtype, device):
    # ``dtype`` is torch.quint8 here (see ``dtypes`` below); we generate a
    # quantized tensor of that dtype directly.
    if dtype != QDTYPE:
        x = torch.randn(shape, device=device)
        qx = torch.quantize_per_tensor(
            x, scale=SCALE, zero_point=ZERO_POINT, dtype=dtype
        )
    else:
        qx = _make_qinput(shape, device)
    yield qx, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Non-cubic kernel/stride/padding (needs spatial dims > 5)
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            yield qx, {
                "kernel_size": (2, 3, 3),
                "stride": (1, 2, 2),
                "padding": (0, 1, 1),
                "dilation": 1,
                "ceil_mode": False,
            }
        # ceil_mode
        yield qx, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": True,
        }


def _torch_op(qx, **kwargs):
    """Baseline running PyTorch's quantized_max_pool3d on the CPU.

    PyTorch has no native QuantizedCUDA kernel, so the reference runs on
    QuantizedCPU. We move the quantized tensor to the CPU first.
    """
    return torch.quantized_max_pool3d(qx.to("cpu"), **kwargs)


class QuantizedMaxPool3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in QUANTIZED_MAX_POOL3D_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.quantized_max_pool3d
def test_quantized_max_pool3d():
    bench = QuantizedMaxPool3dBenchmark(
        input_fn=quantized_max_pool3d_input_fn,
        op_name="quantized_max_pool3d",
        torch_op=_torch_op,
        gems_op=flag_gems.quantized_max_pool3d,
        dtypes=[QDTYPE],
    )
    bench.run()


@pytest.mark.quantized_max_pool3d_out
def test_quantized_max_pool3d_out():
    def out_input_fn(shape, dtype, device):
        for forward_args in quantized_max_pool3d_input_fn(shape, dtype, device):
            qx, params = forward_args
            # Pre-allocate a matching out tensor so the baseline (.out) kernel
            # and the gems kernel share the same output geometry.
            ref_shape = torch.quantized_max_pool3d(qx.to("cpu"), **params).shape
            out_q = torch.quantize_per_tensor(
                torch.zeros(ref_shape, dtype=torch.float32, device=device),
                SCALE,
                ZERO_POINT,
                QDTYPE,
            )
            # ``unpack_to_args_kwargs`` places the two tensors positionally and
            # expands the params dict into kwargs.
            yield qx, out_q, params

    def torch_out_op(qx, out, **kwargs):
        out_cpu = torch.quantize_per_tensor(
            torch.zeros(out.shape, dtype=torch.float32, device="cpu"),
            out.q_scale(),
            out.q_zero_point(),
            QDTYPE,
        )
        torch.ops.aten.quantized_max_pool3d.out(qx.to("cpu"), out=out_cpu, **kwargs)
        out.copy_(out_cpu)
        return out

    def gems_out_op(qx, out, **kwargs):
        return flag_gems.quantized_max_pool3d_out(qx, out=out, **kwargs)

    bench = QuantizedMaxPool3dBenchmark(
        input_fn=out_input_fn,
        op_name="quantized_max_pool3d_out",
        torch_op=torch_out_op,
        gems_op=gems_out_op,
        dtypes=[QDTYPE],
    )
    bench.run()
