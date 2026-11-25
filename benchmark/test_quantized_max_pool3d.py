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
# (QuantizedCPU) while the FlagGems kernel runs on the GPU. The reported speedup
# is therefore GPU-vs-CPU rather than GPU-vs-GPU. Host/device transfers are
# hoisted into the input function so they are not attributed to the baseline.
#
# Because the baseline is CPU-only, prefer ``--mode wrapper`` (wall-clock) for
# these numbers. The default ``kernel`` mode times with CUDA events, which
# cannot observe work that never reaches the GPU and reports a meaningless
# near-constant baseline latency.
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
    # The CPU baseline needs a CPU copy of the input. Take it once here, outside
    # the timed region, so the reported baseline latency is the CPU kernel alone
    # and not a device-to-host transfer (which dominates the smaller shapes).
    qx._cpu_twin = qx.to("cpu") if qx.device.type != "cpu" else qx
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


def _cpu_input(qx):
    """Return the CPU copy of ``qx`` prepared by the input function."""
    twin = getattr(qx, "_cpu_twin", None)
    if twin is None:
        twin = qx.to("cpu") if qx.device.type != "cpu" else qx
    return twin


def _torch_op(qx, **kwargs):
    """Baseline running PyTorch's quantized_max_pool3d on the CPU.

    PyTorch ships no native QuantizedCUDA kernel, so the reference necessarily
    runs on QuantizedCPU. The host copy of the input is made once by the input
    function, so only the pooling itself falls inside the timed region.
    """
    return torch.quantized_max_pool3d(_cpu_input(qx), **kwargs)


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
            ref_shape = torch.quantized_max_pool3d(_cpu_input(qx), **params).shape
            out_q = torch.quantize_per_tensor(
                torch.zeros(ref_shape, dtype=torch.float32, device=device),
                SCALE,
                ZERO_POINT,
                QDTYPE,
            )
            # The CPU baseline writes into a host out tensor. Allocate it once
            # here so neither the allocation nor a host/device copy is timed;
            # both operators then measure just their own pooling work.
            out_q._cpu_twin = torch.quantize_per_tensor(
                torch.zeros(ref_shape, dtype=torch.float32, device="cpu"),
                SCALE,
                ZERO_POINT,
                QDTYPE,
            )
            # ``unpack_to_args_kwargs`` places the two tensors positionally and
            # expands the params dict into kwargs.
            yield qx, out_q, params

    def torch_out_op(qx, out, **kwargs):
        return torch.ops.aten.quantized_max_pool3d.out(
            _cpu_input(qx), out=_cpu_input(out), **kwargs
        )

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
