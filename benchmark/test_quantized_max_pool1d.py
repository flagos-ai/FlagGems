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

POOL_PARAMS = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "dilation": 1,
    "ceil_mode": False,
}

# PyTorch ships no native QuantizedCUDA kernel for quantized_max_pool1d, so the
# baseline necessarily runs on QuantizedCPU while the FlagGems kernel runs on
# the GPU. The reported speedup is therefore GPU-vs-CPU rather than
# GPU-vs-GPU, and ``latency_base`` is the CPU reference latency.
#
# The host copy of the input is made once in the input function, *outside* the
# timed region, so that neither operator is charged a device-to-host transfer.
# Leaving the copy inside the baseline would have attributed it to the CPU
# kernel and inflated the speedup: for a (32, 8192) quint8 input the transfer
# alone measures ~0.04 ms against ~0.06 ms for the pooling itself.
#
# Because the baseline is CPU-only, prefer ``--mode wrapper`` (wall-clock) for
# these numbers. The default ``kernel`` mode times with CUDA events, which
# cannot observe work that never reaches the GPU and reports a meaningless
# near-constant baseline latency.
QDTYPE = torch.quint8
SCALE = 0.1
ZERO_POINT = 0


def _make_quantized(shape, device):
    fp_tensor = torch.randn(shape, device="cpu").clamp_(-2, 2)
    cpu_tensor = torch.quantize_per_tensor(
        fp_tensor, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE
    )
    q_tensor = cpu_tensor if device == "cpu" else cpu_tensor.to(device)
    # Stash the host copy so the timed baseline never performs the transfer.
    q_tensor._cpu_twin = cpu_tensor
    return q_tensor


def _cpu_input(q_tensor):
    """Return the CPU copy of ``q_tensor`` prepared by the input function."""
    twin = getattr(q_tensor, "_cpu_twin", None)
    if twin is None:
        twin = q_tensor.to("cpu") if q_tensor.device.type != "cpu" else q_tensor
    return twin


def _out_length(in_l, params):
    effective = (params["kernel_size"] - 1) * params["dilation"] + 1
    return (in_l + 2 * params["padding"] - effective) // params["stride"] + 1


def quantized_max_pool1d_input_fn(shape, dtype, device) -> Generator:
    """Yield a device input plus its pre-staged host twin and the pool params."""
    yield _make_quantized(shape, device), dict(POOL_PARAMS)


def quantized_max_pool1d_out_input_fn(shape, dtype, device) -> Generator:
    q_tensor = _make_quantized(shape, device)
    out_shape = shape[:-1] + (_out_length(shape[-1], POOL_PARAMS),)
    out_cpu = torch.quantize_per_tensor(
        torch.zeros(out_shape), scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE
    )
    out = out_cpu if device == "cpu" else out_cpu.to(device)
    # The CPU baseline writes into a host out tensor; allocating it here keeps
    # both the allocation and the host/device copy out of the timed region.
    out._cpu_twin = out_cpu
    # ``unpack_to_args_kwargs`` passes the two tensors positionally and expands
    # the params dict into kwargs.
    yield q_tensor, out, dict(POOL_PARAMS)


def _torch_op(q_tensor, **kwargs):
    """Baseline running PyTorch's quantized_max_pool1d on the CPU.

    The reference is the same aten op rather than dequantize + fp32 max_pool1d
    + requantize, which would time a different computation (two elementwise
    passes plus a float pool) instead of the integer pooling under test. The
    host input is pre-staged, so only the pooling is timed.
    """
    return torch.quantized_max_pool1d(_cpu_input(q_tensor), **kwargs)


def _torch_out_op(q_tensor, out, **kwargs):
    return torch.ops.aten.quantized_max_pool1d.out(
        _cpu_input(q_tensor), out=_cpu_input(out), **kwargs
    )


def _gems_out_op(q_tensor, out, **kwargs):
    return flag_gems.quantized_max_pool1d_out(q_tensor, out=out, **kwargs)


class QuantizedMaxPool1dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in QUANT_POOL_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.quantized_max_pool1d
def test_quantized_max_pool1d():
    bench = QuantizedMaxPool1dBenchmark(
        input_fn=quantized_max_pool1d_input_fn,
        op_name="quantized_max_pool1d",
        torch_op=_torch_op,
        gems_op=flag_gems.quantized_max_pool1d,
        dtypes=[QDTYPE],
    )
    bench.run()


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out():
    bench = QuantizedMaxPool1dBenchmark(
        input_fn=quantized_max_pool1d_out_input_fn,
        op_name="quantized_max_pool1d_out",
        torch_op=_torch_out_op,
        gems_op=_gems_out_op,
        dtypes=[QDTYPE],
    )
    bench.run()
