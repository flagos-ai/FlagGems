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

import time

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

SCALE = 0.1
ZERO_POINT = 0
QDTYPE = torch.quint8


def _quantized_device(value, device):
    if flag_gems.vendor_name == "ascend" and torch.device(device).type != "cpu":
        result = torch._empty_affine_quantized(
            value.shape,
            scale=value.q_scale(),
            zero_point=value.q_zero_point(),
            dtype=value.dtype,
            device=device,
        )
        result.untyped_storage().copy_(value.untyped_storage())
        return result
    return value.to(device)


def _make_quantized(shape, device):
    torch.manual_seed(0)
    host = torch.quantize_per_tensor(
        torch.randn(shape).clamp_(-2, 2), SCALE, ZERO_POINT, QDTYPE
    )
    result = _quantized_device(host, device)
    result._cpu_twin = host
    return result


def _quantized_cpu(value):
    if flag_gems.vendor_name == "ascend" and value.device.type != "cpu":
        int_dtype = {
            torch.quint8: torch.uint8,
            torch.qint8: torch.int8,
            torch.qint32: torch.int32,
        }[value.dtype]
        ints = torch.empty(0, dtype=int_dtype).set_(
            value.untyped_storage().cpu(),
            value.storage_offset(),
            value.shape,
            value.stride(),
        )
        return torch._make_per_tensor_quantized_tensor(
            ints, value.q_scale(), value.q_zero_point()
        )
    return value.to("cpu")


def _clone_pool_inputs(value):
    if isinstance(value, torch.Tensor):
        if value.is_quantized and value.device.type == "npu":
            # Own core plans use contiguous per-tensor quantized tensors. Build
            # a genuine independent target clone before each timed call.
            assert value.is_contiguous() and not value.requires_grad
            clone = torch._empty_affine_quantized(
                value.shape,
                scale=value.q_scale(),
                zero_point=value.q_zero_point(),
                dtype=value.dtype,
                device=value.device,
            )
            clone.untyped_storage().copy_(value.untyped_storage())
            return clone
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_pool_inputs(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_pool_inputs(item) for key, item in value.items()}
    return value


def _cpu_input(value):
    twin = getattr(value, "_cpu_twin", None)
    return twin if twin is not None else _quantized_cpu(value)


SHAPES = [(4, 1024), (16, 4096), (32, 8192), (64, 16384), (8, 3, 1024), (16, 8, 2048)]
PARAMS = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "dilation": 1,
    "ceil_mode": False,
}


class QuantizedMaxPool1dBenchmark(OperatorBenchmark):
    def _get_fresh_input_latency(self, op, args, kwargs):
        if flag_gems.vendor_name != "ascend":
            return super()._get_fresh_input_latency(op, args, kwargs)
        # Each sample starts from an independent snapshot. Input restoration
        # happens before the start event/clock, never inside the measured call.
        if self.is_backward:
            raise ValueError("Fresh-input backward measurement is not supported")
        if getattr(base.Config, "reference_only", False):
            fresh_args, fresh_kwargs = _clone_pool_inputs((args, kwargs))
            return base.GenericBenchmark.get_latency(
                self, op, *fresh_args, **fresh_kwargs
            )
        if base.Config.mode == consts.BenchMode.CUDAGRAPH:
            raise ValueError(
                "State-changing benchmarks require fresh inputs; use kernel or operator mode"
            )
        if base.Config.mode not in (
            consts.BenchMode.KERNEL,
            consts.BenchMode.OPERATOR,
            consts.BenchMode.WRAPPER,
        ):
            raise ValueError("Undefined Value of Benchmark Mode.")
        device_timing = base.Config.mode == consts.BenchMode.KERNEL
        if device_timing and not hasattr(base.torch_device_fn, "Event"):
            raise ValueError("Backend has no event timer for fresh-input measurements")

        def sample():
            fresh_args, fresh_kwargs = _clone_pool_inputs((args, kwargs))
            base.torch_device_fn.synchronize()
            if device_timing:
                start = base.torch_device_fn.Event(enable_timing=True)
                end = base.torch_device_fn.Event(enable_timing=True)
                start.record()
                result = op(*fresh_args, **fresh_kwargs)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                start = time.perf_counter()
                result = op(*fresh_args, **fresh_kwargs)
                if base.Config.mode == consts.BenchMode.OPERATOR:
                    base.torch_device_fn.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
            del result
            if elapsed <= 0:
                raise RuntimeError("Fresh-input timer returned a nonpositive latency")
            return elapsed

        warmup_elapsed = 0.0
        while warmup_elapsed < base.Config.warm_up:
            warmup_elapsed += sample()
        measured = []
        elapsed = 0.0
        while not measured or elapsed < base.Config.repetition:
            value = sample()
            measured.append(value)
            elapsed += value
        return sum(measured) / len(measured)

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "N[, C], L"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    out_l = (
        dims[-1]
        + 2 * PARAMS["padding"]
        - (PARAMS["kernel_size"] - 1) * PARAMS["dilation"]
        - 1
    ) // PARAMS["stride"] + 1
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "out": list(dims[:-1] + (out_l,))},
        params=dict(PARAMS),
        builder_args=(dims, out_l),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, out_l = plan.builder_args
    qx = _make_quantized(shape, device)
    out_shape = shape[:-1] + (out_l,)
    out = torch._empty_affine_quantized(
        out_shape, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE, device=device
    )
    out._cpu_twin = torch._empty_affine_quantized(
        out_shape, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE, device="cpu"
    )
    return qx, {**PARAMS, "out": out}


def _torch_op(qx, **kwargs):
    out = kwargs.pop("out")
    return torch.ops.aten.quantized_max_pool1d.out(
        _cpu_input(qx), out=_cpu_input(out), **kwargs
    )


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out():
    QuantizedMaxPool1dBenchmark(
        op_name="quantized_max_pool1d_out",
        torch_op=_torch_op,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[QDTYPE],
        is_inplace=True,
        fresh_inputs=True,
    ).run()
