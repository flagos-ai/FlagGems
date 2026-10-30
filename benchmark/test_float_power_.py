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

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_SCALARS = (0.5, -1.25, 2.0)


def _ku_double_input(tensor, device):
    result = (
        tensor.contiguous()
        .view(torch.uint8)
        .to(device)
        .view(torch.float64)
        .reshape(tensor.shape)
    )
    assert result.dtype == torch.float64 and result.shape == tensor.shape
    assert result.device == torch.empty(0, device=device).device
    return result


def _ku_clone_inputs(value):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.float64:
            raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                value.untyped_storage(), 0, (value.untyped_storage().nbytes(),), (1,)
            )
            return (
                raw.clone()
                .view(value.dtype)
                .as_strided(value.shape, value.stride(), value.storage_offset())
                .requires_grad_(value.requires_grad)
            )
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, (tuple, list)):
        return type(value)(_ku_clone_inputs(item) for item in value)
    if isinstance(value, dict):
        return {name: _ku_clone_inputs(item) for name, item in value.items()}
    return value


@dataclass
class _FloatPowerMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _FloatPowerBenchmark(OperatorBenchmark):
    @cached_property
    def _h800_records(self):
        path = Path(os.environ["FLAGGEMS_FLOAT_POWER_INPLACE_H800_BASELINE"]).resolve(
            strict=True
        )
        payload = path.read_bytes()
        digest = "3c6eaadc9140cb3402b9504800c08655b51376c2f69a563d94178ac8d4a410bc"
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(
                "Expected the published native H800 float_power inplace artifact"
            )
        data = json.loads(payload)
        operator = "float_power_inplace_kunlun_core"
        task = data["_collection"]["tasks"][operator]
        environment = data["_environments"][str(task["device"])]
        records = data[operator]["shapes"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data[operator]["native_api"] != "ops.float_power_inplace_kunlun_core"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "afd284ef16e8c01384b785729d3389dcac76fa8e"
            or environment["source"]["git_dirty"] is not False
            or sum(len(values) for values in records.values()) != 12
        ):
            raise ValueError(
                "Expected twelve traceable native H800 prepared-input records"
            )
        return records, dict(
            artifact=str(path),
            artifact_sha256=digest,
            operator=operator,
            collection_task=task,
            environment=environment,
            input_identity=(
                "Original shape/dtype/params/uniform recipe and seed0 policy; "
                "no cross-device random-byte identity"
            ),
            ratio="Raw native H800 latency / locally timed candidate; no scaling",
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_h800_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._h800_pending = False
        if op is not self.torch_op or self.is_backward or len(args) != 2 or kwargs:
            raise ValueError(
                "Static float_power requires the original forward inplace call"
            )
        tensor = self.op_name == "float_power_tensor_tensor_"
        if self.op_name not in (
            "float_power_tensor_tensor_",
            "float_power_tensor_scalar_",
        ):
            raise ValueError("Unexpected float_power overload")
        inp, exponent = args
        if tuple(inp.shape) not in ((64, 64), (4096, 4096), (64, 512, 512)):
            raise ValueError("Input shape differs from the original Kunlun core")
        for value in args if tensor else args[:1]:
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float64
                or value.device != inp.device
                or value.device != torch.empty(0, device=self.device).device
                or value.shape != inp.shape
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.requires_grad
            ):
                raise ValueError(
                    "Runtime input differs from the H800 Float64 storage contract"
                )
        if not tensor and (type(exponent) is not float or exponent not in _SCALARS):
            raise ValueError("Scalar exponent differs from the original core")
        metadata = dict(
            shape=list(inp.shape),
            dtype=str(inp.dtype),
            stride=list(inp.stride()),
            requires_grad=False,
        )
        expected = dict(
            native_expression="torch.Tensor.float_power_",
            canonical_operator=self.op_name,
            inputs=dict(
                args=[metadata, metadata.copy() if tensor else exponent], kwargs={}
            ),
            params=dict(
                overload="Tensor" if tensor else "Scalar",
                **({} if tensor else dict(exponent=exponent)),
            ),
            source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
            benchmark="benchmark/test_float_power_.py",
            scope="original_kunlun_core",
            descriptor=list(inp.shape),
            input_recipe=dict(
                self_uniform=[0.25, 2.0],
                tensor_exponent_uniform=[-2.0, 3.0] if tensor else None,
                seed=0,
            ),
            timing=dict(
                fresh_inputs=True,
                preparation="original detach.clone.requires_grad recursive recipe then synchronize before start event",
                clock="collector CUDA events",
                statistic="mean",
                warmup_ms=25,
                repetition_ms=100,
            ),
        )
        records, source = self._h800_records
        matches = [
            record
            for values in records.values()
            for dtype, record in values.items()
            if dtype == str(inp.dtype)
            and {
                key: value
                for key, value in record["config"].items()
                if key != "producer_sample"
            }
            == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one exact full-configuration H800 inplace record"
            )
        record = matches[0]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("H800 latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._h800_source = dict(source, config=record["config"])
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if flag_gems.vendor_name != "kunlunxin" or not needs_native:
            return super()._measure_input(input, case_id)
        if not os.environ.get("FLAGGEMS_FLOAT_POWER_INPLACE_H800_BASELINE"):
            raise ValueError(
                "Kunlun float_power_ lacks Float64 native arithmetic; set FLAGGEMS_FLOAT_POWER_INPLACE_H800_BASELINE"
            )
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Static baseline requires the original forward latency measurement"
            )
        self._h800_pending = True
        self._h800_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._h800_source is not None:
                if metric.latency is not None and (
                    isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Candidate must have a real positive latency")
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _FloatPowerMetrics(
                    **asdict(metric), baseline_source=self._h800_source
                )
            return metric
        finally:
            self._h800_pending = False

    def _get_fresh_input_latency(self, op, args, kwargs):
        if flag_gems.vendor_name != "kunlunxin":
            return super()._get_fresh_input_latency(op, args, kwargs)
        # Each sample starts from an independent snapshot. Input restoration
        # happens before the start event/clock, never inside the measured call.
        if self.is_backward:
            raise ValueError("Fresh-input backward measurement is not supported")
        if getattr(base.Config, "reference_only", False):
            fresh_args, fresh_kwargs = _ku_clone_inputs((args, kwargs))
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
            fresh_args, fresh_kwargs = _ku_clone_inputs((args, kwargs))
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


def _tensor_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "exponent": list(dims)},
        params={"overload": "Tensor"},
        builder_args=(dims,),
    )


def _tensor_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    if flag_gems.vendor_name == "kunlunxin":
        inp = _ku_double_input(
            torch.empty(shape, dtype=dtype).uniform_(0.25, 2.0), device
        )
        exponent = _ku_double_input(
            torch.empty(shape, dtype=dtype).uniform_(-2.0, 3.0), device
        )
        return inp, exponent
    inp = torch.empty(shape, dtype=dtype, device=device).uniform_(0.25, 2.0)
    exponent = torch.empty(shape, dtype=dtype, device=device).uniform_(-2.0, 3.0)
    return inp, exponent


def _scalar_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for exponent in _SCALARS:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims)},
            params={"overload": "Scalar", "exponent": exponent},
            builder_args=(dims, exponent),
        )


def _scalar_build_inputs_fn(plan, dtype, device):
    shape, exponent = plan.builder_args
    torch.manual_seed(0)
    if flag_gems.vendor_name == "kunlunxin":
        inp = _ku_double_input(
            torch.empty(shape, dtype=dtype).uniform_(0.25, 2.0), device
        )
        return inp, exponent
    inp = torch.empty(shape, dtype=dtype, device=device).uniform_(0.25, 2.0)
    return inp, exponent


@pytest.mark.float_power_tensor_tensor_
@pytest.mark.float_power_
def test_float_power_tensor_():
    bench = _FloatPowerBenchmark(
        op_name="float_power_tensor_tensor_",
        torch_op=lambda inp, exponent: inp.float_power_(exponent),
        case_fn=_tensor_case_fn,
        build_inputs_fn=_tensor_build_inputs_fn,
        dtypes=[torch.float64],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()


@pytest.mark.float_power_tensor_scalar_
@pytest.mark.float_power_
def test_float_power_scalar_():
    bench = _FloatPowerBenchmark(
        op_name="float_power_tensor_scalar_",
        torch_op=lambda inp, exponent: inp.float_power_(exponent),
        case_fn=_scalar_case_fn,
        build_inputs_fn=_scalar_build_inputs_fn,
        dtypes=[torch.float64],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
