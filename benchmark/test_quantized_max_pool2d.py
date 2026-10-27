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
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Generator

import pytest
import torch

import flag_gems

from flag_gems.ops.quantized_max_pool2d import (
    _parse_pool_params,
    max_pool2d_output_size,
)

from . import base, consts

# ``aten::quantized_max_pool2d`` dispatches over the three per-tensor quantized
# integer dtypes. The accelerator baseline it is measured against is
# ``quantized_max_pool2d_cudnn``, which only accepts qint8 -- for quint8 and
# qint32 the native op raises, so there is no same-device latency_base to
# compare against and the benchmark would fail before reaching the kernel.
# Benchmark the dtype both sides implement.
QUANT_DTYPES = [torch.qint8]

# Spatial shapes exercised by the quantized pooling benchmark, spanning the
# typical ResNet stage outputs from the input image down to the final layer.
POOL_SHAPES = [
    (4, 3, 224, 224),  # Typical input image size
    (16, 64, 56, 56),  # Early ResNet layer output
    (32, 128, 28, 28),  # Mid ResNet layer output
    (64, 256, 14, 14),  # Later ResNet layer output
    (128, 512, 7, 7),  # Final ResNet layer output
]


def _pool_configs(shape, dtype, comprehensive):
    """Yield (kernel_size, stride, padding, dilation, ceil_mode) pooling configs.

    A default 3x3 stride-2 pool, plus extra kernel/stride/padding and ceil_mode
    variants under the comprehensive bench level. Dilation is deliberately left
    at 1: the accelerator baseline is cuDNN-backed and rejects any dilation
    other than [1, 1], so a dilated config has no reference to time against.
    """
    yield 3, 2, 1, 1, False

    if comprehensive:
        if shape[-2] > 5 and shape[-1] > 5:
            yield (3, 5), (2, 1), (1, 2), 1, False
        yield 2, 2, 0, 1, False
        yield 3, 2, 1, 1, True


@dataclass
class _ExternalBaselineMetrics(base.consts.BenchmarkMetrics):
    baseline_source: dict = None


class _ExternalH800Baseline:
    @staticmethod
    def _input_description(value):
        if isinstance(value, torch.Tensor):
            result = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "stride": list(value.stride()),
                "requires_grad": value.requires_grad,
            }
            if value.is_quantized:
                result.update(
                    qscheme=str(value.qscheme()),
                    scale=value.q_scale(),
                    zero_point=value.q_zero_point(),
                )
            return result
        if isinstance(value, (tuple, list)):
            return [_ExternalH800Baseline._input_description(x) for x in value]
        if isinstance(value, dict):
            return {
                k: _ExternalH800Baseline._input_description(v) for k, v in value.items()
            }
        return value

    @cached_property
    def _h800_baseline(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline key: {key}")
                result[key] = value
            return result

        def finite(value):
            raise ValueError(f"Non-finite baseline constant: {value}")

        data = json.loads(payload, object_pairs_hook=unique, parse_constant=finite)
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data[self.op_name]["native_api"] != f"ops.{self.op_name}"
            or environment["run"]["repo"] != "pytorch"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_dirty"] is not False
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or not environment["software"]["packages"]["torch"]
        ):
            raise ValueError(
                "Baseline must be a completed, traceable native H800 collection"
            )
        return data[self.op_name]["shapes"], {
            "kind": "external_nvidia",
            "operator": self.op_name,
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "environment": environment,
            "input_identity": "same distribution and full input contract, not byte-identical random payloads",
            "ratio": "raw H800 latency / locally measured target candidate latency",
        }

    def get_latency(self, op, *args, **kwargs):
        if os.environ.get("FLAGGEMS_NV_BASELINE") and getattr(
            self, "_external_native_pending", False
        ):
            if op is not self.torch_op:
                raise ValueError(
                    "External latency may replace only the original native reference"
                )
            self._external_native_pending = False
            expected = {
                "args": self._input_description(args),
                "kwargs": self._input_description(kwargs),
            }
            shapes, provenance = self._h800_baseline
            matches = [
                record
                for dtypes in shapes.values()
                for record in dtypes.values()
                if record.get("config", {}).get("inputs") == expected
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Expected exactly one H800 record for the full actual input configuration"
                )
            record = matches[0]
            us = record.get("T_us")
            if (
                isinstance(us, bool)
                or not isinstance(us, (int, float))
                or not math.isfinite(us)
                or us <= 0
            ):
                raise ValueError("H800 latency must be finite and positive")
            ms = us / 1000
            recorded_ms = record.get("latency_ms", ms)
            if (
                isinstance(recorded_ms, bool)
                or not isinstance(recorded_ms, (int, float))
                or not math.isfinite(recorded_ms)
                or not math.isclose(recorded_ms, ms, rel_tol=1e-9, abs_tol=1e-12)
            ):
                raise ValueError("H800 latency units disagree")
            self._external_baseline_source = {**provenance, "config": record["config"]}
            return ms
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external and (
            base.Config.skip_native
            or self.is_backward
            or "tflops" in self.to_bench_metrics
        ):
            raise ValueError(
                "External H800 baseline requires the original forward latency measurement"
            )
        self._external_native_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        self._external_baseline_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._external_baseline_source is not None:
                if (
                    metric.latency is None
                    or isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target candidate must still have a real finite positive latency"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _ExternalBaselineMetrics(
                    **asdict(metric), baseline_source=self._external_baseline_source
                )
            return metric
        finally:
            self._external_native_pending = False


class QuantizedMaxPool2dBenchmark(_ExternalH800Baseline, base.GenericBenchmark):
    def get_case_iter(self, dtype):
        comprehensive = base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        ordinal = 0
        for shape in POOL_SHAPES:
            for index, config in enumerate(_pool_configs(shape, dtype, comprehensive)):
                kernel, stride, padding, dilation, ceil_mode = config
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": list(shape)},
                        params={
                            "kernel_size": kernel,
                            "stride": stride,
                            "padding": padding,
                            "dilation": dilation,
                            "ceil_mode": ceil_mode,
                            "scale": _SCALE,
                            "zero_point": 128 if dtype == torch.quint8 else 0,
                        },
                        builder_args=(tuple(shape), index, comprehensive),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        shape, index, comprehensive = case.builder_args[0].builder_args
        return next(
            args
            for i, args in enumerate(
                self.input_fn(shape, case.dtype, self.device, comprehensive)
            )
            if i == index
        )

    def get_input_iter(self, dtype) -> Generator:
        comprehensive = base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        for shape in POOL_SHAPES:
            yield from quantized_max_pool2d_input_fn(
                shape, dtype, self.device, comprehensive
            )


_SCALE = 0.1


def _quantized_input(shape, dtype, device):
    """Build a per-tensor quantized input on ``device``.

    The tensor has to live on the accelerator: the FlagGems kernel is registered
    on the quantized-accelerator dispatch key, so a CPU input would be handled
    by the native CPU kernel and the benchmark would time that instead.
    """
    if dtype == torch.quint8:
        data = torch.rand(shape, device=device) * 255.0
        zero_point = 128
    else:
        data = torch.randint(-128, 128, shape, device=device).float()
        zero_point = 0
    return torch.quantize_per_tensor(data, _SCALE, zero_point, dtype), zero_point


def quantized_max_pool2d_input_fn(shape, dtype, device, comprehensive=True):
    inp, _ = _quantized_input(shape, dtype, device)

    for kernel_size, stride, padding, dilation, ceil_mode in _pool_configs(
        shape, dtype, comprehensive
    ):
        yield inp, {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
        }


class QuantizedMaxPool2dOutBenchmark(QuantizedMaxPool2dBenchmark):
    """Benchmark for the aten::quantized_max_pool2d.out variant.

    The out variant writes the pooled integer representation into a caller
    pre-allocated quantized tensor; the benchmark pre-allocates that tensor
    with the correct output shape (derived from the pooling parameters) and
    the input's scale/zero_point, and passes it through the ``out`` kwarg.
    """

    def get_input_iter(self, dtype) -> Generator:
        comprehensive = base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        for shape in POOL_SHAPES:
            yield from quantized_max_pool2d_out_input_fn(
                shape, dtype, self.device, comprehensive
            )


def quantized_max_pool2d_out_input_fn(shape, dtype, device, comprehensive=True):
    inp, zero_point = _quantized_input(shape, dtype, device)

    for kernel_size, stride, padding, dilation, ceil_mode in _pool_configs(
        shape, dtype, comprehensive
    ):
        (
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        ) = _parse_pool_params(kernel_size, stride, padding, dilation)
        _, _, in_h, in_w = shape
        out_h = max_pool2d_output_size(
            in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
        )
        out_w = max_pool2d_output_size(
            in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
        )
        # Pre-allocate the quantized ``out`` tensor on the same device, with the
        # correct output shape and the input's scale/zero_point, so the timed
        # call neither resizes nor reallocates it.
        out_tensor = torch._empty_affine_quantized(
            (*shape[:2], out_h, out_w),
            scale=_SCALE,
            zero_point=zero_point,
            dtype=dtype,
            device=device,
        )
        yield inp, {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
            "out": out_tensor,
        }


@pytest.mark.quantized_max_pool2d
@pytest.mark.skipif(
    flag_gems.vendor_name == "hygon",
    reason="Hygon native quantized_max_pool2d benchmark baseline requires cuDNN",
)
def test_quantized_max_pool2d():
    bench = QuantizedMaxPool2dBenchmark(
        op_name="quantized_max_pool2d",
        torch_op=torch.quantized_max_pool2d,
        dtypes=QUANT_DTYPES,
        input_fn=quantized_max_pool2d_input_fn,
    )
    bench.run()


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out():
    # The out variant writes the pooled result into the caller-provided quantized
    # tensor. It dispatches through the aten .out overload, which takes the out
    # tensor via the ``out`` keyword; flag_gems registers it as
    # ``quantized_max_pool2d.out`` and the use_gems() context routes the dispatch
    # to the Triton kernel.
    bench = QuantizedMaxPool2dOutBenchmark(
        op_name="quantized_max_pool2d_out",
        torch_op=torch.ops.aten.quantized_max_pool2d.out,
        dtypes=QUANT_DTYPES,
        input_fn=quantized_max_pool2d_out_input_fn,
    )
    bench.run()
