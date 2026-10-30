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

import pytest
import torch

import flag_gems

from . import base, consts

vendor_name = flag_gems.vendor_name

# NOTE: This is a dead function identified during refactoring
# def weight_norm_interface_input_fn(shape, dtype, device):
#    dim = 0
#    v = torch.randn(shape, dtype=dtype, device=device)
#    g = torch.randn(shape[dim], dtype=dtype, device=device)
#    yield v, g, dim


def weight_norm_input_fn(shape, dtype, device):
    v = torch.randn(shape, dtype=dtype, device=device)
    if vendor_name in ["cambricon", "enflame"]:
        # Cambricon and Enflame fix input shape limit.
        g = torch.randn(shape[:1] + (1,) * (len(shape) - 1), dtype=dtype, device=device)
    else:
        g = torch.randn(shape, dtype=dtype, device=device)
    yield v, g, 0


@pytest.mark.weight_norm_interface
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="weight_norm_interface",
        input_fn=weight_norm_input_fn,
        torch_op=torch._weight_norm,
    )
    bench.set_gems(flag_gems.weight_norm)

    bench.run()


_WEIGHT_H800_LEGACY_SHA = (
    "ed09649e4b5e64b94c6c610cc01422117f792ee8316524f3222711e4c5f3f4c3"
)
_WEIGHT_H800_EXTRA_SHA = (
    "63e7b4f7cf14d9f0fff0e28014cf4a5680c9235088e26445e63e6fd415e6d8a4"
)
_WEIGHT_LEGACY_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]
_WEIGHT_EXTRA_CASES = [
    ((4096, 256), 0),
    ((4096, 256), 1),
    ((4, 256, 3), 0),
    ((4, 256, 3), 2),
]


@dataclass
class _WeightBackwardMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _AscendWeightBackwardStaticReference:
    """Exact external native CUDA references for Ascend's CPU-fallback API."""

    @cached_property
    def _weight_h800(self):
        artifacts = {}
        for legacy, variable, expected, name, source in [
            (
                True,
                "FLAGGEMS_WEIGHT_BACKWARD_H800_LEGACY",
                _WEIGHT_H800_LEGACY_SHA,
                "weight_norm_interface_backward",
                "ccc0508d81b3b10283a62dab73fc05e256093848",
            ),
            (
                False,
                "FLAGGEMS_WEIGHT_BACKWARD_H800_EXTRA",
                _WEIGHT_H800_EXTRA_SHA,
                "weight_norm_backward_current_extra_core",
                "933d774cfac9b9eb82e6811ed2008da406ae9c52",
            ),
        ]:
            path = Path(os.environ[variable]).resolve(strict=True)
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Exact reviewed weight-norm H800 artifact required")
            data = json.loads(payload)
            task = data["_collection"]["tasks"][name]
            env = data["_environments"][str(task["device"])]
            if (
                task["status"] != "completed"
                or task["returncode"] != 0
                or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
                or env["source"]["git_commit"] != source
                or env["source"]["git_dirty"] is not False
                or data[name]["native_api"] != "ops." + name
                or data["_collection"]["configuration"]["repo"] != "pytorch"
                or (not legacy and env["run"]["ncu_enabled"] is not False)
            ):
                raise ValueError("Completed clean native H800 provenance required")
            records = [r for ds in data[name]["shapes"].values() for r in ds.values()]
            if len(records) != (5 if legacy else 4):
                raise ValueError("Every original H800 configuration must be present")
            for r in records:
                us = r["T_us"]
                if (
                    isinstance(us, bool)
                    or not isinstance(us, (int, float))
                    or not math.isfinite(us)
                    or us <= 0
                    or not math.isclose(r["latency_ms"], us / 1000, rel_tol=1e-12)
                ):
                    raise ValueError(
                        "Finite native latency and exact unit conversion required"
                    )
            artifacts[legacy] = (
                path,
                expected,
                records,
                dict(
                    artifact=str(path),
                    artifact_sha256=expected,
                    operator=name,
                    collection_task=task,
                    environment=env,
                    environment_qualification=(
                        "Historical per-device environment is the last task, not a per-op worker receipt"
                        if legacy
                        else "Actual four-case native H800 worker; standard collector "
                        "default do_bench, no explicit median"
                    ),
                    baseline="Literal native CUDA H800; no target NPU-native or CPU timing claim",
                    input_identity="Exact original recipe/configuration; independently random payload bytes differ",
                    saved_norms="Actual provided saved_norms is an input; no norm recomputation",
                    timing="Raw H800/local candidate ratio; shared functional timing, no isolated performance claim",
                ),
            )
        return artifacts

    def build_inputs(self, case):
        self._weight_input_case = case
        return super().build_inputs(case)

    def get_latency(self, op, *args, **kwargs):
        # Ordinary Gems dispatch keeps the reference callable's identity.
        # Consume the external reference once; candidate calls use the timer.
        native_reference = getattr(self, "_weight_native_pending", False) or getattr(
            base.Config, "reference_only", False
        )
        if (
            flag_gems.vendor_name != "ascend"
            or op is not self.torch_op
            or not native_reference
        ):
            return super().get_latency(op, *args, **kwargs)
        self._weight_native_pending = False
        if (
            self.op_name != "weight_norm_interface_backward"
            or self.is_backward
            or base.Config.skip_native
            or base.Config.base_data is not None
            or base.Config.bench_level != consts.BenchLevel.CORE
            or len(args) != 5
            or kwargs
            or type(args[4]) is not int
        ):
            raise ValueError(
                "Original forward core backward-API call configuration required"
            )
        case = self._weight_input_case
        tensors, dim = args[:4], args[4]
        target = torch.device(self.device)
        if target.index is None:
            target = torch.device(target.type, base.torch_device_fn.current_device())
        if not all(
            isinstance(t, torch.Tensor)
            and t.device == target
            and t.dtype == torch.float32
            and t.storage_offset() == 0
            and t.is_contiguous()
            and not t.requires_grad
            for t in tensors
        ):
            raise ValueError(
                "Actual original target/dtype/layout required for all saved inputs"
            )
        if len({t.untyped_storage()._cdata for t in tensors}) != 4:
            raise ValueError("Original independent saved-input storage required")
        legacy = case.case_id.startswith("benchmark/test_weight_norm_interface.py::")
        shape = tuple(tensors[0].shape)
        expected_ordinal = (
            _WEIGHT_LEGACY_SHAPES.index(shape)
            if legacy
            else _WEIGHT_EXTRA_CASES.index((shape, dim))
        )
        filename = (
            "benchmark/test_weight_norm_interface.py"
            if legacy
            else "benchmark/test_weight_norm_interface_backward.py"
        )
        expected_id = (
            filename
            + "::test_weight_norm_interface_backward::core::float32::"
            + str(expected_ordinal)
        )
        expected_shape = dict(grad_w=list(shape), saved_v=list(shape))
        if legacy:
            expected_shape.update(saved_g=[shape[0]], saved_norms=[shape[0]])
        if (
            case.case_id != expected_id
            or case.ordinal != expected_ordinal
            or str(case.dtype) != "torch.float32"
            or case.shape != expected_shape
            or case.params != dict(dim=dim)
            or (legacy and dim != 0)
        ):
            raise ValueError(
                "Original complete case ID/dtype/shape/parameter joint required"
            )

        def describe(t):
            value = dict(
                shape=list(t.shape),
                dtype=str(t.dtype),
                stride=list(t.stride()),
                requires_grad=t.requires_grad,
            )
            if not legacy:
                value["storage_offset"] = t.storage_offset()
            return value

        inputs = dict(args=[*[describe(t) for t in tensors], dim], kwargs={})
        path, expected, records, provenance = self._weight_h800[legacy]
        if legacy:
            config = dict(
                inputs=inputs,
                source_commit="9d57a23b4386bd669152a935fe88a96db907da7c",
                benchmark=filename,
                descriptor=list(shape),
                scope="core",
                variant=0,
            )
            matches = [r for r in records if r["config"] == config]
        else:
            matches = [
                r
                for r in records
                if r["config"]["original_case_id"] == expected_id
                and r["config"]["inputs"] == inputs
                and r["config"]["original_shape"] == expected_shape
                and r["config"]["original_params"] == dict(dim=dim)
                and r["config"]["ordinal"] == expected_ordinal
                and r["config"]["source_commit"]
                == "2a3a455fa9eef530dc11a6b96ff333ffb47235fd"
                and r["config"]["benchmark"] == filename
                and r["config"]["native_expression"]
                == "torch.ops.aten._weight_norm_interface_backward.default"
                and r["config"]["input_recipe"]["seed"] == 0
            ]
        if len(matches) != 1:
            raise ValueError("One exact native H800 input configuration required")
        record = matches[0]
        self._weight_imported_source = dict(
            **provenance,
            config=record["config"],
            original_case_id=case.case_id,
            target_timing_mode=base.Config.mode.value,
            random_recipe=(
                "unseeded four torch.randn inputs, including saved_norms"
                if legacy
                else "seed0 grad/v/g; original float32 sqrt-sum norm builder"
            ),
        )
        return record["latency_ms"]

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name != "ascend":
            return super()._measure_input(input, case_id=case_id)
        if case_id != self._weight_input_case.case_id:
            raise ValueError("Measured case must equal the original built case")
        self._weight_imported_source = None
        needs_reference = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        self._weight_native_pending = needs_reference
        try:
            metric = super()._measure_input(input, case_id=case_id)
            source = self._weight_imported_source
            if source is None:
                if needs_reference:
                    raise ValueError("Static native H800 reference was not consumed")
                return metric
            if metric.latency_base is None:
                raise ValueError("Static native H800 reference latency must be present")
            if "latency" in self.to_bench_metrics and (
                metric.latency is None
                or not math.isfinite(metric.latency)
                or metric.latency <= 0
            ):
                raise ValueError(
                    "Actual local candidate latency must be finite and positive"
                )
            if (
                hashlib.sha256(Path(source["artifact"]).read_bytes()).hexdigest()
                != source["artifact_sha256"]
            ):
                raise ValueError("Native H800 artifact changed around candidate timing")
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
            return _WeightBackwardMetrics(**asdict(metric), baseline_source=source)
        finally:
            self._weight_native_pending = False


class _LegacyWeightBackwardBenchmark(
    _AscendWeightBackwardStaticReference, base.GenericBenchmarkExcluse1D
):
    pass


def weight_norm_interface_backward_input_fn(shape, dtype, device):
    dim = 0
    w_grad = torch.randn(shape, dtype=dtype, device=device)
    saved_v = torch.randn(shape, dtype=dtype, device=device)
    saved_g = torch.randn(shape[dim], dtype=dtype, device=device)
    saved_norms = torch.randn(shape[dim], dtype=dtype, device=device)
    yield w_grad, saved_v, saved_g, saved_norms, dim


def _weight_norm_backward_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={
            "grad_w": list(dims),
            "saved_v": list(dims),
            "saved_g": [dims[0]],
            "saved_norms": [dims[0]],
        },
        params={"dim": 0},
        builder_args=(dims,),
    )


def _weight_norm_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return next(weight_norm_interface_backward_input_fn(shape, dtype, device))


@pytest.mark.weight_norm_interface_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_backward():
    bench = _LegacyWeightBackwardBenchmark(
        op_name="weight_norm_interface_backward",
        case_fn=_weight_norm_backward_case_fn,
        build_inputs_fn=_weight_norm_backward_build_inputs_fn,
        torch_op=torch.ops.aten._weight_norm_interface_backward,
        # NOTE: torch.ops.aten._weight_norm_interface_backward only supports float32,
        # using fp16/bf16 causes "expected scalar type Float but found Half" error.
        # Original: dtypes=consts.FLOAT_DTYPES,
        dtypes=[torch.float32],
    )

    bench.run()
