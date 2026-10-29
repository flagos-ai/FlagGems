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

import json
import math
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .baseline import describe_call, positive_number

_H800_PRODUCER_COMMIT = "8dae3fdd50b26e9be9b46622c9c5fde2b9be5e9c"
_PINNED_SOURCE_COMMIT = "9d57a23b4386bd669152a935fe88a96db907da7c"
_TARGET_RECIPE_SOURCE_COMMIT = "b197f259f85795b33498a9a2ef2f7ac46c5ec6bc"
_OUT_BASELINE_RECIPES = {
    "float_power_scalar_tensor_out": {
        "input_recipe": "base=2.0; exponent=unseeded "
        "rand(shape,dtype)*4-2; "
        "out=empty(Float64,shape)",
        "original_builder_sha256": "60a4c322abf34d1ee76f90bc1f5e85031b247754a786be724ebd9377caa10191",
        "target_builder_sha256": "e2a6827401bb4cc9aacd785433d4d62576fb0e31ce89de589c7000959da2ff81",
    },
    "float_power_tensor_scalar_out": {
        "input_recipe": "base=unseeded "
        "rand(shape,dtype)+0.25; "
        "exponent=1.234; "
        "out=empty(Float64,shape)",
        "original_builder_sha256": "d2545ca1ed6f88c451c2359be95e6e22ef676ef3b03b633f55c5c8b2431c8a7e",
        "target_builder_sha256": "c55ebe74a5e33c14440ef8af92d038b03dea8adac0d5e6af9d5d2df3c198ac9c",
    },
    "float_power_tensor_tensor_out": {
        "input_recipe": "base=unseeded "
        "rand(shape,dtype)+0.25; "
        "exponent=unseeded "
        "rand(shape,dtype)*4-2; "
        "out=empty(Float64,shape)",
        "original_builder_sha256": "2c5c95af373f5dd8ffd9fb59bf60630c86e111e9bb8fd60cf2e9c7047ac0d368",
        "target_builder_sha256": "524233832de5143a00f382f30fa653f74b44e143a6bf59abec3c4beb84886a9d",
    },
}


class FloatPowerBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(4,), (1024,), (65536,), (512, 512), (2048, 2048)]


class FloatPowerOutBenchmark(FloatPowerBenchmark):
    def _needs_external_reference(self):
        vendors = {
            "float_power_scalar_tensor_out": {"kunlunxin", "iluvatar"},
            "float_power_tensor_scalar_out": {
                "kunlunxin",
                "iluvatar",
                "ascend",
                "enflame",
            },
            "float_power_tensor_tensor_out": {"kunlunxin", "iluvatar"},
        }
        return flag_gems.vendor_name in vendors.get(self.op_name, ())

    def _require_external_reference(self, dtype, call, named_call=None):
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError(
                "This FloatPower out path requires matching H800 --base-data"
            )
        baseline = base.Config.base_data

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate H800 baseline JSON key")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite H800 baseline constant: {value}")

        data = json.loads(
            Path(baseline.source).read_bytes(),
            object_pairs_hook=unique_keys,
            parse_constant=reject_constant,
        )
        if data != baseline.data:
            raise ValueError("Loaded H800 baseline differs from its artifact")
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        if (
            data[self.op_name]["native_api"] != "ops." + self.op_name
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _H800_PRODUCER_COMMIT
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["repo"] != "pytorch"
            or not isinstance(environment["software"]["packages"]["torch"], str)
            or not environment["software"]["packages"]["torch"]
        ):
            raise ValueError(
                "A completed clean native H800 FloatPower out collection is required"
            )
        record, _, reason = baseline.lookup(
            self.op_name, dtype, call, named_call, target_device=self.device
        )
        if record is None or reason is not None:
            raise ValueError(
                "H800 FloatPower out record does not match the actual input"
            )
        spec = _OUT_BASELINE_RECIPES[self.op_name]
        tensors = [value for value in call["args"] if isinstance(value, dict)]
        shape = tensors[0]["shape"]
        if dtype not in consts.FLOAT_DTYPES or shape not in [
            [4],
            [1024],
            [65536],
            [512, 512],
            [2048, 2048],
        ]:
            raise ValueError("Only the original FloatPower out core grid is supported")
        stride = [1] if len(shape) == 1 else [shape[1], 1]
        tensor = {
            "shape": shape,
            "dtype": str(dtype),
            "stride": stride,
            "requires_grad": False,
        }
        expected_args = (
            [2.0, tensor]
            if self.op_name == "float_power_scalar_tensor_out"
            else (
                [tensor, 1.234]
                if self.op_name == "float_power_tensor_scalar_out"
                else [tensor, tensor]
            )
        )
        expected_call = {
            "args": expected_args,
            "kwargs": {"out": {**tensor, "dtype": "torch.float64"}},
        }
        required = {
            "source_commit": _PINNED_SOURCE_COMMIT,
            "benchmark": "benchmark/test_float_power.py",
            "descriptor": shape,
            "scope": "core",
            "input_contract": "float-power-real-double-out/v1",
            "input_recipe": spec["input_recipe"],
            "original_builder_sha256": spec["original_builder_sha256"],
            "target_builder_sha256": spec["target_builder_sha256"],
            "target_recipe_source_commit": _TARGET_RECIPE_SOURCE_COMMIT,
            "variant": 0,
        }
        config = record["config"]
        if (
            call != expected_call
            or config["inputs"] != call
            or set(config) != set(required) | {"inputs"}
            or any(config.get(key) != value for key, value in required.items())
        ):
            raise ValueError(
                "H800 FloatPower out recipe, overload or output layout differs"
            )
        latency_us, latency_ms = record.get("T_us"), record.get("latency_ms")
        if (
            not positive_number(latency_us)
            or not positive_number(latency_ms)
            or not math.isclose(latency_ms, latency_us / 1000, rel_tol=1e-9)
        ):
            raise ValueError(
                "H800 FloatPower out latency units must be positive and consistent"
            )
        return latency_ms

    def _measure_input(self, input, case_id=None):
        if self._needs_external_reference():
            args, kwargs = self.unpack_to_args_kwargs(input)
            dtype = next(
                value.dtype for value in args if isinstance(value, torch.Tensor)
            )
            self._require_external_reference(dtype, describe_call(args, kwargs))
        # The existing runner still executes the real candidate timer locally.
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        super()._apply_external_baseline(metric, dtype, context)
        if not self._needs_external_reference():
            return
        if context is None:
            raise ValueError("FloatPower out external reference context is missing")
        baseline, call, named_call = context
        latency_ms = self._require_external_reference(dtype, call, named_call)
        if (
            metric.error_msg is not None
            or metric.base_miss_reason is not None
            or metric.base_data_source != baseline.source
            or metric.base_latency_ms != latency_ms
            or not all(
                positive_number(value)
                for value in (
                    metric.latency,
                    metric.base_raw_speedup,
                    metric.compared_speedup,
                )
            )
        ):
            raise ValueError(
                "FloatPower out candidate timing or external comparison is incomplete"
            )
        factor = metric.base_scaling_factor
        if (
            factor is not None
            and not positive_number(factor)
            or not math.isclose(
                metric.base_raw_speedup, latency_ms / metric.latency, rel_tol=1e-9
            )
            or not math.isclose(
                metric.compared_speedup,
                metric.base_raw_speedup / (factor or 1.0),
                rel_tol=1e-9,
            )
        ):
            raise ValueError(
                "FloatPower out comparison ratios do not match the actual timings"
            )


def _simple_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _tensor_tensor_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    return base_tensor, exponent


def _tensor_scalar_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    return base_tensor, 1.234


def _scalar_tensor_build(plan, dtype, device):
    (shape,) = plan.builder_args
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    return 2.0, exponent


def _out_buffer(shape, device):
    # A direct Double factory can silently create Float on Kunlunxin. Preserve
    # the intended native Double output ABI without a typed conversion.
    numel = 1
    for dimension in shape:
        numel *= dimension
    return (
        torch.empty(numel * 8, dtype=torch.uint8, device=device)
        .view(torch.float64)
        .reshape(shape)
    )


def _tensor_tensor_out_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    out = _out_buffer(shape, device)
    return base_tensor, exponent, {"out": out}


def _tensor_scalar_out_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    out = _out_buffer(shape, device)
    return base_tensor, 1.234, {"out": out}


def _scalar_tensor_out_build(plan, dtype, device):
    (shape,) = plan.builder_args
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    out = _out_buffer(shape, device)
    return 2.0, exponent, {"out": out}


def _run_benchmark(op_name, build_inputs_fn, torch_op):
    bench = FloatPowerBenchmark(
        case_fn=_simple_case_fn,
        build_inputs_fn=build_inputs_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _run_out_benchmark(op_name, build_inputs_fn, torch_op):
    FloatPowerOutBenchmark(
        case_fn=_simple_case_fn,
        build_inputs_fn=build_inputs_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    ).run()


@pytest.mark.float_power_tensor_tensor
def test_float_power_tensor_tensor():
    _run_benchmark(
        "float_power_tensor_tensor",
        _tensor_tensor_build,
        torch.ops.aten.float_power.Tensor_Tensor,
    )


@pytest.mark.float_power_tensor_scalar
def test_float_power_tensor_scalar():
    _run_benchmark(
        "float_power_tensor_scalar",
        _tensor_scalar_build,
        torch.ops.aten.float_power.Tensor_Scalar,
    )


@pytest.mark.float_power_scalar_tensor
def test_float_power_scalar_tensor():
    _run_benchmark(
        "float_power_scalar_tensor",
        _scalar_tensor_build,
        torch.ops.aten.float_power.Scalar,
    )


@pytest.mark.float_power_tensor_tensor_out
@pytest.mark.skip_native(
    vendors=["kunlunxin", "iluvatar"],
    reason="native FloatPower Double out is unsupported; require matching H800 --base-data",
)
def test_float_power_tensor_tensor_out():
    _run_out_benchmark(
        "float_power_tensor_tensor_out",
        _tensor_tensor_out_build,
        torch.ops.aten.float_power.Tensor_Tensor_out,
    )


@pytest.mark.float_power_tensor_scalar_out
@pytest.mark.skip_native(
    vendors=["kunlunxin", "iluvatar", "ascend", "enflame"],
    reason="native FloatPower Double out is unsupported; require matching H800 --base-data",
)
def test_float_power_tensor_scalar_out():
    _run_out_benchmark(
        "float_power_tensor_scalar_out",
        _tensor_scalar_out_build,
        torch.ops.aten.float_power.Tensor_Scalar_out,
    )


@pytest.mark.float_power_scalar_tensor_out
@pytest.mark.skip_native(
    vendors=["kunlunxin", "iluvatar"],
    reason="native FloatPower Double out is unsupported; require matching H800 --base-data",
)
def test_float_power_scalar_tensor_out():
    _run_out_benchmark(
        "float_power_scalar_tensor_out",
        _scalar_tensor_out_build,
        torch.ops.aten.float_power.Scalar_out,
    )
