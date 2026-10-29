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
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .baseline import describe_call, positive_number

_AMINMAX_H800_PRODUCER = "1a60bee68368cbdb2f5ada69796302c2fffeb583"
_AMINMAX_SOURCE_BINDINGS = {
    "direct": {
        "_case_fn": "bd835636353cb2703100478d9d32de51b0983fabe987d999284db5f34dbe6657",
        "_build_inputs_fn": "bd6712aeee3823008708ab3f0bf363021a0d7a84596a899ab4511190c666fbc5",
        "generator": "24e59f7572e9db79adb13460155847b72069bbd315db3093b2a10b4b1b9779d3",
    },
    "shared": {
        "get_case_iter": "8e7fcff16f63bbfaf19961471d4610ea25cb2c135a498e30684a49482a6cee1a",
        "build_inputs": "3edecbb308be85061151717090daf9f026ac51444853bd134558546bd678771f",
        "generator": "24e59f7572e9db79adb13460155847b72069bbd315db3093b2a10b4b1b9779d3",
    },
}
_AMINMAX_PROFILE_SHAPES = {
    "shared": [[1048576], [64, 64], [4096, 4096], [64, 512, 512], [1024, 1024, 512]],
    "direct": [
        [1073741824],
        [64, 64],
        [4096, 4096],
        [64, 512, 512],
        [1024, 1024, 1024],
    ],
}


class _AminmaxStaticReference:
    _H800_PROFILE = "shared"

    def _static_config(self, shape, dtype):
        PROFILE = self._H800_PROFILE
        DTYPES = consts.FLOAT_DTYPES
        _CURRENT_SOURCE = "2ff0931a646b97ec98ebd50642b5889619125a6b"
        _SOURCE_BINDINGS = _AMINMAX_SOURCE_BINDINGS
        _SELECT = "OPS_BASELINE_AMINMAX_CORE_PROFILE"
        shape = tuple(shape)
        if list(shape) not in _AMINMAX_PROFILE_SHAPES[PROFILE]:
            raise ValueError("Unknown original _aminmax descriptor")
        if dtype not in DTYPES:
            raise ValueError("Only the original three floating dtypes are supported")
        stride, extent = [], 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size
        inputs = {
            "args": [
                {
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "stride": stride,
                    "requires_grad": False,
                }
            ],
            "kwargs": {},
        }
        scalar = {
            "shape": [],
            "dtype": str(dtype),
            "stride": [],
            "requires_grad": False,
        }
        return {
            "inputs": inputs,
            "native_timer_contract": {
                "kind": "native_private_all_reduce",
                "native_schema": "aten::_aminmax.default",
                "argument_order": ["self"],
                "kwargs": {},
                "returns": {
                    "container": "tuple",
                    "items": [dict(scalar), dict(scalar)],
                },
                "setup_timed": False,
            },
            "random_policy": (
                "source_manual_seed0_then_randn_self"
                if PROFILE == "direct"
                else "source_unseeded_global_generator_no_reset"
            ),
            "input_recipe": (
                "manual_seed(0); generate(self)"
                if PROFILE == "direct"
                else "generate(self)"
            ),
            "benchmark": (
                "benchmark/test__aminmax.py"
                if PROFILE == "direct"
                else "benchmark/test_aminmax.py"
            ),
            "source_commit": _CURRENT_SOURCE,
            "source_test": "test__aminmax",
            "descriptor": list(shape),
            "scope": "core",
            "variant": 0,
            "grid_contract": f"_aminmax.{PROFILE}.core/v1",
            "producer_selector": {_SELECT: PROFILE},
            "source_bindings": dict(_SOURCE_BINDINGS[PROFILE]),
        }

    def _verify_static_recipe(self):
        if self.torch_op is not torch._aminmax or self.is_backward or self.is_inplace:
            raise ValueError("Original private _aminmax native timer contract differs")
        digest = lambda fn: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        expected = _AMINMAX_SOURCE_BINDINGS[self._H800_PROFILE]
        if digest(utils.generate_tensor_input) != expected["generator"]:
            raise ValueError("Original _aminmax input generator differs")
        if self._H800_PROFILE == "shared":
            for name in ("get_case_iter", "build_inputs"):
                if digest(getattr(type(self), name)) != expected[name]:
                    raise ValueError("Original shared _aminmax source differs: " + name)
        else:
            module = inspect.getmodule(type(self))
            for name in ("_case_fn", "_build_inputs_fn"):
                if digest(getattr(module, name)) != expected[name]:
                    raise ValueError("Original direct _aminmax source differs: " + name)

    def _require_external_reference(self, args, kwargs):
        if (
            self.op_name != "_aminmax"
            or not base.Config.skip_native
            or base.Config.base_data is None
        ):
            raise ValueError(
                "CPU-fallback private _aminmax requires its exact H800 --base-data"
            )
        if len(args) != 1 or kwargs or not isinstance(args[0], torch.Tensor):
            raise ValueError(
                "Original private _aminmax requires exactly one Tensor and no kwargs"
            )
        inp = args[0]
        shapes = _AMINMAX_PROFILE_SHAPES[self._H800_PROFILE]
        if list(inp.shape) not in shapes or inp.dtype not in consts.FLOAT_DTYPES:
            raise ValueError("Unknown original _aminmax core dtype/shape")
        if (
            inp.device != torch.empty(0, device=self.device).device
            or inp.layout != torch.strided
            or not inp.is_contiguous()
            or inp.storage_offset() != 0
            or inp.requires_grad
            or not inp.is_leaf
            or inp.is_conj()
            or inp.is_neg()
        ):
            raise ValueError(
                "Original _aminmax input device/layout/grad/lazy contract differs"
            )
        self._verify_static_recipe()
        baseline = base.Config.base_data
        payload = Path(baseline.source).read_bytes()

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate H800 aminmax JSON key")
                result[key] = value
            return result

        def constant(value):
            raise ValueError("Nonfinite H800 aminmax JSON constant: " + value)

        canonical = lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        data = json.loads(payload, object_pairs_hook=unique, parse_constant=constant)
        if canonical(data) != canonical(baseline.data):
            raise ValueError("Loaded aminmax baseline differs from source artifact")
        task = data["_collection"]["tasks"]["_aminmax"]
        env = data["_environments"][str(task["device"])]
        entry = data["_aminmax"]
        if (
            entry["native_api"] != "ops._aminmax"
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != _AMINMAX_H800_PRODUCER
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
            or not isinstance(env["software"]["packages"]["torch"], str)
            or not env["software"]["packages"]["torch"]
        ):
            raise ValueError(
                "A completed clean native H800 aminmax profile is required"
            )
        records = entry["shapes"]
        if set(records) != {str(shape) for shape in shapes}:
            raise ValueError("Incomplete original five-shape H800 profile")
        for shape in shapes:
            cells = records[str(shape)]
            if set(cells) != {str(dtype) for dtype in consts.FLOAT_DTYPES}:
                raise ValueError("Incomplete original three-dtype H800 profile")
            for dtype in consts.FLOAT_DTYPES:
                cell = cells[str(dtype)]
                if canonical(cell["config"]) != canonical(
                    self._static_config(shape, dtype)
                ):
                    raise ValueError("Native aminmax timer/input/RNG profile differs")
                if (
                    not positive_number(cell.get("T_us"))
                    or not positive_number(cell.get("latency_ms"))
                    or not math.isclose(
                        cell["latency_ms"],
                        cell["T_us"] / 1000,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError("Native H800 aminmax latency units differ")
        call = describe_call(args, kwargs)
        record, key, reason = baseline.lookup(
            self.op_name, inp.dtype, call, target_device=self.device
        )
        expected = self._static_config(list(inp.shape), inp.dtype)
        if (
            record is None
            or reason is not None
            or canonical(record["config"]) != canonical(expected)
            or canonical(call) != canonical(expected["inputs"])
        ):
            raise ValueError(
                "Actual case does not match the selected H800 aminmax profile"
            )
        return (
            baseline,
            canonical(call),
            record["latency_ms"],
            key,
            hashlib.sha256(payload).hexdigest(),
        )

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name == "enflame":
            self._aminmax_external_checked = self._require_external_reference(
                *self.unpack_to_args_kwargs(input)
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if flag_gems.vendor_name != "enflame":
            return super()._apply_external_baseline(metric, dtype, context)
        try:
            if context is None or context[1] is None:
                raise ValueError("External aminmax reference context missing")
            baseline, call, named_call = context
            checked, captured, latency, key, artifact = self._aminmax_external_checked
            if (
                baseline is not checked
                or json.dumps(call, sort_keys=True, separators=(",", ":")) != captured
                or hashlib.sha256(Path(baseline.source).read_bytes()).hexdigest()
                != artifact
            ):
                raise ValueError(
                    "External aminmax reference changed around candidate timing"
                )
            super()._apply_external_baseline(metric, dtype, context)
            factor = metric.base_scaling_factor
            if (
                metric.error_msg is not None
                or metric.latency_base is not None
                or metric.base_miss_reason is not None
                or metric.base_data_source != baseline.source
                or metric.base_shape_key != key
                or metric.base_latency_ms != latency
                or not all(
                    positive_number(value)
                    for value in [
                        metric.latency,
                        metric.base_raw_speedup,
                        metric.compared_speedup,
                    ]
                )
                or factor is not None
                and not positive_number(factor)
                or not math.isclose(
                    metric.base_raw_speedup, latency / metric.latency, rel_tol=1e-9
                )
                or not math.isclose(
                    metric.compared_speedup,
                    metric.base_raw_speedup / (factor or 1.0),
                    rel_tol=1e-9,
                )
            ):
                raise ValueError(
                    "Candidate aminmax timing or H800 comparison is incomplete"
                )
        finally:
            self._aminmax_external_checked = None


def _aminmax_variants(shape):
    # Test dim=None (whole tensor reduction); dim=-1; and dim=0 when ndim > 1.
    variants = [{}, {"dim": -1}]
    if len(shape) > 1:
        variants.append({"dim": 0})
    return variants


def _aminmax_build(shape, kwargs, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    if kwargs:
        return inp, kwargs
    return (inp,)


class AminmaxBenchmark(base.UnaryReductionBenchmark):
    def get_case_iter(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            for kwargs in _aminmax_variants(shape):
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": shape},
                        params=dict(kwargs),
                        builder_args=(shape, kwargs),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, kwargs = plan.builder_args
        return _aminmax_build(shape, kwargs, case.dtype, self.device)


@pytest.mark.aminmax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_aminmax():
    bench = AminmaxBenchmark(
        op_name="aminmax",
        torch_op=torch.aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


# ---------------------------------------------------------------------------
# aten::_aminmax / aten::_aminmax.out
#
# Operator ids stay `_aminmax` / `_aminmax_out`; the pytest markers are spelled
# `underscore_aminmax` / `underscore_aminmax_out` because pytest will not build a
# marker from an attribute beginning with an underscore.
# ---------------------------------------------------------------------------


class AminmaxAllReduceBenchmark(_AminmaxStaticReference, base.UnaryReductionBenchmark):
    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        # ``_aminmax`` reduces the whole tensor; there is no dim argument.
        return (inp,)


class AminmaxOutBenchmark(base.UnaryReductionBenchmark):
    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        cur_dtype = case.dtype
        inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
        min_out = torch.empty((), dtype=cur_dtype, device=self.device)
        max_out = torch.empty((), dtype=cur_dtype, device=self.device)
        # The aten ``.out`` overload takes keyword-only ``out0``/``out1``.
        return inp, {"out0": min_out, "out1": max_out}


@pytest.mark.underscore_aminmax
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="native private _aminmax is CPU fallback; require exact shared H800 profile",
)
def test__aminmax():
    bench = AminmaxAllReduceBenchmark(
        op_name="_aminmax",
        torch_op=torch._aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_aminmax_out
def test__aminmax_out():
    bench = AminmaxOutBenchmark(
        op_name="_aminmax_out",
        torch_op=torch.ops.aten._aminmax.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
