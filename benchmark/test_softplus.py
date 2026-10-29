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

from . import base, consts
from .baseline import describe_call, positive_number

_SOFTPLUS_H800_PRODUCER = "274ce58a685f019b40ebffb59710e1e251fc6d0a"
_SOFTPLUS_MAIN_SHAPES = [
    [536870912],
    [64, 64],
    [4096, 4096],
    [64, 512, 512],
    [1024, 1024, 512],
]
_SOFTPLUS_DIRECT_SHAPES = [
    [1073741824],
    [64, 64],
    [4096, 4096],
    [64, 512, 512],
    [1024, 1024, 1024],
]
_SOFTPLUS_SOURCE_BINDINGS = {
    "main": {
        "adapter_source_snapshot": "4a0724463fe32c5fd5641413c5f7ad8b02ca8fa7",
        "backend_core_shapes_sha256": "69f31cd4f11f227449f7b57fad62f31dbd28730c19688c28cd127053e5f62ff1",
        "current_adapter_builder": "50844457a2bfabed3f3d2bd842c794f13da30b70bb66ca7f7ec983a1a25a3a8a",
        "current_adapter_callable": "94cafb82b3a9ef4635c3502d7239c82f7a0b31f4c064c1a2f443db2a8fd83a07",
        "current_adapter_dout": "fcd15ba4d207b3aed0d380003fdf7c1dfab56494986750d4baa6b53354813b11",
        "current_base_callable": "95acdf392196ee81ac1c822226f179f9cb434b014a48e2786d3390f13d4578da",
        "current_builder": "93625f7fbccd258d315a7148b445bb25bec81d3c868c899bab82938cf05198b8",
        "current_generator": "67b686d91b6ec438eea768c81b2ae154182be8a52a9939d0c57af8c9aa10a79b",
        "current_unpack": "3b1a1bbd45f98219a8b2126f367fa67e8b7075f29cf5a12a3c7ebeba72904e8c",
    },
    "direct": {
        "direct_builder": "520400ce881f6b15ed6cbd26c5c5ca8011e7c23d179a885670b5a0de2d04e4ea",
        "direct_casefn": "608d3a609644e739d87402a137d6d98e99a6e11dcf0c3cc70f700c1a8e817957",
        "generic_framework_source_sha256": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
    },
}


class _SoftplusStaticReference:
    _H800_PROFILE = "main"

    def _static_config(self, shape, dtype):
        stride, extent = [], 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size

        def tensor(requires_grad=False):
            return dict(
                shape=list(shape),
                dtype=str(dtype),
                stride=stride.copy(),
                requires_grad=requires_grad,
            )

        if self._H800_PROFILE == "main":
            timed = {"args": [tensor(True), tensor(True), tensor(False)], "kwargs": {}}
            logical = {"args": [tensor(True)], "kwargs": {}}
            config = dict(
                inputs=logical.copy(),
                native_timed_inputs=timed,
                logical_case_inputs=logical,
                native_timer_contract=dict(
                    kind="prepared_autograd_backward",
                    forward="torch.nn.functional.softplus",
                    backward="torch.autograd.grad",
                    argument_order=["forward_output", "self", "grad_output"],
                    beta=1,
                    threshold=20,
                    retain_graph=True,
                    create_graph=False,
                    returns={"container": "tuple", "items": [tensor(False)]},
                    graph=dict(
                        self_is_leaf=True,
                        forward_output_is_leaf=False,
                        forward_output_has_grad_fn=True,
                        forward_output_created_from_this_self=True,
                    ),
                    setup_timed=False,
                ),
                random_policy="source_unseeded_global_generator_no_reset",
                input_recipe="generate(self); clone.requires_grad; softplus; randn_like(out)",
                benchmark="benchmark/test_softplus.py",
            )
            contract = "softplus-backward-main-logical-self-to-prepared-native/v1"
        elif self._H800_PROFILE == "direct":
            timed = {"args": [tensor(), tensor(), 1.5, 20.0], "kwargs": {}}
            config = dict(
                inputs=timed.copy(),
                native_timed_inputs=timed,
                logical_case_inputs=timed.copy(),
                native_timer_contract=dict(
                    kind="direct_aten_backward",
                    native_schema="aten::softplus_backward.default",
                    beta=1.5,
                    threshold=20.0,
                    returns={"container": "Tensor", "metadata": tensor()},
                    setup_timed=False,
                ),
                random_policy="source_manual_seed0_then_randn_grad_then_randn_self",
                input_recipe="manual_seed(0); generate(grad_output); generate(self)",
                benchmark="benchmark/test_softplus_backward.py",
            )
            contract = "softplus-backward-direct/v1"
        else:
            raise ValueError("Unknown native softplus timer profile")
        config.update(
            source_commit="2ff0931a646b97ec98ebd50642b5889619125a6b",
            source_test="test_softplus_backward",
            descriptor=list(shape),
            scope="core",
            variant=0,
            grid_contract=f"softplus_backward.{self._H800_PROFILE}.core/v1",
            input_contract=contract,
            producer_selector={
                "OPS_BASELINE_SOFTPLUS_BACKWARD_CORE_PROFILE": self._H800_PROFILE
            },
            source_bindings=_SOFTPLUS_SOURCE_BINDINGS[self._H800_PROFILE],
        )
        return config

    def _verify_static_recipe(self):
        bindings = _SOFTPLUS_SOURCE_BINDINGS[self._H800_PROFILE]
        digest = lambda value: hashlib.sha256(
            inspect.getsource(value).lstrip().rstrip().encode()
        ).hexdigest()
        if self._H800_PROFILE == "main":
            checks = {
                "current_generator": base.generate_tensor_input,
                "current_builder": base.UnaryPointwiseBenchmark.build_inputs,
                "current_unpack": base.Benchmark.unpack_to_args_kwargs,
                "current_base_callable": base.Benchmark._benchmark_callable,
                "current_adapter_callable": type(self)._benchmark_callable,
                "current_adapter_dout": type(self)._prepared_dout,
                "current_adapter_builder": type(self).build_inputs,
            }
            for key, value in checks.items():
                if digest(value) != bindings[key]:
                    raise ValueError("Current prepared softplus source differs: " + key)
            shape_file = (
                Path(flag_gems.__file__).parent
                / "runtime/backend/_enflame/core_shapes.yaml"
            )
            if (
                hashlib.sha256(shape_file.read_bytes()).hexdigest()
                != bindings["backend_core_shapes_sha256"]
            ):
                raise ValueError("Current Enflame softplus core shapes differ")
            if (
                self.torch_op is not torch.nn.functional.softplus
                or not self.is_backward
            ):
                raise ValueError(
                    "Prepared softplus forward/backward timer contract differs"
                )
        else:
            module = inspect.getmodule(type(self))
            for key, name in [
                ("direct_builder", "_build_inputs_fn"),
                ("direct_casefn", "_case_fn"),
            ]:
                if digest(getattr(module, name)) != bindings[key]:
                    raise ValueError("Current direct softplus source differs: " + key)
            if (
                hashlib.sha256(
                    (Path(__file__).parent / "generated_operator_utils.py").read_bytes()
                ).hexdigest()
                != bindings["generic_framework_source_sha256"]
            ):
                raise ValueError("Current direct benchmark framework differs")
            if (
                self.torch_op is not torch.ops.aten.softplus_backward
                or self.is_backward
            ):
                raise ValueError("Direct softplus native callable differs")

    def _require_external_reference(self, args, kwargs):
        if (
            self.op_name != "softplus_backward"
            or not base.Config.skip_native
            or base.Config.base_data is None
        ):
            raise ValueError(
                "CPU-fallback softplus requires its exact H800 --base-data"
            )
        profile = self._H800_PROFILE
        expected_arity = 1 if profile == "main" else 4
        if len(args) != expected_arity or kwargs:
            raise ValueError("Native softplus profile argument count differs")
        tensors = args[:1] if profile == "main" else args[:2]
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("Softplus inputs must be tensors")
        inp = tensors[0] if profile == "main" else tensors[1]
        shapes = _SOFTPLUS_MAIN_SHAPES if profile == "main" else _SOFTPLUS_DIRECT_SHAPES
        if list(inp.shape) not in shapes or inp.dtype not in consts.FLOAT_DTYPES:
            raise ValueError("Unknown original softplus core dtype/shape")
        for value in tensors:
            if (
                value.device != inp.device
                or value.dtype != inp.dtype
                or value.shape != inp.shape
                or value.layout != torch.strided
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.requires_grad != (profile == "main")
                or not value.is_leaf
                or value.is_conj()
                or value.is_neg()
            ):
                raise ValueError(
                    "Softplus input dtype/device/shape/layout/grad contract differs"
                )
        if profile == "direct":
            if (
                type(args[2]) is not float
                or type(args[3]) is not float
                or tuple(args[2:]) != (1.5, 20.0)
            ):
                raise ValueError("Direct beta/threshold differs")
            a, b = tensors
            sa, sb = a.untyped_storage(), b.untyped_storage()
            if sa._cdata == sb._cdata or not (
                sa.data_ptr() + sa.nbytes() <= sb.data_ptr()
                or sb.data_ptr() + sb.nbytes() <= sa.data_ptr()
            ):
                raise ValueError("Direct inputs must have independent storage")
        self._verify_static_recipe()
        baseline = base.Config.base_data
        payload = Path(baseline.source).read_bytes()

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate H800 softplus JSON key")
                result[key] = value
            return result

        def constant(value):
            raise ValueError("Nonfinite H800 softplus JSON constant: " + value)

        canonical = lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        data = json.loads(payload, object_pairs_hook=unique, parse_constant=constant)
        if canonical(data) != canonical(baseline.data):
            raise ValueError("Loaded softplus baseline differs from source artifact")
        task = data["_collection"]["tasks"]["softplus_backward"]
        env = data["_environments"][str(task["device"])]
        entry = data["softplus_backward"]
        if (
            entry["native_api"] != "ops.softplus_backward"
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != _SOFTPLUS_H800_PRODUCER
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
            or not isinstance(env["software"]["packages"]["torch"], str)
            or not env["software"]["packages"]["torch"]
        ):
            raise ValueError(
                "A completed clean native H800 softplus profile is required"
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
                    raise ValueError("Native softplus timer/input/RNG profile differs")
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
                    raise ValueError("Native H800 softplus latency units differ")
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
                "Actual case does not match the selected H800 softplus profile"
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
            self._softplus_external_checked = self._require_external_reference(
                *self.unpack_to_args_kwargs(input)
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if flag_gems.vendor_name != "enflame":
            return super()._apply_external_baseline(metric, dtype, context)
        try:
            if context is None or context[1] is None:
                raise ValueError("External softplus reference context missing")
            baseline, call, named_call = context
            checked, captured, latency, key, artifact = self._softplus_external_checked
            if (
                baseline is not checked
                or json.dumps(call, sort_keys=True, separators=(",", ":")) != captured
                or hashlib.sha256(Path(baseline.source).read_bytes()).hexdigest()
                != artifact
            ):
                raise ValueError(
                    "External softplus reference changed around candidate timing"
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
                    "Candidate softplus timing or H800 comparison is incomplete"
                )
        finally:
            self._softplus_external_checked = None


@pytest.mark.softplus
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softplus():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus",
        torch_op=torch.nn.functional.softplus,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class SoftplusBackwardBenchmark(_SoftplusStaticReference, base.UnaryPointwiseBenchmark):
    def build_inputs(self, case):
        if flag_gems.vendor_name == "kunlunxin" and case.dtype == torch.bfloat16:
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("native BF16 softplus backward is not implemented on Kunlunxin")
        return super().build_inputs(case)

    def _has_backward_override(self):
        registry = base.Config.override_registry
        return registry is not None and registry.get_override(self.op_name) is not None

    def _prepared_dout(self, inp, out=None):
        if getattr(self, "_prepared_backward_input", None) is not inp:
            if out is None:
                out = self.torch_op(inp)
            self._prepared_backward_input = inp
            self._prepared_backward_dout = torch.randn_like(out)
        return self._prepared_backward_dout

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if not overridden:
            return op, dispatch, overridden

        def backward_from_case(inp):
            return op(self._prepared_dout(inp), inp, 1, 20)

        backward_from_case._raw_softplus_backward = op
        return backward_from_case, dispatch, overridden

    def _benchmark_callable(self, op, *args, **kwargs):
        if not self._has_backward_override():
            return super()._benchmark_callable(op, *args, **kwargs)
        assert len(args) == 1 and not kwargs
        (inp,) = args
        if op is self.torch_op:
            out = self.torch_op(inp)
            dout = self._prepared_dout(inp, out)
            return lambda: torch.autograd.grad(
                (out,), (inp,), grad_outputs=(dout,), retain_graph=True
            ), [inp]
        candidate = getattr(op, "_raw_softplus_backward", op)
        dout = self._prepared_dout(inp)
        return lambda: candidate(dout, inp, 1, 20), [inp]


@pytest.mark.softplus_backward
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="native softplus backward is CPU fallback; require exact main H800 profile",
)
def test_softplus_backward():
    bench = SoftplusBackwardBenchmark(
        op_name="softplus_backward",
        torch_op=torch.nn.functional.softplus,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
