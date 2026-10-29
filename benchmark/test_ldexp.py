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

from . import base, consts
from .baseline import describe_call, positive_number
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={"exponent_dtype": "int32"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    return inp, exponent


_CURRENT_NATIVE_PRODUCER = "e4415155be3c6930767b5743250e2866bde7fee0"
_CURRENT_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]
_CURRENT_SOURCE_BINDINGS = {
    "_case_fn": "76d3b145ad3ba1a2dfb50d7d39e1c2c0eb2985fb58099689e578bbb1403d26af",
    "_build_inputs_fn": "1b3e7d963b530b11549967ce595fbd2e32c79375536b1ca53e6e12937b713f22",
    "_out_build_inputs_fn": "69bf0ff0d5a5d81bbde435157880bffe1fdd504b8bbb66adbe4494b2d25738e3",
    "benchmark/generated_operator_utils.py": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
    "benchmark/base.py": "daa8c109e43a06b03f34a14333135cf8cdc6efdbedab082eedb3bc4d7a3e3f5d",
    "benchmark/core_shapes.yaml": "720ad7776ce6e172e6c6eb78ee94b84b243ec7ee84a295543ac76fddd63b664d",
    "src/flag_gems/runtime/backend/_enflame/core_shapes.yaml": (
        "69f31cd4f11f227449f7b57fad62f31db" "d28730c19688c28cd127053e5f62ff1"
    ),
}


class _CurrentNativeStaticReference:
    def _current_applies(self):
        return base.vendor_name == "enflame" and self.op_name == "ldexp_out"

    def _current_config(self, shape, dtype):
        shape = tuple(shape)
        if shape not in _CURRENT_SHAPES or dtype not in consts.FLOAT_DTYPES:
            raise ValueError("Unknown original ldexp_out current core cell")
        stride, extent = ([], 1)
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size

        def tensor(dt):
            return dict(
                shape=list(shape),
                dtype=str(dt),
                stride=list(stride),
                requires_grad=False,
            )

        return dict(
            profile="current-seed0-out",
            grid_contract="ldexp_out.suiyuan.current-seed0.core/v1",
            operator="ldexp_out",
            benchmark_node="benchmark/test_ldexp.py::test_ldexp_out",
            original_case_ordinal=_CURRENT_SHAPES.index(shape),
            original_case_params={"exponent_dtype": "int32"},
            flag_gems_recipe_source="2ff0931a646b97ec98ebd50642b5889619125a6b",
            producer_selector={"OPS_BASELINE_LDEXP_OUT_CURRENT_CORE": "1"},
            source_bindings=dict(_CURRENT_SOURCE_BINDINGS),
            inputs=dict(
                args=[tensor(dtype), tensor(torch.int32)], kwargs={"out": tensor(dtype)}
            ),
            input_recipe="manual_seed(0);randn(self);randint(-8,9,int32);empty_like(out)",
            out_initial_values_undefined=True,
            native_timer_contract=dict(
                native_schema="aten::ldexp.out",
                argument_order=["self", "other"],
                keyword_tensor="out",
                returns_original_out=True,
                mutates=["out"],
                fresh_inputs=False,
                setup_timed=False,
                original_is_inplace_flag=True,
            ),
        )

    def _require_current_reference(self, args, kwargs):
        if len(_CURRENT_NATIVE_PRODUCER) != 40 or any(
            c not in "0123456789abcdef" for c in _CURRENT_NATIVE_PRODUCER
        ):
            raise ValueError(
                "Unresolved producer commit: draft cannot consume real data"
            )
        if not self._current_applies():
            raise ValueError("Wrong current native operator/vendor route")
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError("CPU fallback requires exact H800 --base-data")
        if len(args) != 2 or not all(isinstance(v, torch.Tensor) for v in args):
            raise ValueError("Current native input arity differs")
        inp, exponent = args
        if set(kwargs) != {"out"} or not isinstance(kwargs["out"], torch.Tensor):
            raise ValueError("Current native requires original tensor out")
        out = kwargs["out"]
        target = torch.device(self.device)
        if inp.device.type != target.type or (
            target.type != "cpu"
            and inp.device.index != (target.index if target.index is not None else 0)
        ):
            raise ValueError("Actual input device differs from requested target")
        for value, dtype in (
            (inp, inp.dtype),
            (exponent, torch.int32),
            (out, inp.dtype),
        ):
            if (
                value.device != inp.device
                or value.dtype != dtype
                or value.shape != inp.shape
                or value.layout != torch.strided
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.requires_grad
                or not value.is_leaf
                or value.is_conj()
                or value.is_neg()
            ):
                raise ValueError("Current native input/out ABI differs")
        # The original seeded builder allocates all three buffers independently.
        # Distinct StorageImpl wrappers alone do not prove that physical ranges differ.
        values = (inp, exponent, out)
        for index, value in enumerate(values):
            left = value.untyped_storage()
            for other in values[:index]:
                right = other.untyped_storage()
                if left._cdata == right._cdata or (
                    left.nbytes() > 0
                    and right.nbytes() > 0
                    and left.data_ptr() < right.data_ptr() + right.nbytes()
                    and right.data_ptr() < left.data_ptr() + left.nbytes()
                ):
                    raise ValueError(
                        "Original current input buffers must be independent"
                    )
        if (
            self.torch_op is not torch.ops.aten.ldexp.out
            or self.is_backward
            or self.is_inplace is not True
            or getattr(self, "fresh_inputs", False) is not False
        ):
            raise ValueError("Native callable/fresh-input/timer boundary differs")
        module = inspect.getmodule(type(self))
        for name in ("_case_fn", "_build_inputs_fn", "_out_build_inputs_fn"):
            if (
                hashlib.sha256(
                    (inspect.getsource(getattr(module, name)).rstrip() + "\n").encode()
                ).hexdigest()
                != _CURRENT_SOURCE_BINDINGS[name]
            ):
                raise ValueError("Original input/case builder source differs")
        product_root = Path(__file__).resolve().parents[1]
        for name, expected in _CURRENT_SOURCE_BINDINGS.items():
            if "/" not in name:
                continue
            if (
                hashlib.sha256((product_root / name).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError("Original core/timer framework source differs")
        baseline = base.Config.base_data
        payload = Path(baseline.source).read_bytes()

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate H800 JSON key")
                result[key] = value
            return result

        def nonfinite(value):
            raise ValueError("Nonfinite H800 JSON constant")

        canonical = lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        data = json.loads(payload, object_pairs_hook=unique, parse_constant=nonfinite)
        if canonical(data) != canonical(baseline.data):
            raise ValueError("Loaded static source differs")
        task = data["_collection"]["tasks"]["ldexp_out"]
        env = data["_environments"][str(task["device"])]
        entry = data["ldexp_out"]
        if (
            entry["native_api"] != "ops.ldexp_out"
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != _CURRENT_NATIVE_PRODUCER
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
            or not isinstance(env["software"]["packages"]["torch"], str)
            or not env["software"]["packages"]["torch"]
        ):
            raise ValueError("Completed exact clean H800 native source required")
        records = entry["shapes"]
        expected_keys = {
            "current-seed0-out" + ":" + "x".join(map(str, shape))
            for shape in _CURRENT_SHAPES
        }
        if set(records) != expected_keys:
            raise ValueError("Incomplete exact original current core grid")
        for shape in _CURRENT_SHAPES:
            cells = records["current-seed0-out" + ":" + "x".join(map(str, shape))]
            if set(cells) != {
                str(dtype) for dtype in (torch.float16, torch.float32, torch.bfloat16)
            }:
                raise ValueError("Incomplete exact three-dtype native grid")
            for dtype in (torch.float16, torch.float32, torch.bfloat16):
                cell = cells[str(dtype)]
                if canonical(cell["config"]) != canonical(
                    self._current_config(shape, dtype)
                ):
                    raise ValueError(
                        "Native profile/source/input/timer metadata differs"
                    )
                if (
                    not positive_number(cell.get("latency_ms"))
                    or not positive_number(cell.get("T_us"))
                    or not math.isclose(
                        cell["latency_ms"],
                        cell["T_us"] / 1000,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError("Invalid native latency units")
        call = describe_call(args, kwargs)
        record, key, reason = baseline.lookup(
            self.op_name, inp.dtype, call, target_device=self.device
        )
        if (
            record is None
            or reason is not None
            or canonical(record["config"])
            != canonical(self._current_config(list(inp.shape), inp.dtype))
            or canonical(call)
            != canonical(self._current_config(list(inp.shape), inp.dtype)["inputs"])
        ):
            raise ValueError(
                "Actual input does not select the exact current native record"
            )
        return (
            baseline,
            canonical(call),
            record["latency_ms"],
            key,
            hashlib.sha256(payload).hexdigest(),
        )

    def _measure_input(self, input, case_id=None):
        if self._current_applies():
            self._current_checked = self._require_current_reference(
                *self.unpack_to_args_kwargs(input)
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if not self._current_applies():
            return super()._apply_external_baseline(metric, dtype, context)
        try:
            if context is None or context[1] is None:
                raise ValueError("Static context missing")
            baseline, call, _ = context
            checked, captured, latency, key, digest = self._current_checked
            if (
                baseline is not checked
                or json.dumps(call, sort_keys=True, separators=(",", ":")) != captured
                or hashlib.sha256(Path(baseline.source).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError("Static source changed around candidate timer")
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
                    positive_number(x)
                    for x in [
                        metric.latency,
                        metric.base_raw_speedup,
                        metric.compared_speedup,
                    ]
                )
                or (factor is not None and not positive_number(factor))
                or not math.isclose(
                    metric.base_raw_speedup, latency / metric.latency, rel_tol=1e-9
                )
                or not math.isclose(
                    metric.compared_speedup,
                    metric.base_raw_speedup / (factor or 1.0),
                    rel_tol=1e-9,
                )
            ):
                raise ValueError("Candidate/static metric incomplete")
        finally:
            self._current_checked = None


class LdexpBenchmark(
    _CurrentNativeStaticReference, base.BinaryPointwiseBenchmark, OperatorBenchmark
):
    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.ldexp
def test_ldexp():
    bench = LdexpBenchmark(
        op_name="ldexp",
        torch_op=torch.ops.aten.ldexp.Tensor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _out_build_inputs_fn(plan, dtype, device):
    inp, exponent = _build_inputs_fn(plan, dtype, device)
    out = torch.empty_like(inp)
    return inp, exponent, {"out": out}


@pytest.mark.ldexp_out
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="Floating self/integer exponent kernel missing; require exact H800 current15",
)
def test_ldexp_out():
    LdexpBenchmark(
        op_name="ldexp_out",
        torch_op=torch.ops.aten.ldexp.out,
        case_fn=_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    ).run()
