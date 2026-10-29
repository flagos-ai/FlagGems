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
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={"decimals": 0}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return utils.generate_tensor_input(shape, dtype, device), plan.params


_CURRENT_NATIVE_PRODUCER = "fe33a75f17271c0e8abc0dd50e47c4d58667aa2b"
_CURRENT_NATIVE_PROFILE = "current-decimals0"
_CURRENT_NATIVE_SHAPES = [
    [536870912],
    [64, 64],
    [4096, 4096],
    [64, 512, 512],
    [1024, 1024, 512],
]
_CURRENT_NATIVE_CONFIG = {
    "profile": "current-decimals0",
    "operator": "special_round",
    "benchmark_node": "benchmark/test_special_round.py::test_special_round",
    "grid_contract": "special_round.suiyuan.current-decimals0.core/v1",
    "native_api": "torch.special.round(input,decimals=0)",
    "seed_policy": "unseeded_original_builder_no_reset",
    "fresh_inputs": False,
    "latency_boundary": "native_decimals_overload",
    "producer_selector": {"OPS_BASELINE_SPECIAL_ROUND_CURRENT_CORE": "1"},
    "flag_gems_recipe_source": "2ff0931a646b97ec98ebd50642b5889619125a6b",
    "source_bindings": {
        "builder": "b42c345aaba1f33f5fe541195a792536dccfeab29be18c57c69eda8a9bb9b641",
        "case_fn": "f18d265600090a634059e6a0aaa426aa947f984f2665bef5aa322eeb15c1c0bc",
        "input_generator": "e991af3667ae315b99d8508a74c52528e478f5bc00ddd96eab06dad01b219495",
        "generated_framework_sha256": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
        "base_sha256": "daa8c109e43a06b03f34a14333135cf8cdc6efdbedab082eedb3bc4d7a3e3f5d",
        "backend_core_shapes_sha256": "69f31cd4f11f227449f7b57fad62f31dbd28730c19688c28cd127053e5f62ff1",
    },
}


class _CurrentNativeStaticReference:
    def _current_applies(self):
        return flag_gems.vendor_name == "enflame" and self.op_name == "special_round"

    def _current_config(self, shape, dtype):
        import copy

        if list(shape) not in _CURRENT_NATIVE_SHAPES or dtype not in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("Unknown original current native core cell")
        stride = []
        extent = 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size
        tensor = dict(
            shape=list(shape), dtype=str(dtype), stride=stride, requires_grad=False
        )
        result = copy.deepcopy(_CURRENT_NATIVE_CONFIG)
        result["inputs"] = {"args": [tensor], "kwargs": {"decimals": 0}}
        return result

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
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("Current native input arity differs")
        inp = args[0]
        target = torch.device(self.device)
        if inp.device.type != target.type or (
            target.type != "cpu"
            and inp.device.index != (target.index if target.index is not None else 0)
        ):
            raise ValueError("Actual input device differs from requested target")
        tensors = [inp]
        if (
            set(kwargs) != {"decimals"}
            or type(kwargs["decimals"]) is not int
            or kwargs["decimals"] != 0
        ):
            raise ValueError("Current Round requires exact decimals=0 overload")
        for value in tensors:
            if (
                value.device != inp.device
                or value.dtype != inp.dtype
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
        if (
            self.torch_op is not torch.special.round
            or self.is_backward
            or self.is_inplace is not False
            or getattr(self, "fresh_inputs", False) is not False
        ):
            raise ValueError("Native callable/fresh-input/timer boundary differs")
        module = inspect.getmodule(type(self))
        for name, expected in [
            ("_build_inputs_fn", _CURRENT_NATIVE_CONFIG["source_bindings"]["builder"]),
            ("_case_fn", _CURRENT_NATIVE_CONFIG["source_bindings"]["case_fn"]),
        ]:
            if (
                hashlib.sha256(
                    inspect.getsource(getattr(module, name)).strip().encode()
                ).hexdigest()
                != expected
            ):
                raise ValueError("Original input/case builder source differs")
        if (
            hashlib.sha256(
                inspect.getsource(utils.generate_tensor_input).strip().encode()
            ).hexdigest()
            != _CURRENT_NATIVE_CONFIG["source_bindings"]["input_generator"]
        ):
            raise ValueError("Original generator source differs")
        for path, key in [
            (
                Path(__file__).parent / "generated_operator_utils.py",
                "generated_framework_sha256",
            ),
            (Path(__file__).parent / "base.py", "base_sha256"),
            (
                Path(flag_gems.__file__).parent
                / "runtime/backend/_enflame/core_shapes.yaml",
                "backend_core_shapes_sha256",
            ),
        ]:
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != _CURRENT_NATIVE_CONFIG["source_bindings"][key]
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
        task = data["_collection"]["tasks"]["special_round"]
        env = data["_environments"][str(task["device"])]
        entry = data["special_round"]
        if (
            entry["native_api"] != "ops.special_round"
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
            "current-decimals0" + ":" + "x".join(map(str, shape))
            for shape in _CURRENT_NATIVE_SHAPES
        }
        if set(records) != expected_keys:
            raise ValueError("Incomplete exact original current core grid")
        for shape in _CURRENT_NATIVE_SHAPES:
            cells = records["current-decimals0" + ":" + "x".join(map(str, shape))]
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


class RoundBenchmark(
    _CurrentNativeStaticReference, base.UnaryPointwiseBenchmark, OperatorBenchmark
):
    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.special_round
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="native CPU fallback; require exact current H800 profile",
)
def test_special_round():
    RoundBenchmark(
        op_name="special_round",
        torch_op=torch.special.round,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
