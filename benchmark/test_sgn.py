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

import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .baseline import BaselineData, describe_call, positive_number
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "out": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    return (utils.generate_tensor_input(shape, dtype, device),)


def _out_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    return inp, {"out": out}


_SGN_CURRENT_SHAPES = {
    "current-shared": [
        [536870912],
        [64, 64],
        [4096, 4096],
        [64, 512, 512],
        [1024, 1024, 512],
    ],
    "current-fresh": [
        [1073741824],
        [64, 64],
        [4096, 4096],
        [64, 512, 512],
        [1024, 1024, 1024],
    ],
}

_SGN_BASE_CONFIGS = {
    "current-shared": {
        "profile": "current-shared",
        "grid_contract": "sgn_out.suiyuan.current-shared.core/v1",
        "operator": "sgn_out",
        "benchmark_node": "benchmark/test_sgn.py::test_sgn_out",
        "original_case_ordinal": 1,
        "original_case_params": {},
        "flag_gems_recipe_source": "2ff0931a646b97ec98ebd50642b5889619125a6b",
        "original_target_catalog_sha256": "1b94eaf06aeb416d9c2c4102939ba7aa062e801144f575586310b71b13ff7bad",
        "source_bindings": {
            "_out_case_fn": "3f65cd08d0921dfc673a999e425f51e6346e807f1fb2722d5d3f4927e7ffa895",
            "_out_build_inputs_fn": "6a78fbb0faab3bfc6e1736df2219e3487ff6eb7bcd67ea4be8cef37aac520d4c",
            "source_file_sha256": "943623da48c152b78ecbbbbe86a1468e477e3dd9e316de5e8b1c17d946425edb",
        },
        "clone_source_sha256": None,
        "native_api": "torch.sgn(input,out=out)",
        "seed_policy": "manual_seed(0)",
        "fresh_inputs": False,
        "out_initial_values_undefined": True,
        "input_recipe": "generate_tensor_input;empty(shape,dtype,device)",
        "native_timer_contract": {
            "native_schema": "aten::sgn.out",
            "returns_sample_out": True,
            "mutates": ["out"],
            "fresh_inputs": False,
            "setup_timed": False,
            "preparation": "none_reused_buffers",
            "clock": "unchanged_triton_do_bench",
            "warmup_GPU_ms": 25,
            "measured_GPU_ms": 100,
            "NCU_supported": True,
        },
        "producer_selector": {"OPS_BASELINE_SGN_OUT_PROFILE": "current-shared"},
        "inputs": {
            "args": [
                {
                    "shape": [64, 64],
                    "dtype": "torch.float16",
                    "stride": [64, 1],
                    "requires_grad": False,
                }
            ],
            "kwargs": {
                "out": {
                    "shape": [64, 64],
                    "dtype": "torch.float16",
                    "stride": [64, 1],
                    "requires_grad": False,
                }
            },
        },
    },
    "current-fresh": {
        "profile": "current-fresh",
        "grid_contract": "sgn_out.suiyuan.current-fresh.core/v1",
        "operator": "sgn_out",
        "benchmark_node": "benchmark/test_sgn_out.py::test_sgn_out",
        "original_case_ordinal": 1,
        "original_case_params": {},
        "flag_gems_recipe_source": "2ff0931a646b97ec98ebd50642b5889619125a6b",
        "original_target_catalog_sha256": "1b94eaf06aeb416d9c2c4102939ba7aa062e801144f575586310b71b13ff7bad",
        "source_bindings": {
            "_case_fn": "88414178e65eec0d2b4866562c6febf729f23af1ac6d253e287fbfcbb6858448",
            "_build_inputs_fn": "c1f400adff2960eda451a89cc840e9cb17b179d8c41221f5eac906a6d5fd9fbf",
            "source_file_sha256": "bd6e99a5b4eb510bd66eb8c9ce3bc223cd2d866ec4d60beeb1b7259f73cbc9e9",
        },
        "clone_source_sha256": "ca5f7cad847f6c0b9e1e9d8984bb987f634699cedbaea14d8937b7c022160e10",
        "native_api": "torch.ops.aten.sgn.out(input,out=out)",
        "seed_policy": "manual_seed(0)",
        "fresh_inputs": True,
        "out_initial_values_undefined": True,
        "input_recipe": "generate_tensor_input;empty_like(out)",
        "native_timer_contract": {
            "native_schema": "aten::sgn.out",
            "returns_sample_out": True,
            "mutates": ["out"],
            "fresh_inputs": True,
            "setup_timed": False,
            "preparation": "clone_inputs_then_device_sync_before_start_event",
            "clock": "collector_CUDA_event_arithmetic_mean",
            "warmup_GPU_ms": 25,
            "measured_GPU_ms": 100,
            "NCU_supported": False,
        },
        "producer_selector": {"OPS_BASELINE_SGN_OUT_PROFILE": "current-fresh"},
        "inputs": {
            "args": [
                {
                    "shape": [64, 64],
                    "dtype": "torch.float16",
                    "stride": [64, 1],
                    "requires_grad": False,
                }
            ],
            "kwargs": {
                "out": {
                    "shape": [64, 64],
                    "dtype": "torch.float16",
                    "stride": [64, 1],
                    "requires_grad": False,
                }
            },
        },
    },
}

_SGN_PROFILE_ARTIFACTS = {
    "current-shared": {
        "file": "op_pytorch_h800_sgn_out_suiyuan_shared_core_20261028.json",
        "sha256": "ec637fb78f0308257ff3d159907961af3d9a297f892173fdba497982fd688991",
        "records": 15,
        "benchmark_node": "benchmark/test_sgn.py::test_sgn_out",
        "fresh_inputs": False,
    },
    "current-fresh": {
        "file": "op_pytorch_h800_sgn_out_suiyuan_fresh_core_20261028.json",
        "sha256": "cc7d62cc4b48052fbcdc1b58930e5c9aa22538b6120df7173d04b8d80232e4ef",
        "records": 15,
        "benchmark_node": "benchmark/test_sgn_out.py::test_sgn_out",
        "fresh_inputs": True,
    },
}

_SGN_PROFILE_MANIFEST_SHA256 = (
    "90f778b405de31a012c0344afc54357b1e96b74973042c69b5a82ab1636ad86c"
)

_SGN_NATIVE_PRODUCER = "468659cd07d68ddc33a365ab31533b53b79f00b2"

_SGN_FRAMEWORK_BINDINGS = {
    "benchmark/generated_operator_utils.py": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
    "benchmark/base.py": "daa8c109e43a06b03f34a14333135cf8cdc6efdbedab082eedb3bc4d7a3e3f5d",
    "benchmark/utils.py": "2fa295e0f5d02e000d9e30967354cc663011de77366dd469feb8ddee7395a4a5",
    "benchmark/consts.py": "7ef1bbbd3250023dac499bf1a0463b7e4aa154b75423301cfb32113d096ce426",
    "benchmark/cases.py": "8dbcf48ee8fa821869c98fc5d99fe147894e1ea034218df0cf61a425e408dca6",
    "src/flag_gems/runtime/backend/_enflame/core_shapes.yaml": (
        "69f31cd4f11f227449f7b57fad62f31d" "bd28730c19688c28cd127053e5f62ff1"
    ),
}


def _sgn_canonical_call(call):
    return json.dumps(call, sort_keys=True, separators=(",", ":"))


class _SgnOutCurrentNativeReference:
    _sgn_profile = "current-shared"

    def _sgn_static_applies(self):
        return flag_gems.vendor_name == "enflame" and self.op_name == "sgn_out"

    def _sgn_expected_config(self, shape, dtype):
        profile = self._sgn_profile
        shapes = _SGN_CURRENT_SHAPES[profile]
        if list(shape) not in shapes or dtype not in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("Unknown original SGNOut profile input")
        result = copy.deepcopy(_SGN_BASE_CONFIGS[profile])
        stride, extent = [], 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size
        tensor = dict(
            shape=list(shape), dtype=str(dtype), stride=stride, requires_grad=False
        )
        result["original_case_ordinal"] = shapes.index(list(shape))
        result["inputs"] = dict(args=[tensor], kwargs={"out": dict(tensor)})
        return result

    def _require_sgn_static(self, args, kwargs, case_id):
        profile = self._sgn_profile
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError(
                "SGNOut CPU fallback requires exact H800 profile --base-data"
            )
        if len(args) != 1 or set(kwargs) != {"out"}:
            raise ValueError("SGNOut native input/out schema differs")
        inp, out = args[0], kwargs["out"]
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise ValueError("SGNOut original input/out must be tensors")
        target = torch.device(self.device)
        if inp.device.type != target.type or (
            target.type != "cpu"
            and inp.device.index != (target.index if target.index is not None else 0)
        ):
            raise ValueError("SGNOut input device differs from requested target")
        for value in (inp, out):
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
                raise ValueError("SGNOut actual original input/out ABI differs")
        left, right = inp.untyped_storage(), out.untyped_storage()
        if left._cdata == right._cdata or (
            left.nbytes()
            and right.nbytes()
            and left.data_ptr() < right.data_ptr() + right.nbytes()
            and right.data_ptr() < left.data_ptr() + left.nbytes()
        ):
            raise ValueError("SGNOut original core input/out must be independent")
        fresh = profile == "current-fresh"
        native = torch.ops.aten.sgn.out if fresh else torch.sgn
        if (
            self.torch_op is not native
            or self.is_backward
            or self.is_inplace is not True
            or getattr(self, "fresh_inputs", False) is not fresh
        ):
            raise ValueError("SGNOut native callable/fresh-input boundary differs")
        expected = self._sgn_expected_config(list(inp.shape), inp.dtype)
        expected_id = (
            expected["benchmark_node"]
            + "::core::"
            + str(inp.dtype)[6:]
            + "::"
            + str(expected["original_case_ordinal"])
        )
        if case_id != expected_id:
            raise ValueError("SGNOut case identity differs from original profile")
        module = inspect.getmodule(type(self))
        bindings = expected["source_bindings"]
        for name in (
            ("_case_fn", "_build_inputs_fn")
            if fresh
            else ("_out_case_fn", "_out_build_inputs_fn")
        ):
            actual = inspect.getsource(getattr(module, name)).encode()
            if hashlib.sha256(actual).hexdigest() != bindings[name]:
                raise ValueError("Original SGNOut case/input builder source differs")
        # Whole recipe-file hashes describe original 2ff provenance. Only the
        # unchanged builders and timer/shape dependencies are current-source guards.
        for relative, digest in _SGN_FRAMEWORK_BINDINGS.items():
            path = Path(__file__).resolve().parent.parent / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Original SGNOut timer/core-shape framework differs")
        manifest = base.Config.base_data
        manifest_path = Path(manifest.source)
        manifest_payload = manifest_path.read_bytes()
        if hashlib.sha256(manifest_payload).hexdigest() != _SGN_PROFILE_MANIFEST_SHA256:
            raise ValueError("Exact SGNOut H800 profile manifest required")
        if manifest.data != json.loads(manifest_payload):
            raise ValueError("Loaded SGNOut profile manifest differs")
        declared = manifest.data["_sgn_out_profiles"]
        spec = declared["profiles"][profile]
        if (
            declared["producer_commit"] != _SGN_NATIVE_PRODUCER
            or spec != _SGN_PROFILE_ARTIFACTS[profile]
            or Path(spec["file"]).name != spec["file"]
        ):
            raise ValueError("SGNOut static profile/source differs")
        raw_path = manifest_path.parent / spec["file"]
        raw_payload = raw_path.read_bytes()
        if hashlib.sha256(raw_payload).hexdigest() != spec["sha256"]:
            raise ValueError("SGNOut native profile artifact bytes differ")
        baseline = BaselineData.load(raw_path)
        if baseline is None or baseline.data != json.loads(raw_payload):
            raise ValueError("SGNOut native profile could not be loaded")
        task = baseline.data["_collection"]["tasks"]["sgn_out"]
        env = baseline.data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != _SGN_NATIVE_PRODUCER
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
        ):
            raise ValueError(
                "Completed clean NVIDIA H800 native SGNOut source required"
            )
        call = describe_call(args, kwargs)
        record, key, reason = baseline.lookup(
            "sgn_out", inp.dtype, call, target_device=self.device
        )
        if (
            record is None
            or reason is not None
            or record["config"] != expected
            or call != expected["inputs"]
            or not positive_number(record["latency_ms"])
            or not positive_number(record["T_us"])
            or not math.isclose(
                record["T_us"], 1000 * record["latency_ms"], rel_tol=1e-9
            )
        ):
            raise ValueError(
                "Actual SGNOut input does not select the exact native record"
            )
        return (
            manifest,
            manifest_path,
            baseline,
            raw_path,
            _sgn_canonical_call(call),
            record["latency_ms"],
            key,
        )

    def _measure_input(self, input, case_id=None):
        if self._sgn_static_applies():
            self._sgn_checked = self._require_sgn_static(
                *self.unpack_to_args_kwargs(input), case_id
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if not self._sgn_static_applies():
            return super()._apply_external_baseline(metric, dtype, context)
        try:
            manifest, manifest_path, baseline, raw_path, captured, latency, key = (
                self._sgn_checked
            )
            if (
                context is None
                or context[0] is not manifest
                or context[1] is None
                or base.Config.base_data is not manifest
                or _sgn_canonical_call(context[1]) != captured
                or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                != _SGN_PROFILE_MANIFEST_SHA256
                or hashlib.sha256(raw_path.read_bytes()).hexdigest()
                != _SGN_PROFILE_ARTIFACTS[self._sgn_profile]["sha256"]
            ):
                raise ValueError(
                    "SGNOut static source/input changed around candidate timer"
                )
            super()._apply_external_baseline(
                metric, dtype, (baseline, context[1], context[2])
            )
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
                    for value in (
                        metric.latency,
                        metric.base_raw_speedup,
                        metric.compared_speedup,
                    )
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
                raise ValueError("SGNOut candidate/static comparison metric incomplete")
        finally:
            self._sgn_checked = None


class SgnBenchmark(
    _SgnOutCurrentNativeReference, base.UnaryPointwiseBenchmark, OperatorBenchmark
):
    """Keep unary shape policy while using the generated case-listing adapter."""

    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.sgn
def test_sgn():
    bench = SgnBenchmark(
        op_name="sgn",
        torch_op=torch.sgn,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.sgn_
def test_sgn_inplace():
    bench = SgnBenchmark(
        op_name="sgn_",
        torch_op=lambda value: value.sgn_(),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()


@pytest.mark.sgn_out
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="Native SGNOut executes on CPU; use the exact H800 profile",
)
def test_sgn_out():
    # The out buffer is fully overwritten on every invocation, so cloning giant
    # default-shape tensors between samples is unnecessary and can cause OOM.
    bench = SgnBenchmark(
        op_name="sgn_out",
        torch_op=torch.sgn,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
