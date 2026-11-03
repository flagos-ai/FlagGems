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
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .baseline import describe_call, positive_number

# A clean producer is pinned here; no data is accepted until it is measured.
_H800_PRODUCER_COMMIT = "f31df74943936769a3a27ccf8411aefa378784e7"
_MEDIAN_CORE_SHAPES = [
    [1073741824],
    [64, 64],
    [4096, 4096],
    [64, 512, 512],
    [1024, 1024, 1024],
]
_MEDIAN_CORE_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int16,
    torch.int32,
]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr49:op_pytorch_h800_median_suiyuan_core_20261028": {
        "sha256": "90837247d6c610e4bbbabfeaa3032b6cb522b230286d4c6bebea8fa14372d023",
        "artifact": "op_pytorch_h800_median_suiyuan_core_20261028.json",
        "operators": ["median"],
        "provenance": {
            "median_h800_suiyuan_core_20261028.proof.json": {
                "key": "median_h800_suiyuan_core_20261028.proof.json",
                "sha256": "2df361f97a23c01ad475e86d87cf0d8147bbd7d0f81afe7cf30dc75358d5d0d9",
            }
        },
    },
    "pr56:op_pytorch_h800_median_row_lanczos_union_perf_baseline": {
        "sha256": "be758c4f292c1d8ae7b25789b06d2733f23716fde65e798c80c62bc403d16cfb",
        "artifact": "op_pytorch_h800_median_row_lanczos_union_perf_baseline.json",
        "operators": ["median", "median_dim", "row_stack_out"],
        "provenance": {
            "op_pytorch_h800_median_dim_aten_current_core_perf_baseline.provenance.json": {
                "key": "op_pytorch_h800_median_dim_aten_current_core_perf_baseline.provenance.json",
                "sha256": "b5da3769aa205005afa95138d688d5fe96e0e7edd675bdf08d41bb5042095366",
            }
        },
    },
    "pr56:op_pytorch_h800_median_dim_aten_current_core_perf_baseline": {
        "sha256": "6633a06619aaeddde7bcf813f8aae2f203da894d5481f897d7ff0915051a744a",
        "artifact": "op_pytorch_h800_median_dim_aten_current_core_perf_baseline.json",
        "operators": ["median_dim_aten_current_core"],
        "provenance": {
            "op_pytorch_h800_median_dim_aten_current_core_perf_baseline.provenance.json": {
                "key": "op_pytorch_h800_median_dim_aten_current_core_perf_baseline.provenance.json",
                "sha256": "b5da3769aa205005afa95138d688d5fe96e0e7edd675bdf08d41bb5042095366",
            }
        },
        "operator_aliases": {"median_dim": "median_dim_aten_current_core"},
    },
}


def _aten_unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate consolidated baseline key: " + key)
        result[key] = value
    return result


def _aten_load_json(raw):
    def reject(value):
        raise ValueError("Nonfinite consolidated baseline constant: " + value)

    return json.loads(raw, object_pairs_hook=_aten_unique_keys, parse_constant=reject)


def _aten_project_snapshot(data, source_id):
    spec = _ATEN_SNAPSHOT_SPECS[source_id]
    manifest = data["_aten_consolidation"]
    if spec.get("auxiliary"):
        entry = manifest["auxiliary_artifacts"][spec["auxiliary"]]
        raw = entry["raw_json"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Consolidated baseline manifest digest differs")
        if _aten_load_json(raw) != entry["content"]:
            raise ValueError(
                "Consolidated manifest content differs from original bytes"
            )
        return raw, None
    source = manifest["sources"][source_id]
    raw = source["raw_json"].encode("utf-8")
    if (
        hashlib.sha256(raw).hexdigest() != spec["sha256"]
        or source["artifact_sha256"] != spec["sha256"]
    ):
        raise ValueError("Consolidated native snapshot digest differs")
    original = _aten_load_json(raw)
    original_ops = {
        key
        for key, value in original.items()
        if isinstance(value, dict) and "shapes" in value
    }
    if set(source["operators"]) != set(spec["operators"]):
        raise ValueError("Consolidated native operator scope differs")
    if source["original_metadata"] != {
        key: value for key, value in original.items() if key not in original_ops
    }:
        raise ValueError("Consolidated collection metadata differs")
    for operator in spec["operators"]:
        entry = source["operators"][operator]
        expected = original[operator]
        if entry["entry_metadata"] != {
            key: value for key, value in expected.items() if key != "shapes"
        } or set(entry["shape_keys"]) != set(expected["shapes"]):
            raise ValueError("Consolidated native entry or case scope differs")
        seen = set()
        for old_key, new_key in entry["shape_keys"].items():
            if (
                new_key in seen
                or new_key != source_id + ":" + old_key
                or data[operator]["shapes"][new_key] != expected["shapes"][old_key]
            ):
                raise ValueError("Consolidated native timing/input record differs")
            seen.add(new_key)
    return raw, source


@lru_cache(maxsize=4)
def _aten_cached_payload(raw, source_id):
    if source_id is None:
        return raw, None
    data = _aten_load_json(raw)
    if "_aten_consolidation" not in data:
        return raw, None
    return _aten_project_snapshot(data, source_id)


def _aten_existing_path(path, source_id=None):
    path = Path(path)
    spec = _ATEN_SNAPSHOT_SPECS.get(source_id, {})
    original_name = Path(spec.get("artifact", spec.get("auxiliary", ""))).name
    if not path.exists() and path.name == original_name:
        unified = path.parent / "op_pytorch_h800_perf_baseline.json"
        if unified.is_file():
            return unified
    return path


def _aten_read_bytes(path, source_id):
    raw = _aten_existing_path(path, source_id).read_bytes()
    return _aten_cached_payload(raw, source_id)[0]


@lru_cache(maxsize=8)
def _aten_cached_view(baseline, source_id, payload):
    data = _aten_load_json(payload)
    for alias, original in (
        _ATEN_SNAPSHOT_SPECS[source_id].get("operator_aliases", {}).items()
    ):
        data[alias] = data[original]
    return type(baseline)(data, _aten_existing_path(baseline.source, source_id))


def _aten_baseline_view(baseline, source_id):
    if baseline is None or source_id is None:
        return baseline
    if "_aten_consolidation" not in baseline.data:
        aliases = _ATEN_SNAPSHOT_SPECS[source_id].get("operator_aliases", {})
        if not aliases or not all(
            original in baseline.data for original in aliases.values()
        ):
            return baseline
        payload = Path(baseline.source).read_bytes()
        if (
            hashlib.sha256(payload).hexdigest()
            != _ATEN_SNAPSHOT_SPECS[source_id]["sha256"]
        ):
            raise ValueError("Standalone native profile digest differs")
        return _aten_cached_view(baseline, source_id, payload)
    payload, _ = _aten_project_snapshot(baseline.data, source_id)
    return _aten_cached_view(baseline, source_id, payload)


def _aten_capture_context(context, operator, source_id):
    if context is None or source_id is None:
        return context
    spec = _ATEN_SNAPSHOT_SPECS[source_id]
    allowed = spec.get("canonical_operators")
    if allowed is None:
        allowed = tuple(spec.get("operators", ())) + tuple(
            spec.get("operator_aliases", ())
        )
    if allowed and operator not in allowed:
        return context
    return _aten_baseline_view(context[0], source_id), context[1], context[2]


class MedianBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def _require_external_reference(self, dtype, call, named_call=None):
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError("Suiyuan median requires matching H800 --base-data")
        baseline = _aten_baseline_view(
            base.Config.base_data,
            (
                "pr49:op_pytorch_h800_median_suiyuan_core_20261028"
                if flag_gems.vendor_name == "enflame"
                else "pr56:op_pytorch_h800_median_row_lanczos_union_perf_baseline"
            ),
        )

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate H800 median JSON key")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite H800 median constant: {value}")

        data = json.loads(
            _aten_read_bytes(
                Path(baseline.source),
                (
                    "pr49:op_pytorch_h800_median_suiyuan_core_20261028"
                    if flag_gems.vendor_name == "enflame"
                    else "pr56:op_pytorch_h800_median_row_lanczos_union_perf_baseline"
                ),
            ),
            object_pairs_hook=unique_keys,
            parse_constant=reject_constant,
        )
        if data != baseline.data:
            raise ValueError("Loaded median baseline differs from its artifact")
        task = data["_collection"]["tasks"]["median"]
        environment = data["_environments"][str(task["device"])]
        if (
            self.op_name != "median"
            or data["median"]["native_api"] != "ops.median"
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
                "A completed clean native H800 median collection is required"
            )
        shape_records = data["median"]["shapes"]
        if set(shape_records) != {str(shape) for shape in _MEDIAN_CORE_SHAPES}:
            raise ValueError(
                "H800 median must contain the complete original five shapes"
            )
        for records in shape_records.values():
            if set(records) != {str(value) for value in _MEDIAN_CORE_DTYPES}:
                raise ValueError(
                    "H800 median must contain the complete original five dtypes"
                )
            for record in records.values():
                if (
                    not positive_number(record.get("T_us"))
                    or not positive_number(record.get("latency_ms"))
                    or not math.isclose(
                        record["latency_ms"], record["T_us"] / 1000, rel_tol=1e-9
                    )
                ):
                    raise ValueError(
                        "H800 median units must be positive and consistent"
                    )
        record, _, reason = baseline.lookup(
            "median", dtype, call, named_call, target_device=self.device
        )
        if record is None or reason is not None:
            raise ValueError("H800 median record does not match the actual input")
        if dtype not in _MEDIAN_CORE_DTYPES or len(call["args"]) != 1:
            raise ValueError("Only original no-dim median core inputs are supported")
        shape = call["args"][0]["shape"]
        if shape not in _MEDIAN_CORE_SHAPES:
            raise ValueError("Unknown median core input shape")
        stride, extent = [], 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size
        expected_call = {
            "args": [
                {
                    "shape": shape,
                    "dtype": str(dtype),
                    "stride": stride,
                    "requires_grad": False,
                }
            ],
            "kwargs": {},
        }
        required = {
            "benchmark": "benchmark/test_median.py",
            "scope": "core",
            "input_contract": "median-no-dim-readonly-scalar/v1",
            "grid_contract": "median.suiyuan.kernelgen-dev.core/v1",
            "producer_selector": {"OPS_BASELINE_MEDIAN_SUIYUAN_CORE": "1"},
            "flag_gems_recipe_source": "2ff0931a646b97ec98ebd50642b5889619125a6b",
            "input_recipe": "manual_seed(0); floating randn(shape,dtype,device); integer randint full iinfo on CPU then target transfer; torch.median no dim",
            "input_builder_hash_method": "ast.get_source_segment(MedianBenchmark.build_inputs),ast.get_source_segment(generate_tensor_input)",
            "input_builder_sha256": "d95708621137c7551aefb533f2d6c5bf94b008b7b2e649a75db89a761abad920",
            "input_helper_sha256": "e991af3667ae315b99d8508a74c52528e478f5bc00ddd96eab06dad01b219495",
        }
        config = record["config"]
        if (
            call != expected_call
            or config["inputs"] != call
            or set(config) != set(required) | {"inputs"}
            or any(config.get(key) != value for key, value in required.items())
        ):
            raise ValueError("H800 median recipe, overload or input layout differs")
        return record["latency_ms"]

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name == "enflame":
            args, kwargs = self.unpack_to_args_kwargs(input)
            self._require_external_reference(args[0].dtype, describe_call(args, kwargs))
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        super()._apply_external_baseline(metric, dtype, context)
        if flag_gems.vendor_name != "enflame":
            return
        if context is None:
            raise ValueError("Suiyuan median external reference context is missing")
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
                "Median candidate timing or external comparison is incomplete"
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
            raise ValueError("Median comparison ratios do not match the actual timings")

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        return (utils.generate_tensor_input(shape, case.dtype, self.device),)

    def _capture_external_baseline(self, args, kwargs):
        context = super()._capture_external_baseline(args, kwargs)
        return _aten_capture_context(
            context,
            self.op_name,
            (
                "pr49:op_pytorch_h800_median_suiyuan_core_20261028"
                if flag_gems.vendor_name == "enflame"
                else "pr56:op_pytorch_h800_median_row_lanczos_union_perf_baseline"
            ),
        )


class MedianOutBenchmark(MedianBenchmark):
    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        out = torch.empty((), dtype=case.dtype, device=self.device)
        return inp, {"out": out}


@pytest.mark.median
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="native no-dim median is CPU fallback; use matching H800 --base-data",
)
def test_median():
    MedianBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


@pytest.mark.median_out
def test_median_out():
    MedianOutBenchmark(
        op_name="median_out",
        torch_op=torch.ops.aten.median.out,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


# The dim overloads previously had correctness suites but no benchmark entry.
_MEDIAN_DIM_CASES = [
    ((9, 7), 0, False),
    ((9, 7), -1, True),
    ((384, 7), 0, False),
    ((5, 9), 1, False),
]


class MedianDimBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(_MEDIAN_DIM_CASES)
        self.shape_desc = "input shape, reduction dim, keepdim"

    def _capture_external_baseline(self, args, kwargs):
        context = super()._capture_external_baseline(args, kwargs)
        return _aten_capture_context(
            context,
            self.op_name,
            ("pr56:op_pytorch_h800_median_dim_aten_current_core_perf_baseline"),
        )


def _median_dim_case_fn(shape, dtype):
    del dtype
    dims, dim, keepdim = shape
    dims = tuple(dims)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"dim": dim, "keepdim": keepdim},
        builder_args=(dims,),
    )


def _median_dim_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, plan.params["dim"], plan.params["keepdim"]


def _median_dim_values_build_inputs_fn(plan, dtype, device):
    inp, dim, keepdim = _median_dim_build_inputs_fn(plan, dtype, device)
    normalized_dim = dim % inp.ndim
    out_shape = list(inp.shape)
    if keepdim:
        out_shape[normalized_dim] = 1
    else:
        del out_shape[normalized_dim]
    values = torch.empty(out_shape, dtype=dtype, device=device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=device)
    return inp, dim, keepdim, {"values": values, "indices": indices}


@pytest.mark.median_dim
def test_median_dim():
    MedianDimBenchmark(
        op_name="median_dim",
        torch_op=torch.ops.aten.median.dim,
        case_fn=_median_dim_case_fn,
        build_inputs_fn=_median_dim_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


@pytest.mark.median_dim_values
def test_median_dim_values():
    MedianDimBenchmark(
        op_name="median_dim_values",
        torch_op=torch.ops.aten.median.dim_values,
        case_fn=_median_dim_case_fn,
        build_inputs_fn=_median_dim_values_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()
