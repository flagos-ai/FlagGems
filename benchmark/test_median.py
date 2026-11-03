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
from typing import Generator

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


class MedianNoDimBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"
    with_out = False

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            if self.with_out:
                out = torch.empty((), dtype=cur_dtype, device=self.device)
                yield inp, {"out": out}
            else:
                yield (inp,)


class MedianReductionBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape or [input shape, dim, keepdim]"
    with_out = False

    def get_input_iter(self, cur_dtype) -> Generator:
        for case_id, shape_spec in enumerate(self.shapes):
            if shape_spec and isinstance(shape_spec[0], (list, tuple)):
                shape = tuple(shape_spec[0])
                dim = int(shape_spec[1])
                keepdim = bool(shape_spec[2]) if len(shape_spec) > 2 else False
            else:
                shape = shape_spec
                keepdim = case_id % 3 == 0
                if len(shape) == 1:
                    dim = 0
                elif case_id % 2 == 0:
                    dim = len(shape) - 1
                else:
                    dim = 0
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            if self.with_out:
                out_shape = list(shape)
                if keepdim:
                    out_shape[dim] = 1
                else:
                    del out_shape[dim]
                values = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                indices = torch.empty(out_shape, dtype=torch.int64, device=self.device)
                yield inp, dim, {"keepdim": keepdim, "out": (values, indices)}
            else:
                yield inp, dim, {"keepdim": keepdim}


@pytest.mark.median
def test_median():
    bench = MedianNoDimBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median_out
def test_median_out():
    bench = MedianNoDimBenchmark(
        op_name="median_out",
        torch_op=torch.ops.aten.median.out,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
        with_out=True,
    )
    bench.run()


@pytest.mark.median_dim
def test_median_dim():
    bench = MedianReductionBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median_dim_values
def test_median_dim_values():
    bench = MedianReductionBenchmark(
        op_name="median_dim_values",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
        with_out=True,
    )
    bench.run()


class _ForwardShapeBenchmark:
    """Resolve the core shapes of the forward operator for a backward bench.

    `Benchmark.__init__` appends `_backward` to `op_name`, but `core_shapes.yaml`
    only carries the forward entries (`median`, `median_dim`). Resolve the
    forward entry instead of silently falling back to `Benchmark` shapes so the
    backward benchmark covers exactly the same cases as the forward one.
    """

    def set_shapes(self, shape_file_path=None):
        op_name = self.op_name
        if op_name.endswith("_backward"):
            self.op_name = op_name[: -len("_backward")]
        try:
            super().set_shapes(shape_file_path)
        finally:
            self.op_name = op_name


class MedianNoDimBackwardBenchmark(_ForwardShapeBenchmark, MedianNoDimBenchmark):
    """Times `torch.autograd.grad` of the global median.

    `is_backward=True` makes `Benchmark.unpack_to_args_kwargs` turn the input
    into a differentiable leaf and `_benchmark_callable` time
    `torch.autograd.grad` with a retained graph.
    """


class MedianDimBackwardBenchmark(_ForwardShapeBenchmark, MedianReductionBenchmark):
    def _benchmark_callable(self, op, *args, **kwargs):
        inp = args[0]
        dim = args[1]
        keepdim = kwargs.get("keepdim", False)
        values = op(inp, dim=dim, keepdim=keepdim)[0]
        dout = torch.randn_like(values)
        xs = [arg for arg in args if torch.is_tensor(arg) and arg.requires_grad]
        return (
            lambda: torch.autograd.grad(
                (values,), xs, grad_outputs=(dout,), retain_graph=True
            ),
            xs,
        )


@pytest.mark.median_backward
def test_median_backward():
    bench = MedianNoDimBackwardBenchmark(
        op_name="median",
        torch_op=torch.median,
        gems_op=flag_gems.median,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()


@pytest.mark.median_backward
def test_median_dim_backward():
    bench = MedianDimBackwardBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        gems_op=flag_gems.median_dim,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
