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

# row_stack creates 3 inputs + 1 output. Cap total elements to avoid OOM /
# invalid-argument errors on the largest generated shapes.
MAX_ELEMENTS = 2**29


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
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
    }
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


class RowStackBenchmark(base.Benchmark):
    # row_stack creates 3 inputs + 1 output. Cap total elements to avoid OOM /
    # invalid-argument errors on the largest generated shapes.
    MAX_ELEMENTS = 2**29

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def init_user_config(self):
        super().init_user_config()
        # Filter out shapes whose total element count exceeds MAX_ELEMENTS;
        # row_stack materializes 3 inputs plus the stacked output in memory.
        self.shapes = [s for s in self.shapes if math.prod(s) <= self.MAX_ELEMENTS]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def set_more_shapes(self):
        # Extra 2D / 3D shapes spanning a range of trailing-axis widths so the
        # benchmark covers both shallow-wide and deeper stacks.
        more_shapes_2d = [(1024, 2**i) for i in range(1, 11, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 8, 4)]
        return more_shapes_2d + more_shapes_3d

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"inputs": [list(shape)] * 3},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return next(self.input_fn(shape, case.dtype, self.device))

    def _capture_external_baseline(self, args, kwargs):
        context = super()._capture_external_baseline(args, kwargs)
        return _aten_capture_context(
            context,
            self.op_name,
            ("pr56:op_pytorch_h800_median_row_lanczos_union_perf_baseline"),
        )


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)

    yield [inp1, inp2, inp3],


@pytest.mark.row_stack
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_row_stack():
    bench = RowStackBenchmark(
        op_name="row_stack",
        input_fn=_input_fn,
        torch_op=torch.row_stack,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.row_stack_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_row_stack_out():
    def _out_input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        inp3 = utils.generate_tensor_input(shape, dtype, device)
        # Pre-allocate the out tensor with the vertically-stacked shape.
        out = torch.empty(
            (inp1.shape[0] + inp2.shape[0] + inp3.shape[0],) + inp1.shape[1:],
            dtype=dtype,
            device=device,
        )
        yield [inp1, inp2, inp3], {"out": out}

    bench = RowStackBenchmark(
        op_name="row_stack_out",
        input_fn=_out_input_fn,
        torch_op=torch.row_stack,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
