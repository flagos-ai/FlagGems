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
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .baseline import describe_call, positive_number

_CANONICAL_OPERATOR = "_embedding_bag_dense_backward"
_BASELINE_OPERATOR = "embedding_bag_dense_backward"
_H800_PRODUCER_COMMIT = "d17cdb9e2dfef9129cd52bdfb4991efb7ec19bf2"
_RECIPE_SOURCE_COMMIT = "1b96c4e3e6af089e5c1af7ca691712f95428ecea"
_INPUT_BUILDER_SHA256 = (
    "b98b44bdedf641bb73e5e8567bebbc6148984d23e5be7429b91abb37f87c45e4"
)

SHAPES = [
    (8, 16, 50, 4),
    (16, 32, 100, 4),
    (32, 64, 100, 4),
    (64, 128, 200, 4),
    (128, 256, 500, 4),
    (256, 128, 500, 8),
    (512, 256, 1000, 8),
]


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr38:op_pytorch_h800_embedding_bag_target_core_20261027": {
        "sha256": "bd0ec92ee235a0254e65fc70ff1474a92498f7dc42dbaee7848985b779394990",
        "artifact": "op_pytorch_h800_embedding_bag_target_core_20261027.json",
        "operators": ["embedding_bag_dense_backward"],
        "provenance": {
            "op_pytorch_h800_embedding_bag_target_core_20261027.provenance.json": {
                "key": "op_pytorch_h800_embedding_bag_target_core_20261027.provenance.json",
                "sha256": "42e2a1f8e32118e460106ac27ee4db85c7cd9abdc2281c800b3bb0633a4efeee",
            }
        },
        "canonical_operators": [
            "_embedding_bag_dense_backward",
            "embedding_bag_dense_backward",
        ],
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


class EmbeddingBagDenseBackwardBenchmark(base.GenericBenchmark):
    def _require_external_reference(self, args, kwargs):
        if (
            self.op_name != _CANONICAL_OPERATOR
            or not base.Config.skip_native
            or base.Config.base_data is None
        ):
            raise ValueError("CPU-fallback backends require matching H800 --base-data")
        if kwargs or len(args) != 10:
            raise ValueError("External reference requires the original positional ABI")
        grad, indices, mapping, sizes, maxima, weights, frequency, mode, sample, pad = (
            args
        )
        tensors = [grad, indices, mapping, sizes, maxima]
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("External reference requires the original tensor inputs")
        if (
            grad.ndim != 2
            or grad.dtype not in consts.FLOAT_DTYPES
            or type(weights) is not int
            or frequency is not False
            or type(mode) is not int
            or mode != 0
            or sample is not None
            or type(pad) is not int
            or pad != -1
        ):
            raise ValueError("External reference is restricted to original sum core")
        bags, width = grad.shape
        if not bags or indices.numel() % bags:
            raise ValueError("Invalid sample-to-bag geometry")
        per_bag = indices.numel() // bags
        descriptor = (bags, width, weights, per_bag)
        if descriptor not in SHAPES:
            raise ValueError("External reference has no matching original core shape")
        expected_shapes = [
            grad.shape,
            (bags * per_bag,),
            (bags * per_bag,),
            (bags,),
            (bags,),
        ]
        expected_strides = [(width, 1), (1,), (1,), (1,), (1,)]
        for i, (value, shape, stride) in enumerate(
            zip(tensors, expected_shapes, expected_strides)
        ):
            if (
                value.layout != torch.strided
                or value.device != grad.device
                or tuple(value.shape) != tuple(shape)
                or value.stride() != stride
                or value.storage_offset() != 0
                or value.requires_grad
                or (i and value.dtype != torch.int64)
            ):
                raise ValueError("External reference input layout or dtype differs")
        storages = [value.untyped_storage()._cdata for value in tensors]
        if len(set(storages)) != len(tensors):
            raise ValueError("External reference requires independent input storages")
        metadata = {
            "offset2bag": mapping.cpu().tolist(),
            "bag_size": sizes.cpu().tolist(),
            "maximum_indices": maxima.cpu().tolist(),
        }
        if metadata != {
            "offset2bag": [bag for bag in range(bags) for _ in range(per_bag)],
            "bag_size": [0] * bags,
            "maximum_indices": [0] * bags,
        }:
            raise ValueError("External reference must match actual unused sum metadata")
        if (
            hashlib.sha256(inspect.getsource(_build_inputs_fn).encode()).hexdigest()
            != _INPUT_BUILDER_SHA256
        ):
            raise ValueError("Current builder differs from the NVIDIA recipe")
        call = describe_call(args, kwargs)
        expected = {
            "inputs": call,
            "metadata_values": metadata,
            "tensor_alias_groups": [0, 1, 2, 3, 4],
            "seed": 42,
            "descriptor": list(descriptor),
            "variant": 0,
            "benchmark": "benchmark/test_embedding_bag_dense_backward.py",
            "flag_gems_operator": _CANONICAL_OPERATOR,
            "native_schema": "aten::_embedding_bag_dense_backward.default",
            "scope": "core",
            "input_contract": "_embedding_bag_dense_backward.sum_target_metadata/v1",
            "grid_contract": "_embedding_bag_dense_backward.kernelgen-dev.core/v1",
            "producer_selector": {"OPS_BASELINE_EMBEDDING_BAG_TARGET_CORE": "1"},
            "input_builder_sha256": _INPUT_BUILDER_SHA256,
            "flag_gems_source_commit": _RECIPE_SOURCE_COMMIT,
        }
        baseline = _aten_baseline_view(
            base.Config.base_data,
            ("pr38:op_pytorch_h800_embedding_bag_target_core_20261027"),
        )
        payload = _aten_read_bytes(
            Path(baseline.source),
            ("pr38:op_pytorch_h800_embedding_bag_target_core_20261027"),
        )

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate baseline JSON key")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite baseline JSON constant: {value}")

        canonical = lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        if canonical(data) != canonical(baseline.data):
            raise ValueError("Loaded baseline differs from the source artifact")
        task = data["_collection"]["tasks"][_BASELINE_OPERATOR]
        environment = data["_environments"][str(task["device"])]
        if (
            data[_BASELINE_OPERATOR]["native_api"] != "ops.embedding_bag_dense_backward"
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
            raise ValueError("A completed clean native H800 profile is required")
        # The collector's legacy module key is explicit. The candidate's
        # canonical Definition/override/op_name is never renamed for lookup.
        record, key, reason = baseline.lookup(
            _BASELINE_OPERATOR, grad.dtype, call, target_device=self.device
        )
        if (
            record is None
            or reason is not None
            or canonical(record["config"]) != canonical(expected)
        ):
            raise ValueError(
                "H800 input/metadata/alias profile differs or is ambiguous"
            )
        latency_us, latency_ms = record.get("T_us"), record.get("latency_ms")
        if (
            not positive_number(latency_us)
            or not positive_number(latency_ms)
            or not math.isclose(
                latency_ms, latency_us / 1000, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            raise ValueError("H800 latency must be positive with matching units")
        return (
            baseline,
            canonical(call),
            latency_ms,
            key,
            hashlib.sha256(payload).hexdigest(),
        )

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name in {"ascend", "enflame"}:
            self._checked_external_reference = self._require_external_reference(
                *self.unpack_to_args_kwargs(input)
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if flag_gems.vendor_name not in {"ascend", "enflame"}:
            return super()._apply_external_baseline(metric, dtype, context)
        try:
            if context is None or context[1] is None:
                raise ValueError("External reference context is missing")
            baseline, call, named_call = context
            checked, captured, latency, key, artifact = self._checked_external_reference
            if (
                baseline is not checked
                or json.dumps(call, sort_keys=True, separators=(",", ":")) != captured
                or hashlib.sha256(
                    _aten_read_bytes(
                        Path(baseline.source),
                        ("pr38:op_pytorch_h800_embedding_bag_target_core_20261027"),
                    )
                ).hexdigest()
                != artifact
            ):
                raise ValueError("External reference changed around candidate timing")
            try:
                chip = base.torch_device_fn.get_device_name()
            except (AttributeError, RuntimeError, TypeError):
                chip = ""
            baseline.apply(
                metric,
                _BASELINE_OPERATOR,
                dtype,
                call,
                named_call,
                flag_gems.vendor_name,
                chip,
                target_device=self.device,
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
                raise ValueError(
                    "Candidate timing or external comparison is incomplete"
                )
        finally:
            self._checked_external_reference = None

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES

    def _capture_external_baseline(self, args, kwargs):
        context = super()._capture_external_baseline(args, kwargs)
        return _aten_capture_context(
            context,
            self.op_name,
            ("pr38:op_pytorch_h800_embedding_bag_target_core_20261027"),
        )


def _case_fn(shape, dtype):
    del dtype
    num_bags, embedding_dim, num_weights, samples_per_bag = shape
    yield base.BenchmarkCasePlan(
        shape={
            "grad": [num_bags, embedding_dim],
            "indices": [num_bags * samples_per_bag],
        },
        params={"num_weights": num_weights, "mode": 0, "padding_idx": -1},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(42)
    (shape,) = plan.builder_args
    num_bags, embedding_dim, num_weights, samples_per_bag = shape
    num_samples = num_bags * samples_per_bag
    weight = torch.randn(num_weights, embedding_dim, dtype=dtype, device=device)
    indices = torch.randint(0, num_weights, (num_samples,), device=device)
    offsets = torch.arange(0, num_samples, samples_per_bag, device=device)
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, False, None, False, -1
    )
    if offset2bag.numel() == 0:
        offset2bag = (
            torch.arange(num_bags, dtype=offset2bag.dtype, device="cpu")
            .repeat_interleave(samples_per_bag)
            .to(indices.device)
        )
    grad = torch.randn_like(output)
    return (
        grad,
        indices,
        offset2bag,
        bag_size,
        maximum_indices,
        num_weights,
        False,
        0,
        None,
        -1,
    )


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.underscore_embedding_bag_dense_backward
@pytest.mark.skip_native(
    vendors=["ascend", "enflame"],
    reason="native backward dispatch uses CPU fallback; compare matching H800 --base-data",
)
def test_embedding_bag_dense_backward():
    bench = EmbeddingBagDenseBackwardBenchmark(
        op_name="_embedding_bag_dense_backward",
        torch_op=torch.ops.aten._embedding_bag_dense_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
