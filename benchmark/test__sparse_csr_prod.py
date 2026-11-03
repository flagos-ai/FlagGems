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
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# The CSR kernel asserts input_dim == 2 and core_shapes.yaml has no
# _sparse_csr_prod entry, so these 2-D defaults replace the inherited 1-D/3-D
# ones. Inputs come from a dense tensor converted to CSR, so the stored entry
# count depends on how many random values are nonzero. A caller-supplied
# --shape_file entry for the operator still overrides them.
_PROD_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 512),
    (512, 4096),
    (1, 65536),
    (65536, 1),
]

# Single-axis and multi-axis (empty dim list) reduction forms share one plan
# shape, so listing and replay cover both without allocating tensors.
_DIMS = ([0], [1], [])

_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


# Linux fixture recipe shared by value with the ops-baseline producer.
# Cross-environment RNG output is not assumed: actual payload and geometry
# must match the collected config, or external-reference timing fails.
# Keep this operator-local: target installations need no ops-baseline import.
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
RECIPE = "csr-prod-pcg64-normal-f32-cast-compact-v2"
CASE_SOURCE = "633f62d145b0c36ba374e4595dc3540a11044622"


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr28:op_pytorch_sparse_csr_prod_h800_perf_baseline": {
        "sha256": "5d409c7d36331f2da6db8b8eb74360a5d96606174a46a2c548f2e307475f1a43",
        "artifact": "op_pytorch_sparse_csr_prod_h800_perf_baseline.json",
        "operators": ["_sparse_csr_prod"],
        "provenance": {},
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


@lru_cache(maxsize=1)
def _aten_container_sha(raw):
    return (
        hashlib.sha256(raw).hexdigest()
        if "_aten_consolidation" in _aten_load_json(raw)
        else None
    )


def _aten_qualify_metadata(metadata, source_id):
    if source_id is None:
        return metadata
    path = _aten_existing_path(metadata["artifact"], source_id)
    container_sha = _aten_container_sha(path.read_bytes())
    if container_sha is None:
        return metadata
    selected_sha = _ATEN_SNAPSHOT_SPECS[source_id]["sha256"]
    reported_sha = metadata["artifact_sha256"]
    if reported_sha not in (selected_sha, container_sha):
        raise ValueError("Reported native baseline digest has no matching scope")
    return {
        **metadata,
        "artifact_sha256_scope": (
            "container_file" if reported_sha == container_sha else "selected_snapshot"
        ),
        "snapshot_source_id": source_id,
        "snapshot_sha256": selected_sha,
        "container_sha256": container_sha,
    }


def _seed(shape, dtype):
    data = json.dumps(
        {"recipe": RECIPE, "shape": list(shape), "dtype": str(dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def _cpu_input(shape, dtype):
    if len(shape) != 2 or any(
        isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in shape
    ):
        raise ValueError("Expected a nonnegative two-dimensional shape")
    if dtype not in DTYPES:
        raise TypeError("Unsupported CSR product benchmark dtype")
    rng = np.random.Generator(np.random.PCG64(_seed(shape, dtype)))
    dense = torch.from_numpy(rng.standard_normal(tuple(shape), dtype=np.float32))
    csr = dense.to(dtype).to_sparse_csr()
    # Native dense-to-CSR may share the index allocation and leave col at a
    # nonzero storage offset. Give each component its own compact allocation
    # so CPU configuration and target replay have the same physical geometry.
    return torch.sparse_csr_tensor(
        csr.crow_indices().clone(),
        csr.col_indices().clone(),
        csr.values().clone(),
        size=csr.shape,
        device="cpu",
    )


def _part_identity(part):
    # Even contiguous() may retain a singleton view's non-unit stride.
    logical = torch.empty(part.numel(), dtype=part.dtype, device="cpu")
    logical.copy_(part.detach().cpu().reshape(-1))
    raw = logical.view(torch.uint8)
    return {
        "dtype": str(part.dtype),
        "shape": list(part.shape),
        "stride": list(part.stride()),
        "storage_offset": part.storage_offset(),
        "storage_nbytes": part.untyped_storage().nbytes(),
        "is_conj": part.is_conj(),
        "is_neg": part.is_neg(),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def input_identity(inp):
    if inp.layout != torch.sparse_csr or inp.ndim != 2:
        raise ValueError("Expected a two-dimensional CSR input")
    shape = list(inp.shape)
    parts = (inp.crow_indices(), inp.col_indices(), inp.values())
    return {
        "recipe": RECIPE,
        "seed_hex": f"{_seed(shape, inp.dtype):016x}",
        "layout": str(inp.layout),
        "shape": shape,
        "dtype": str(inp.dtype),
        "nnz": inp._nnz(),
        "requires_grad": inp.requires_grad,
        "component_storage_aliases": [
            parts[i].untyped_storage()._cdata == parts[j].untyped_storage()._cdata
            for i, j in ((0, 1), (0, 2), (1, 2))
        ],
        "crow": _part_identity(inp.crow_indices()),
        "col": _part_identity(inp.col_indices()),
        "values": _part_identity(inp.values()),
    }


def config(binding, dtype):
    return {
        "native_expression": "torch.ops.aten._sparse_csr_prod.dim_dtype",
        "case_source_commit": CASE_SOURCE,
        "benchmark": "benchmark/test__sparse_csr_prod.py",
        "scope": "core",
        "dim": list(binding["dim"]),
        "keepdim": True,
        "input": input_identity(_cpu_input(binding["shape"], dtype)),
    }


def _uses_external_baseline():
    return bool(os.environ.get("FLAGGEMS_NV_BASELINE")) or flag_gems.vendor_name in {
        "ascend",
        "kunlunxin",
    }


def _case_fn(shape, dtype):
    del dtype
    for dim in _DIMS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"dim": dim},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    if _uses_external_baseline():
        cpu = _cpu_input(shape, dtype)
        inp = torch.sparse_csr_tensor(
            cpu.crow_indices().to(device),
            cpu.col_indices().to(device),
            cpu.values().to(device),
            size=cpu.shape,
            device=device,
        )
        if input_identity(inp) != input_identity(cpu):
            raise ValueError("Target CSR payload or storage geometry changed")
    else:
        inp = utils.generate_tensor_input(shape, dtype, device).to_sparse_csr()
    # unpack_to_args_kwargs turns the returned dict into call kwargs, so the
    # reference and the candidate share op(input, dim=..., keepdim=True).
    return inp, {"dim": plan.params["dim"], "keepdim": True}


def _move_reference(value, device):
    if value.layout == torch.sparse_csr:
        result = torch.sparse_csr_tensor(
            value.crow_indices().to(device).clone(),
            value.col_indices().to(device).clone(),
            value.values().to(device).clone(),
            size=value.shape,
            device=device,
        )
    else:
        result = value.to(device).clone()
    assert result.dtype == value.dtype
    assert result.shape == value.shape
    assert result.layout == value.layout
    return result


def _cpu_oracle(inp, dim, keepdim=False, *, dtype=None, out=None):
    native = torch.ops.aten._sparse_csr_prod
    cpu = _move_reference(inp, "cpu")
    if out is None:
        expected = native.dim_dtype(cpu, dim, keepdim, dtype=dtype)
        # Empty global CSR reductions on the inspected CPU stacks expose an
        # uninitialized scalar values buffer despite crow=[0, 0] and no column
        # indices. It represents no stored entry. Preserve the native support,
        # result shape and promoted dtype, with a valid empty values component.
        if (
            expected.layout == torch.sparse_csr
            and expected.col_indices().numel() == 0
            and expected.values().ndim == 0
        ):
            expected = torch.sparse_csr_tensor(
                expected.crow_indices().clone(),
                expected.col_indices().clone(),
                torch.empty(0, dtype=expected.dtype, device="cpu"),
                size=expected.shape,
                device="cpu",
            )
        result = _move_reference(expected, inp.device)
    else:
        if out.device != inp.device:
            raise RuntimeError("Input and out devices must match")
        cpu_out = _move_reference(out, "cpu")
        expected = native.dim_dtype_out(cpu, dim, keepdim, dtype=dtype, out=cpu_out)
        target = _move_reference(expected, out.device)
        if out.shape == target.shape and all(
            a.shape == b.shape
            for a, b in zip(
                (out.crow_indices(), out.col_indices(), out.values()),
                (target.crow_indices(), target.col_indices(), target.values()),
            )
        ):
            out.crow_indices().copy_(target.crow_indices())
            out.col_indices().copy_(target.col_indices())
            out.values().copy_(target.values())
        else:
            # CPU ATen decides whether this out geometry may be resized.
            out.data = target
        result = out
    assert result.device == inp.device
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert result.layout == expected.layout
    return result


# Filled from the reviewed, immutable native H800 collection before publication.
_NV_ARTIFACT_SHA256 = "5d409c7d36331f2da6db8b8eb74360a5d96606174a46a2c548f2e307475f1a43"
_NV_PUBLISHED_REVISION = "7a1ff9ab087ac1f753e9fbb8b8a14ba7c300d140"
_NV_COLLECTION_SOURCE = "6e9d4fec5adb0c0d0bb864d74b3c7c1df50adfdd"


@dataclass
class _CrossMachineMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None
    candidate_timer: dict | None = None


class SparseCsrProdBenchmark(OperatorBenchmark):
    def _candidate_call(self):
        registry = base.Config.override_registry
        override = registry.get_override(self.op_name) if registry is not None else None
        op = override if override is not None else self.gems_op
        if not callable(op):
            raise ValueError(
                "CSR product benchmark requires a direct candidate callable"
            )
        # Missing candidates must never fall back to native/use_gems dispatch.
        return op, nullcontext(), override is not None

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_PROD_SHAPES)

    @cached_property
    def _nvidia_baseline(self):
        value = os.environ.get("FLAGGEMS_NV_BASELINE")
        if not value:
            raise ValueError("Set FLAGGEMS_NV_BASELINE to the reviewed H800 snapshot")
        path = Path(value).resolve(strict=True)
        payload = _aten_read_bytes(
            path, ("pr28:op_pytorch_sparse_csr_prod_h800_perf_baseline")
        )
        if hashlib.sha256(payload).hexdigest() != _NV_ARTIFACT_SHA256:
            raise ValueError("Expected the reviewed CSR product H800 snapshot")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite baseline JSON constant: {value}")

        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        # Leading underscore is part of this operator name, not a metadata tag.
        entry = data[self.op_name]
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or entry["native_api"] != f"ops.{self.op_name}"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _NV_COLLECTION_SOURCE
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["selected_ops"] != [self.op_name]
            or environment["run"]["repo"] != "pytorch"
            or sum(len(dtypes) for dtypes in entry["shapes"].values()) != 54
        ):
            raise ValueError("Incomplete or inconsistent H800 CSR product collection")
        return entry["shapes"], _aten_qualify_metadata(
            {
                "kind": "external_nvidia",
                "artifact": str(path),
                "artifact_sha256": _NV_ARTIFACT_SHA256,
                "published_revision": _NV_PUBLISHED_REVISION,
                "environment": environment,
                "ratio": "(H800 latency_ms + EPS) / (target latency_ms + EPS); no peak normalization",
                "target_timing_mode": base.Config.mode.value,
            },
            ("pr28:op_pytorch_sparse_csr_prod_h800_perf_baseline"),
        )

    def _external_input_config(self, args, kwargs):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("Expected one CSR product input")
        inp = args[0]
        expected_device = torch.device(flag_gems.device)
        if expected_device.type != "cpu" and expected_device.index is None:
            expected_device = torch.device(
                expected_device.type, base.torch_device_fn.current_device()
            )
        if (
            inp.device != expected_device
            or inp.layout != torch.sparse_csr
            or tuple(inp.shape) not in _PROD_SHAPES
            or inp.dtype not in DTYPES
            or inp.requires_grad
            or set(kwargs) != {"dim", "keepdim"}
            or kwargs["keepdim"] is not True
            or kwargs["dim"] not in _DIMS
        ):
            raise ValueError(
                "External baseline requires the original core CSR product ABI"
            )
        binding = {"shape": list(inp.shape), "dim": kwargs["dim"]}
        expected = config(binding, inp.dtype)
        if input_identity(inp) != expected["input"]:
            raise ValueError("Target CSR product input differs from the H800 fixture")
        return expected

    def _import_nvidia_latency(self, args, kwargs):
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError("External CSR baseline supports forward latency only")
        expected = self._external_input_config(args, kwargs)
        shapes, provenance = self._nvidia_baseline
        matches = [
            record
            for dtypes in shapes.values()
            for dtype, record in dtypes.items()
            if dtype == str(args[0].dtype) and record.get("config") == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one H800 record matching the actual CSR input"
            )
        record = matches[0]
        latency_us = record.get("T_us")
        if (
            isinstance(latency_us, bool)
            or not isinstance(latency_us, (int, float))
            or not math.isfinite(latency_us)
            or latency_us <= 0
        ):
            raise ValueError("H800 T_us must be finite and positive")
        latency_ms = latency_us / 1000
        recorded_ms = record.get("latency_ms")
        if (
            isinstance(recorded_ms, bool)
            or not isinstance(recorded_ms, (int, float))
            or not math.isfinite(recorded_ms)
            or not math.isclose(recorded_ms, latency_ms, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": expected,
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(args[0].device),
                "device_name": base.torch_device_fn.get_device_name(),
                "torch": str(torch.__version__),
            },
        }
        if base.Config.reference_only:
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError("Cannot identify the active external reference case")
            active[0]["baseline_source"] = self._imported_baseline
        return latency_ms

    def get_latency(self, op, *args, **kwargs):
        if _uses_external_baseline() and (
            base.Config.reference_only
            or getattr(self, "_native_measurement_pending", False)
        ):
            if base.Config.skip_native or op is not self.torch_op:
                raise ValueError("External baseline requires the reference measurement")
            # Import only the reference timing. The next candidate measurement
            # always executes the real timer, even for the identical callable.
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        latency = super().get_latency(op, *args, **kwargs)
        if (
            flag_gems.vendor_name == "ascend"
            and base.Config.mode.value == "kernel"
            and latency == math.inf
        ):
            # do_bench_npu returns +Inf when no kernel_details.csv exists.
            # CPU-oracle bootstraps perform CPU work and transfers without NPU
            # compute kernels. Measure the actual callable with synchronization
            # instead; this is target timing, never the imported H800 baseline.
            latency = base.measure_calls(
                lambda: op(*args, **kwargs), warmup_calls=1, repeat_calls=3
            )
            self._candidate_timer = {
                "kind": "synchronized_host_wall_clock",
                "requested_mode": base.Config.mode.value,
                "reason": "NPU kernel profiler returned +Inf",
                "warmup_calls": 1,
                "timed_calls": 3,
            }
        return latency

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        self._native_measurement_pending = False
        self._candidate_timer = None
        external = _uses_external_baseline()
        if external and (
            self.is_backward
            or "tflops" in self.to_bench_metrics
            or base.Config.skip_native
        ):
            raise ValueError(
                "External baseline supports forward native-reference latency only"
            )
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if (
                external
                and metric.latency is not None
                and (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                )
            ):
                raise ValueError("Target candidate latency must be finite and positive")
            if self._imported_baseline is not None or self._candidate_timer is not None:
                return _CrossMachineMetrics(
                    **asdict(metric),
                    baseline_source=self._imported_baseline,
                    candidate_timer=self._candidate_timer,
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def _run_reference_cases(self, case_ids):
        first_record = len(base.Config.reference_records)
        if _uses_external_baseline() and base.Config.skip_native:
            raise ValueError("Do not combine an external baseline with skip_native")
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            for record in base.Config.reference_records[first_record:]:
                if record.get("baseline_source") and record.get("status") == "PASSED":
                    record["stage"] = "external_baseline"


@pytest.mark.underscore_sparse_csr_prod
def test__sparse_csr_prod():
    bench = SparseCsrProdBenchmark(
        op_name="_sparse_csr_prod",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_csr_prod.dim_dtype,
        gems_op=flag_gems._sparse_csr_prod,
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
