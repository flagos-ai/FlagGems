# Copyright 2026 FlagOS Contributors. All rights reserved.
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
import importlib
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

from . import base
from .generated_operator_utils import OperatorBenchmark

# The native operator carries float32/float64 kernels; float64 stays behind the static
# capability flag. Listing and execution share this list.
_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)

# A case is (shape, nnz[, sparse_dim, dim]); sparse_dim defaults to the full rank and
# dim to the last dimension. A shape file may use any of these spellings.
_DEFAULT_CASES = [
    ((1 << 20,), 1 << 20),
    ((1024, 1024), 1 << 20),
    ((2048, 2048), 1 << 20),
    ((4096, 4096), 1 << 20),
    ((256, 256, 256), 1 << 20),
    ((1024, 64), 1024, 1, 1),
    # Middle sparse reduction axis with a trailing sparse axis: the group is identified
    # by axes 0 and 2, so a prefix-only group id would collapse every entry into one
    # group. Kept in the core level on purpose.
    ((1024, 64, 16), 8192, 3, 1),
    # Duplicate coordinates are valid uncoalesced COO input, so a requested nnz above
    # the number of distinct positions is kept as asked instead of being clamped.
    ((4096,), 8192),
    # Native-valid scalar COO (dim=-1 is its only in-range dim) and empty operands;
    # the operator accepts nnz=0.
    ((), 1, 0, -1),
    ((), 0, 0, -1),
    ((1 << 20,), 0),
]


RECIPE = "coo-log-softmax-pcg64-native-width-compact-v1"

DTYPES = (torch.float32, torch.float64)


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr30:op_pytorch_sparse_log_softmax_h800_perf_baseline": {
        "sha256": "462befac7ca68148638169bc4df56c65a00adc3d5e9c85dd3b89736805f5b5c2",
        "artifact": "op_pytorch_sparse_log_softmax_h800_perf_baseline.json",
        "operators": ["_sparse_log_softmax", "_sparse_log_softmax_backward_data"],
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


def _validate(shape, nnz, sparse_dim, dim, dtype):
    shape = tuple(shape)
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Expected nonnegative integral extents")
    if type(nnz) is not int or nnz < 0:
        raise ValueError("Invalid stored count")
    if type(sparse_dim) is not int or not 0 <= sparse_dim <= len(shape):
        raise ValueError("Invalid sparse dimension count")
    if type(dim) is not int or (
        dim != -1 if not shape else not -len(shape) <= dim < len(shape)
    ):
        raise ValueError("Invalid normalization dimension")
    if nnz and any(n == 0 for n in shape[:sparse_dim]):
        raise ValueError("Nonempty support cannot occupy a zero sparse extent")
    if dtype not in DTYPES:
        raise TypeError("Only original float32/float64 dtypes")
    return shape


def seed_hex(kind, shape, nnz, sparse_dim, dim, dtype):
    fields = {
        "recipe": RECIPE,
        "kind": kind,
        "shape": list(shape),
        "nnz": nnz,
        "sparse_dim": sparse_dim,
        "dim": dim,
        "dtype": str(dtype),
    }
    return (
        hashlib.sha256(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        )
        .digest()[:8]
        .hex()
    )


def _rng(kind, shape, nnz, sparse_dim, dim, dtype):
    _validate(shape, nnz, sparse_dim, dim, dtype)
    return np.random.Generator(
        np.random.PCG64(
            int.from_bytes(
                bytes.fromhex(seed_hex(kind, shape, nnz, sparse_dim, dim, dtype)),
                "little",
            )
        )
    )


def _compact_cpu(array, dtype):
    result = torch.empty(tuple(array.shape), dtype=dtype, device="cpu")
    result.copy_(torch.from_numpy(array))
    return result


def forward_indices(shape, sparse_dim, nnz, dim):
    """Original O(nnz) grouped/rank-sorted unique forward support, no random draw."""
    if nnz == 0 or sparse_dim == 0:
        return np.empty((sparse_dim, nnz), dtype=np.int64)
    sizes = tuple(shape[:sparse_dim])
    axis = dim + len(shape) if dim < 0 else dim
    axis = axis if 0 <= axis < sparse_dim else sparse_dim - 1
    others = [i for i in range(sparse_dim) if i != axis]
    extent = math.prod(sizes[i] for i in others)
    group = min(sizes[axis], max(2, -(-nnz // extent)))
    order = np.arange(nnz, dtype=np.int64)
    base = order // group
    coords = [None] * sparse_dim
    coords[axis] = order % group
    for i in reversed(others):
        coords[i] = base % sizes[i]
        base = base // sizes[i]
    rank = np.ravel_multi_index(tuple(coords), sizes)
    rank.sort()
    return np.stack(np.unravel_index(rank, sizes)).astype(np.int64, copy=False)


def backward_indices(shape, sparse_dim, nnz, dim):
    """Original support groups, dense-axis spread, duplicates and original order."""
    if sparse_dim == 0 or nnz == 0:
        return np.empty((sparse_dim, nnz), dtype=np.int64)
    if dim < 0:
        dim += len(shape)
    index = np.arange(nnz, dtype=np.int64)
    if dim < sparse_dim:
        groups = tuple(shape[a] for a in range(sparse_dim) if a != dim)
        n_groups = math.prod(groups) if groups else 1
        cols = max(1, min(shape[dim], nnz))
        used = max(1, -(-nnz // cols))
        step = max(1, n_groups // used)
        group_id = ((index // cols) * step) % n_groups
        parts = iter(np.unravel_index(group_id, groups) if groups else ())
        coords = [index % cols if a == dim else next(parts) for a in range(sparse_dim)]
    else:
        spread = math.prod(shape[:sparse_dim])
        step = max(1, spread // nnz)
        coords = np.unravel_index((index * step) % spread, tuple(shape[:sparse_dim]))
    return np.stack(coords).astype(np.int64, copy=False)


def _canonical_coo(indices, values, shape, dtype, coalesced=None):
    kwargs = {} if coalesced is None else {"is_coalesced": coalesced}
    return torch.sparse_coo_tensor(
        _compact_cpu(indices, torch.int64),
        _compact_cpu(values, dtype),
        size=tuple(shape),
        dtype=dtype,
        device="cpu",
        **kwargs,
    )


def forward_cpu(shape, nnz, sparse_dim, dim, dtype):
    shape = _validate(shape, nnz, sparse_dim, dim, dtype)
    if not shape:
        raise ValueError("Original forward core excludes scalar shape")
    if nnz > math.prod(shape[:sparse_dim]):
        raise ValueError("Forward support must stay unique")
    rng = _rng("forward", shape, nnz, sparse_dim, dim, dtype)
    values = rng.standard_normal(
        (nnz,) + shape[sparse_dim:],
        dtype=np.float64 if dtype == torch.float64 else np.float32,
    )
    return _canonical_coo(
        forward_indices(shape, sparse_dim, nnz, dim), values, shape, dtype, True
    )


def backward_cpu(shape, nnz, sparse_dim, dim, dtype):
    shape = _validate(shape, nnz, sparse_dim, dim, dtype)
    rng = _rng("backward", shape, nnz, sparse_dim, dim, dtype)
    indices = backward_indices(shape, sparse_dim, nnz, dim)
    # All THREE original operands are full sparse COO tensors, independent normal
    # values and storage, equal support CONTENT. self is never a [1] dense placeholder.
    return tuple(
        _canonical_coo(
            indices.copy(),
            rng.standard_normal(
                (nnz,) + shape[sparse_dim:],
                dtype=np.float64 if dtype == torch.float64 else np.float32,
            ),
            shape,
            dtype,
        )
        for _ in range(3)
    )


def _contiguous_stride(shape):
    stride, result = 1, []
    for extent in reversed(shape):
        result.append(stride)
        stride *= max(extent, 1)
    return tuple(reversed(result))


def _part_identity(part):
    if not isinstance(part, torch.Tensor) or part.layout != torch.strided:
        raise TypeError("Expected a dense component Tensor")
    extent = part.numel() * part.element_size()
    # The canonical contract is compact, so every hashed storage byte is defined.
    if (
        part.storage_offset() != 0
        or part.untyped_storage().nbytes() != extent
        or tuple(part.stride()) != _contiguous_stride(part.shape)
        or part.is_conj()
        or part.is_neg()
    ):
        raise ValueError("Noncanonical component storage geometry or lazy flags")
    raw = (
        torch.empty(0, dtype=torch.uint8, device=part.device)
        .set_(part.untyped_storage(), 0, (extent,), (1,))
        .cpu()
    )
    if raw.dtype != torch.uint8 or raw.device.type != "cpu":
        raise ValueError("Byte observation changed its ABI")
    return {
        "dtype": str(part.dtype),
        "shape": list(part.shape),
        "stride": list(part.stride()),
        "storage_offset": part.storage_offset(),
        "storage_nbytes": extent,
        "is_conj": part.is_conj(),
        "is_neg": part.is_neg(),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def _tensor_identity(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected an input Tensor")
    if value.layout != torch.sparse_coo:
        return {
            "layout": str(value.layout),
            "requires_grad": value.requires_grad,
            "part": _part_identity(value),
        }
    return {
        "layout": str(value.layout),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sparse_dim": value.sparse_dim(),
        "dense_dim": value.dense_dim(),
        "nnz": value._nnz(),
        "is_coalesced": value.is_coalesced(),
        "requires_grad": value.requires_grad,
        "is_conj": value.is_conj(),
        "is_neg": value.is_neg(),
        "indices": _part_identity(value._indices()),
        "values": _part_identity(value._values()),
    }


def input_identity(named_inputs):
    """Complete defined input content/geometry plus every component alias pair.

    Absolute pointers/devices are not cross-machine identities. build transfer
    checks the exact requested physical device separately; config uses target role.
    """
    components = []
    tensors = {}
    for name, value in named_inputs.items():
        tensors[name] = _tensor_identity(value)
        parts = (
            [value._indices(), value._values()]
            if value.layout == torch.sparse_coo
            else [value]
        )
        labels = (
            ["indices", "values"] if value.layout == torch.sparse_coo else ["tensor"]
        )
        components.extend(
            (name + "." + label, part) for label, part in zip(labels, parts)
        )
    aliases = {}
    for i, (left, first) in enumerate(components):
        for right, second in components[i + 1 :]:
            aliases[left + "|" + right] = (
                first.untyped_storage()._cdata == second.untyped_storage()._cdata
            )
    return {"tensors": tensors, "component_storage_aliases": aliases}


def to_device(named_inputs, device):
    """Byte-preserving input transport only; native callable is resolved elsewhere."""
    expected = torch.empty(0, dtype=torch.uint8, device=device).device
    before = input_identity(named_inputs)

    def part_on_target(part):
        # All components are compact and physically defined. Independent copies
        # avoid typed F64 conversion/narrowing and preserve storage independence.
        raw = (
            torch.empty(0, dtype=torch.uint8, device="cpu")
            .set_(
                part.untyped_storage(), 0, (part.numel() * part.element_size(),), (1,)
            )
            .to(device=expected, copy=True)
        )
        result = raw.view(part.dtype).reshape(part.shape)
        if (
            result.device != expected
            or result.dtype != part.dtype
            or result.shape != part.shape
        ):
            raise ValueError("Input transport changed requested device/dtype/shape")
        return result

    result = {}
    for name, value in named_inputs.items():
        if value.layout == torch.sparse_coo:
            output = torch.sparse_coo_tensor(
                part_on_target(value._indices()),
                part_on_target(value._values()),
                size=value.shape,
                dtype=value.dtype,
                device=expected,
                is_coalesced=value.is_coalesced(),
            )
        else:
            output = part_on_target(value)
        if (
            output.device != expected
            or output.dtype != value.dtype
            or output.shape != value.shape
        ):
            raise ValueError("Input construction changed required ABI")
        result[name] = output
    if input_identity(result) != before:
        raise ValueError("Input transport changed payload, geometry, flags or aliases")
    return result


OP_NAME = "_sparse_log_softmax_backward_data"

DTYPES = [torch.float32, torch.float64]

IS_INPLACE = False

CASE_SOURCE = "f30a563479449016490ef674b3670ede6c101988"

BENCHMARK_SHA256 = "b825cab775806d3912d0fbe9c1976d3fabdb919165546d5e0ccbfb216daeccc0"

SHAPES = (
    ((1048576,), 1048576, 1, 0),
    ((1024, 1024), 1048576, 2, 1),
    ((2048, 2048), 1048576, 2, 1),
    ((4096, 4096), 1048576, 2, 1),
    ((256, 256, 256), 1048576, 3, 2),
    ((1024, 64), 1024, 1, 1),
    ((1024, 64, 16), 8192, 3, 1),
    ((4096,), 8192, 1, 0),
    ((), 1, 0, -1),
    ((), 0, 0, -1),
    ((1048576,), 0, 1, 0),
)


def native():
    try:
        return getattr(torch.ops.aten, OP_NAME).default
    except (AttributeError, RuntimeError):
        return None


def grid():
    return [
        {"shape": list(shape), "nnz": nnz, "sparse_dim": sd, "dim": dim, "ordinal": i}
        for i, (shape, nnz, sd, dim) in enumerate(SHAPES)
    ]


def _checked(binding, dtype):
    if dtype not in DTYPES:
        raise TypeError("Unsupported original dtype")
    if not isinstance(binding, dict) or type(binding.get("ordinal")) is not int:
        raise ValueError("Expected original descriptor and ordinal")
    i = binding["ordinal"]
    if not 0 <= i < len(SHAPES) or binding != grid()[i]:
        raise ValueError("Not an exact original core descriptor")
    return SHAPES[i]


def _cpu_inputs(binding, dtype):
    shape, nnz, sd, dim = _checked(binding, dtype)
    grad, output, self_tensor = backward_cpu(shape, nnz, sd, dim, dtype)
    return {"grad": grad, "output": output, "self": self_tensor}


def build_inputs(binding, dtype, device):
    _checked(binding, dtype)
    moved = to_device(_cpu_inputs(binding, dtype), device)
    return (moved["grad"], moved["output"], binding["dim"], moved["self"]), {}


def key_shape(binding):
    shape, nnz, sd, dim = _checked(binding, DTYPES[0])
    return (
        "coo_"
        + ("x".join(map(str, shape)) or "scalar")
        + f"_nnz_{nnz}_sd_{sd}_dim_{dim}"
    )


def config(binding, dtype):
    shape, nnz, sd, dim = _checked(binding, dtype)
    return {
        "native_expression": "torch.ops.aten._sparse_log_softmax_backward_data.default",
        "case_source_commit": CASE_SOURCE,
        "benchmark": "benchmark/test__sparse_log_softmax_backward_data.py",
        "benchmark_sha256": BENCHMARK_SHA256,
        "scope": "core",
        "case_id": "benchmark/test__sparse_log_softmax_backward_data.py::"
        "test__sparse_log_softmax_backward_data_benchmark::core::"
        + f"{str(dtype).removeprefix('torch.')}::{binding['ordinal']}",
        "recipe": RECIPE,
        "seed_hex": seed_hex("backward", shape, nnz, sd, dim, dtype),
        "requested_nnz": nnz,
        "sparse_dim": sd,
        "dim": dim,
        "input_device_role": "target",
        "inputs": input_identity(_cpu_inputs(binding, dtype)),
    }


def _uses_external_baseline():
    return flag_gems.vendor_name in ("ascend", "kunlunxin") or bool(
        os.environ.get("FLAGGEMS_NV_BASELINE")
    )


def _cpu_oracle(*args, **kwargs):
    """Explicit runner bootstrap only; never an NVIDIA latency source.

    The integrated correctness module owns the CPU native oracle. This base
    snapshot intentionally fails if that repair has not yet been integrated.
    Candidate selection never silently falls back to this function.
    """
    module = importlib.import_module("tests.test__sparse_log_softmax_backward_data")
    return module._cpu_oracle(*args, **kwargs)


# Fail closed until the new 40-record native H800 collection is reviewed.
_NV_ARTIFACT_SHA256 = "462befac7ca68148638169bc4df56c65a00adc3d5e9c85dd3b89736805f5b5c2"
_NV_PUBLISHED_REVISION = "2d308ec4a2c455a10accb340aee0e84ac916f2a0"
_NV_COLLECTION_SOURCE = "0f73cb27eb44781aa75cc2519482701c06f03e1e"
_NV_COLLECTION_ENVIRONMENT = {
    "schema_version": 1,
    "collected_at_utc": "2026-10-22T15:54:47.973038+00:00",
    "hardware": {
        "gpu": {
            "name": "NVIDIA H800",
            "uuid": "555633f8-c39c-bb8a-d549-791cb8451f44",
            "total_memory_bytes": 85017624576,
            "total_memory_gib": 79.18,
            "compute_capability": "9.0",
            "multiprocessor_count": 132,
            "cuda_device_index": 0,
            "driver_version": "580.105.08",
        },
        "cpu_model": "Intel(R) Xeon(R) Platinum 8468V",
        "cpu_architecture": "x86_64",
        "cpu_socket_count": 2,
        "physical_cpu_core_count": 96,
        "logical_cpu_count": 192,
        "cpu_threads_per_core": 2,
        "numa_node_count": 2,
        "process_cpu_affinity_count": 192,
        "hypervisor_vendor": None,
    },
    "software": {
        "os": "Linux",
        "os_release": "5.15.0-78-generic",
        "machine": "x86_64",
        "python": "3.12.3",
        "packages": {
            "torch": "2.8.0a0+5228986c39.nv25.5",
            "triton": "3.6.0",
            "vllm": None,
            "sglang": None,
        },
        "torch_cuda_version": "12.9",
        "nvcc_version": "nvcc: NVIDIA (R) Cuda compiler driver\n"
        "Copyright (c) 2005-2025 NVIDIA Corporation\n"
        "Built on Wed_Apr__9_19:24:57_PDT_2025\n"
        "Cuda compilation tools, release 12.9, V12.9.41\n"
        "Build cuda_12.9.r12.9/compiler.35813241_0",
        "ncu_version": "NVIDIA (R) Nsight Compute Command Line Profiler\n"
        "Copyright (c) 2018-2025 NVIDIA Corporation\n"
        "Version 2025.2.0.0 (build 35613519) (public-release)",
    },
    "source": {
        "git_commit": "0f73cb27eb44781aa75cc2519482701c06f03e1e",
        "git_branch": "HEAD",
        "git_dirty": False,
    },
    "run": {
        "repo": "pytorch",
        "device": None,
        "ncu_enabled": False,
        "report_dir": "/data/tj/workspace/ops-baseline-log-softmax-run-v1-20261022/h800.json.work/_sparse_log_softmax_backward_data/fdbb27ef7ce6408690c23d8a0949abd9/ncu_reports",
        "output_path": "/data/tj/workspace/ops-baseline-log-softmax-run-v1-20261022/h800.json.work/_sparse_log_softmax_backward_data/fdbb27ef7ce6408690c23d8a0949abd9/result.json",
        "op_timeout_seconds": 1800.0,
        "reference_chip": "H800",
        "only": ["_sparse_log_softmax_backward_data"],
        "whitelist": None,
        "blacklist": None,
        "selected_ops": ["_sparse_log_softmax_backward_data"],
        "latency_warmup_ms": 25,
        "latency_rep_ms": 100,
        "ncu_warmup_calls": 3,
    },
    "environment_variables": {"CUDA_VISIBLE_DEVICES": "7", "CUDA_DEVICE_ORDER": None},
}


def _plan_fields(entry):
    """Validate a (shape, nnz[, sparse_dim, dim]) descriptor without altering it."""
    entry = tuple(entry)
    if entry and isinstance(entry[0], (tuple, list)):
        shape = tuple(entry[0])
        nnz = entry[1] if len(entry) > 1 else math.prod(shape)
        sparse_dim = entry[2] if len(entry) > 2 else len(shape)
        dim = entry[3] if len(entry) > 3 else len(shape) - 1
    else:
        shape = tuple(entry)
        nnz, sparse_dim, dim = math.prod(shape), len(shape), len(shape) - 1
    nnz, sparse_dim, dim = int(nnz), int(sparse_dim), int(dim)
    if not 0 <= sparse_dim <= len(shape):
        raise ValueError(
            f"invalid case {entry!r}: sparse_dim must be in [0, {len(shape)}]"
        )
    if nnz < 0:
        raise ValueError(f"invalid case {entry!r}: nnz must not be negative")
    if len(shape) == 0:
        # A 0-dim COO tensor is only valid at dim=-1; 0 and -2 are out of range.
        if dim != -1:
            raise ValueError(
                f"invalid case {entry!r}: a scalar shape only accepts dim=-1"
            )
    elif not -len(shape) <= dim < len(shape):
        raise ValueError(
            f"invalid case {entry!r}: dim must be in [{-len(shape)}, {len(shape) - 1}]"
        )
    return shape, nnz, sparse_dim, dim


def _coords(shape, sparse_dim, dim, nnz, device):
    """Support coordinates built on `device` in O(nnz).

    A normalization group is identified by every sparse coordinate except the one along
    `dim`, so the group ids enumerate those axes and the `dim` coordinate is inserted
    back at its own position. Several entries are placed in each group so the reduction
    has a within-group sum; the three operands of a case share this support and differ
    only in their values, otherwise a group of a single random entry sums to zero about
    half the time and the workload degrades to the trivial result.
    """
    shape = tuple(shape)
    if sparse_dim == 0:
        # A 0-dim COO tensor carries indices of shape (0, nnz).
        return torch.empty((0, nnz), dtype=torch.long, device=device)
    if nnz == 0:
        return torch.empty((sparse_dim, 0), dtype=torch.long, device=device)
    if dim < 0:
        dim += len(shape)
    if dim < sparse_dim:
        dim_size = shape[dim]
        group_shape = tuple(shape[a] for a in range(sparse_dim) if a != dim)
        n_groups = math.prod(group_shape) if group_shape else 1
        cols = max(1, min(dim_size, nnz))
        index = torch.arange(nnz, device=device)
        # The used groups are scattered over the other axes and their count is rounded
        # up: with a floor count the trailing partial group can wrap back onto group 0
        # and repeat coordinates that were meant to span the index space.
        used = max(1, -(-nnz // cols))
        step = max(1, n_groups // used)
        group_id = ((index // cols) * step) % n_groups
        dim_coord = index % cols
        parts = torch.unravel_index(group_id, group_shape) if group_shape else ()
        axes = iter(parts)
        flat = torch.stack(
            [dim_coord if a == dim else next(axes) for a in range(sparse_dim)]
        )
    else:
        # `dim` reduces over a dense dim: no sparse coordinate positions form the group,
        # so the entries are spread evenly over the sparse index space.
        spread = math.prod(shape[:sparse_dim])
        step = max(1, spread // nnz)
        flat = torch.stack(
            torch.unravel_index(
                (torch.arange(nnz, device=device) * step) % spread, shape[:sparse_dim]
            )
        )
    return flat.to(torch.long)


def _coo(shape, sparse_dim, dim, nnz, dtype, device):
    # Values are generated directly: benchmark.utils.generate_tensor_input returns None
    # for float64 because consts.FLOAT_DTYPES omits it.
    size = tuple(shape)
    values = torch.randn((nnz,) + size[sparse_dim:], dtype=dtype, device=device)
    indices = _coords(shape, sparse_dim, dim, nnz, device)
    return torch.sparse_coo_tensor(indices, values, torch.Size(size), device=device)


def _case_fn(shape, dtype):
    del dtype
    sparse_shape, nnz, sparse_dim, dim = _plan_fields(shape)
    yield base.BenchmarkCasePlan(
        shape={"sparse_shape": list(sparse_shape), "nnz": nnz},
        params={
            "sparse_dim": sparse_dim,
            "dense_dim": len(sparse_shape) - sparse_dim,
            "dim": dim,
        },
        builder_args=(sparse_shape, nnz, sparse_dim, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    if _uses_external_baseline():
        descriptor = tuple(plan.builder_args)
        binding = grid()[SHAPES.index(descriptor)]
        args, kwargs = build_inputs(binding, dtype, device)
        return (*args, kwargs)
    sparse_shape, nnz, sparse_dim, dim = plan.builder_args
    grad_output = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    output = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    self_ = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    # Flat tuple of positional arguments, then a trailing empty keyword mapping so the
    # value of dim is never mistaken for it.
    return grad_output, output, dim, self_, {}


@dataclass
class _CrossMachineMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None
    candidate_timer: dict | None = None


class SparseLogSoftmaxBackwardBenchmark(OperatorBenchmark):
    def _candidate_call(self):
        registry = base.Config.override_registry
        override = registry.get_override(self.op_name) if registry is not None else None
        op = override if override is not None else self.gems_op
        if not callable(op) or (override is None and op is self.torch_op):
            raise ValueError(
                "A direct flag_gems callable or explicit candidate override is required"
            )
        return op, nullcontext(), override is not None

    def build_inputs(self, case):
        # Bind the actual case before constructing operands. Case IDs come from
        # the existing generic catalog; no alternative numbering is introduced.
        if _uses_external_baseline():
            if type(case.ordinal) is not int or not 0 <= case.ordinal < len(SHAPES):
                raise ValueError("Expected an original core case ordinal")
            binding = grid()[case.ordinal]
            expected_id = (
                "benchmark/test__sparse_log_softmax_backward_data.py::"
                "test__sparse_log_softmax_backward_data_benchmark::core::"
                f"{str(case.dtype).removeprefix('torch.')}::{case.ordinal}"
            )
            if case.case_id != expected_id:
                raise ValueError(
                    "External baseline requires the exact original core case ID"
                )
            self._active_external_case = (case.case_id, binding, case.dtype)
        return super().build_inputs(case)

    @cached_property
    def _nvidia_baseline(self):
        if not all(
            (
                _NV_ARTIFACT_SHA256,
                _NV_PUBLISHED_REVISION,
                _NV_COLLECTION_SOURCE,
                _NV_COLLECTION_ENVIRONMENT,
            )
        ):
            raise ValueError(
                "Native H800 artifact/provenance is pending collection and review"
            )
        value = os.environ.get("FLAGGEMS_NV_BASELINE")
        if not value:
            raise ValueError("Set FLAGGEMS_NV_BASELINE to the reviewed H800 snapshot")
        path = Path(value).resolve(strict=True)
        payload = _aten_read_bytes(
            path, ("pr30:op_pytorch_sparse_log_softmax_h800_perf_baseline")
        )
        if hashlib.sha256(payload).hexdigest() != _NV_ARTIFACT_SHA256:
            raise ValueError(
                "Expected the reviewed native H800 sparse log-softmax snapshot"
            )

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
        # Underscores are part of these exact operator keys.
        entry = data[self.op_name]
        task = data["_collection"]["tasks"][self.op_name]
        # The combined collector's device environment may be the last task on
        # that GPU. Use the independently pinned ORIGINAL per-op worker result,
        # never relabel that last-task environment as this op's provenance.
        environment = _NV_COLLECTION_ENVIRONMENT
        if (
            task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or entry["native_api"] != f"ops.{self.op_name}"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"] != _NV_COLLECTION_SOURCE
            or environment["source"]["git_dirty"] is not False
            or environment["run"]["selected_ops"] != [self.op_name]
            or environment["run"]["repo"] != "pytorch"
            or sum(len(dtypes) for dtypes in entry["shapes"].values()) != 22
        ):
            raise ValueError("Incomplete or inconsistent per-op H800 collection")
        return entry["shapes"], _aten_qualify_metadata(
            {
                "kind": "external_nvidia_native",
                "artifact": str(path),
                "artifact_sha256": _NV_ARTIFACT_SHA256,
                "published_revision": _NV_PUBLISHED_REVISION,
                "collection_source": _NV_COLLECTION_SOURCE,
                "environment": environment,
                "ratio": (
                    "(H800 latency_ms + EPS) / (target latency_ms + EPS); "
                    "no peak normalization"
                ),
                "target_timing_mode": base.Config.mode.value,
            },
            ("pr30:op_pytorch_sparse_log_softmax_h800_perf_baseline"),
        )

    def _external_input_config(self, args, kwargs):
        active = getattr(self, "_active_external_case", None)
        if active is None:
            raise ValueError("No active original core input identity")
        case_id, binding, dtype = active
        expected_device = torch.device(flag_gems.device)
        if expected_device.type != "cpu" and expected_device.index is None:
            expected_device = torch.device(
                expected_device.type, base.torch_device_fn.current_device()
            )
        if (
            len(args) != 4
            or kwargs
            or type(args[2]) is not int
            or args[2] != binding["dim"]
        ):
            raise ValueError(
                "Expected the original sparse log-softmax backward call ABI"
            )
        named = {"grad": args[0], "output": args[1], "self": args[3]}
        for value in named.values():
            if (
                not isinstance(value, torch.Tensor)
                or value.device != expected_device
                or value.dtype != dtype
            ):
                raise ValueError(
                    "Input Tensor/device/dtype must match before byte observation"
                )
        expected = config(binding, dtype)
        for name, value in named.items():
            metadata = expected["inputs"]["tensors"][name]
            if str(value.layout) != metadata["layout"]:
                raise ValueError("Input layout differs before byte observation")
            if value.layout == torch.sparse_coo:
                if (
                    list(value.shape) != metadata["shape"]
                    or value.sparse_dim() != metadata["sparse_dim"]
                    or value.dense_dim() != metadata["dense_dim"]
                    or value._nnz() != metadata["nnz"]
                    or value.is_coalesced() != metadata["is_coalesced"]
                    or value.requires_grad != metadata["requires_grad"]
                    or value.is_conj() != metadata["is_conj"]
                    or value.is_neg() != metadata["is_neg"]
                ):
                    raise ValueError("Sparse metadata differs before byte observation")
            elif (
                list(value.shape) != metadata["part"]["shape"]
                or value.requires_grad != metadata["requires_grad"]
            ):
                raise ValueError("Dense metadata differs before byte observation")
        if (
            expected["case_id"] != case_id
            or input_identity(named) != expected["inputs"]
        ):
            raise ValueError(
                "Actual target input differs from the H800 canonical fixture"
            )
        return expected

    def _import_nvidia_latency(self, args, kwargs):
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "External sparse log-softmax baseline supports forward latency only"
            )
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
                "Expected exactly one H800 record matching the actual COO input"
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
        if _uses_external_baseline() and (
            getattr(self, "_active_external_case", (None,))[0] != case_id
        ):
            raise ValueError("Measurement case ID differs from its constructed inputs")
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
            if metric.latency is not None and (
                isinstance(metric.latency, bool)
                or not isinstance(metric.latency, (int, float))
                or not math.isfinite(metric.latency)
                or metric.latency <= 0
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

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_CASES)


@pytest.mark.underscore_sparse_log_softmax_backward_data
def test__sparse_log_softmax_backward_data_benchmark():
    bench = SparseLogSoftmaxBackwardBenchmark(
        op_name="_sparse_log_softmax_backward_data",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_log_softmax_backward_data,
        gems_op=flag_gems._sparse_log_softmax_backward_data,
        dtypes=_DTYPES,
    )
    bench.run()
