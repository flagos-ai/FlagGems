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
from torch.autograd.function import once_differentiable

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# The default half_to_float=False form is native for float32/float64 only, so the
# fp16/bf16 part of consts.FLOAT_DTYPES is not measurable for this operator; fp64 is
# gated on the target's capability flag.
_BENCH_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)

# Work per case scales with nnz and with the normalization group count, the product
# of every axis other than dim, so both are varied across the descriptors. The nnz=0
# descriptor keeps the empty-reduction workload measurable.
_SPARSE_SOFTMAX_SHAPES = [
    ((1024, 1024), 0, 1),
    ((1024, 1024), 262144, 1),
    ((1024, 1024), 1048576, 1),
    ((4096, 4096), 1048576, 1),
    ((2048, 2048, 64), 1048576, 2),
    ((256, 256, 256), 1048576, 2),
]


RECIPE = "coo-softmax-pcg64-native-width-compact-v1"
DTYPES = (torch.float32, torch.float64)


# Each operator keeps its reader local; the shared benchmark framework is unchanged.
_ATEN_SNAPSHOT_SPECS = {
    "pr29:op_pytorch_sparse_softmax_h800_perf_baseline": {
        "sha256": "8c81710abb79d4ac591b514e3efde080aadfa544ad3899d7ca3fa641886e51f5",
        "artifact": "op_pytorch_sparse_softmax_h800_perf_baseline.json",
        "operators": ["_sparse_softmax", "_sparse_softmax_backward_data"],
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


def _validate(shape, nnz, sparse_dim, dim, dtype):
    shape = tuple(shape)
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Expected a nonempty positive integral shape")
    if type(nnz) is not int or nnz < 0:
        raise ValueError("Expected nonnegative integral nnz")
    if type(sparse_dim) is not int or not 0 <= sparse_dim <= len(shape):
        raise ValueError("Invalid sparse_dim")
    if type(dim) is not int or not 0 <= dim < len(shape):
        raise ValueError("Invalid normalization dim")
    if dtype not in DTYPES:
        raise TypeError("Only original float32/float64 inputs are supported")
    return shape


def seed_hex(kind, shape, nnz, sparse_dim, dim, dtype):
    payload = json.dumps(
        {
            "recipe": RECIPE,
            "kind": kind,
            "shape": list(shape),
            "nnz": nnz,
            "sparse_dim": sparse_dim,
            "dim": dim,
            "dtype": str(dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).digest()[:8].hex()


def _draw(kind, shape, nnz, sparse_dim, dim, dtype):
    shape = _validate(shape, nnz, sparse_dim, dim, dtype)
    seed = bytes.fromhex(seed_hex(kind, shape, nnz, sparse_dim, dim, dtype))
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed, "little")))
    # Same per-axis replacement distribution/range and draw order as the source.
    indices = np.empty((sparse_dim, nnz), dtype=np.int64)
    for axis, extent in enumerate(shape[:sparse_dim]):
        indices[axis] = rng.integers(0, extent, size=nnz, dtype=np.int64)
    # Do not generate float64 through float32: retain genuine requested width.
    npdtype = np.float64 if dtype == torch.float64 else np.float32
    value_shape = (nnz,) + shape[sparse_dim:]
    return shape, rng, indices, value_shape, npdtype


def _compact_cpu(array, dtype):
    result = torch.empty(tuple(array.shape), dtype=dtype, device="cpu")
    result.copy_(torch.from_numpy(array))
    return result


def _canonical_coalesce(indices, values, shape):
    """Stable lexicographic support and typed accumulation of replacement draws.

    This is a NEW fixture recipe, not reproduction of old Torch RNG payloads.
    Stored zero sums are retained. No dense shape-sized allocation is made.
    """
    if indices.shape[1] == 0:
        return indices.copy(), values.copy()
    keys = np.ravel_multi_index(tuple(indices), shape)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    starts = np.r_[True, ordered[1:] != ordered[:-1]]
    groups = np.cumsum(starts, dtype=np.int64) - 1
    result = np.zeros(int(groups[-1]) + 1, dtype=values.dtype)
    # Unbuffered repeated-index addition preserves original duplicate order.
    np.add.at(result, groups, values[order])
    return indices[:, order[starts]].copy(), result


def forward_cpu(shape, nnz, dim, dtype):
    shape, rng, indices, value_shape, npdtype = _draw(
        "forward", shape, nnz, len(shape), dim, dtype
    )
    values = rng.standard_normal(value_shape, dtype=npdtype)
    indices, values = _canonical_coalesce(indices, values, shape)
    return torch.sparse_coo_tensor(
        _compact_cpu(indices, torch.int64),
        _compact_cpu(values, dtype),
        size=shape,
        dtype=dtype,
        device="cpu",
        is_coalesced=True,
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


CASE_SOURCE = "38fd6d1ba0374f40f9c60b6291d951e24e52a4c0"
BENCHMARK_SHA256 = "f44a7b4388513d8e3279183f9b2c6b92f8fc5b76321887ad3786b43aa4d10a88"
OP_NAME = "_sparse_softmax"
SHAPES = tuple(_SPARSE_SOFTMAX_SHAPES)


def grid():
    return [
        {"shape": list(shape), "nnz": nnz, "dim": dim, "ordinal": i}
        for i, (shape, nnz, dim) in enumerate(SHAPES)
    ]


def _checked(binding, dtype):
    if dtype not in DTYPES:
        raise TypeError("Unsupported original sparse softmax dtype")
    if not isinstance(binding, dict) or type(binding.get("ordinal")) is not int:
        raise ValueError("Expected an original descriptor and ordinal")
    i = binding["ordinal"]
    if not 0 <= i < len(SHAPES) or binding != grid()[i]:
        raise ValueError("Binding does not match the original core descriptor")
    return SHAPES[i]


def _cpu_inputs(binding, dtype):
    shape, nnz, dim = _checked(binding, dtype)
    return {"input": forward_cpu(shape, nnz, dim, dtype)}


def key_shape(binding):
    shape, nnz, dim = _checked(binding, DTYPES[0])
    return "coo_" + "x".join(map(str, shape)) + f"_nnz_{nnz}_sd_{len(shape)}_dim_{dim}"


def config(binding, dtype):
    shape, nnz, dim = _checked(binding, dtype)
    return {
        "native_expression": "torch.ops.aten._sparse_softmax.default",
        "case_source_commit": CASE_SOURCE,
        "benchmark": "benchmark/test__sparse_softmax.py",
        "benchmark_sha256": BENCHMARK_SHA256,
        "scope": "core",
        "case_id": (
            "benchmark/test__sparse_softmax.py"
            "::test__sparse_softmax::core::"
            f"{str(dtype).removeprefix('torch.')}::{binding['ordinal']}"
        ),
        "recipe": RECIPE,
        "seed_hex": seed_hex("forward", shape, nnz, len(shape), dim, dtype),
        "requested_nnz": nnz,
        "sparse_dim": len(shape),
        "dim": dim,
        "half_to_float": False,
        "input_device_role": "target",
        "inputs": input_identity(_cpu_inputs(binding, dtype)),
    }


def _copy_dense_physical(x, device):
    """Copy raw storage, then reconstruct geometry and lazy bits; detached by design."""
    assert isinstance(x, torch.Tensor) and x.layout == torch.strided
    shape, stride, offset = tuple(x.shape), x.stride(), x.storage_offset()
    raw = torch.empty(0, dtype=torch.uint8, device=x.device).set_(
        x.untyped_storage(), 0, (x.untyped_storage().nbytes(),), (1,)
    )
    raw = raw.to(device=device).clone()
    y = raw.view(x.dtype).as_strided(shape, stride, offset)
    if x.is_conj():
        y = y.conj()
    if x.is_neg():
        y = torch._neg_view(y)
    assert y.dtype == x.dtype and y.shape == x.shape and y.stride() == stride
    assert y.storage_offset() == offset and y.is_neg() == x.is_neg()
    assert y.is_conj() == x.is_conj() and y.device == torch.device(device)
    return y


def _copy_tensor(x, device, *, preserve_requires_grad=False):
    assert isinstance(x, torch.Tensor)
    if x.layout == torch.sparse_coo:
        y = torch.sparse_coo_tensor(
            _copy_dense_physical(x._indices(), device),
            _copy_dense_physical(x._values(), device),
            tuple(x.shape),
            device=device,
            dtype=x.dtype,
            is_coalesced=x.is_coalesced(),
        )
        assert y.sparse_dim() == x.sparse_dim() and y.dense_dim() == x.dense_dim()
        assert y.is_coalesced() == x.is_coalesced()
    elif x.layout == torch.strided:
        y = _copy_dense_physical(x, device)
    else:
        raise NotImplementedError("Expected COO or dense-negative input")
    if preserve_requires_grad and x.requires_grad:
        y.requires_grad_(True)
    return y


def _forward_native(x, dim, overload, dtype, half_to_float):
    if overload == "default":
        return torch.ops.aten._sparse_softmax.default(x, dim, half_to_float)
    if overload == "int":
        return torch.ops.aten._sparse_softmax.int(x, dim, dtype=dtype)
    raise ValueError(overload)


class _ForwardBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, overload, dtype, half_to_float):
        cpu = _copy_tensor(inp, "cpu")
        out = _forward_native(cpu, dim, overload, dtype, half_to_float)
        ctx.save_for_backward(out)
        ctx.dim, ctx.input_dtype, ctx.input_device = dim, inp.dtype, inp.device
        return _copy_tensor(out, inp.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (output_cpu,) = ctx.saved_tensors
        grad_cpu = _copy_tensor(grad_output, "cpu")
        self_cpu = torch.empty(1, dtype=ctx.input_dtype, device="cpu")
        result = torch.ops.aten._sparse_softmax_backward_data.default(
            grad_cpu, output_cpu, ctx.dim, self_cpu
        )
        # .int's CPU conversion gradient returns the original floating input dtype.
        if result.dtype != ctx.input_dtype:
            result = result.to(dtype=ctx.input_dtype)
        return _copy_tensor(result, ctx.input_device), None, None, None, None


def forward_default(inp, dim, half_to_float=False):
    return _ForwardBridge.apply(inp, dim, "default", None, half_to_float)


def forward_int(inp, dim, dtype=None):
    return _ForwardBridge.apply(inp, dim, "int", dtype, False)


def _write_out(original_out, cpu_result):
    # Only original out-object identity is promised; native may resize/rebind storage.
    target = _copy_tensor(cpu_result, original_out.device)
    original_out.data = target
    return original_out


def forward_out(inp, dim, half_to_float, *, out):
    if out.device != inp.device:
        raise RuntimeError("input/out device mismatch before CPU transport")
    cpu = _copy_tensor(inp, "cpu", preserve_requires_grad=True)
    cpu_out = _copy_tensor(out, "cpu", preserve_requires_grad=True)
    result = torch.ops.aten._sparse_softmax.out(cpu, dim, half_to_float, out=cpu_out)
    assert result is cpu_out
    return _write_out(out, result)


def _cpu_oracle(inp, dim, half_to_float=None, *, dtype=None, out=None):
    if out is not None:
        return forward_out(
            inp, dim, False if half_to_float is None else half_to_float, out=out
        )
    if half_to_float is not None:
        return forward_default(inp, dim, half_to_float)
    return forward_int(inp, dim, dtype)


def _case_fn(shape, dtype):
    # Listing metadata only: one plan per descriptor keeps nnz and dim in the case
    # id/params and defers construction to _build_inputs_fn.
    del dtype
    sparse_shape, nnz, softmax_dim = shape
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape},
        params={"nnz": nnz, "dim": softmax_dim},
        builder_args=(sparse_shape, nnz, softmax_dim),
    )


def _build_random_inputs_fn(plan, dtype, device):
    # Coordinates are drawn with replacement, as in real sparse data, but the tensor
    # is coalesced before timing, so duplicate merging is outside the measurement and
    # both callables receive the same coalesced input. nnz is used as requested;
    # nnz=0 builds a valid empty COO. half_to_float is passed by name because a
    # positional bool binds ScalarType on this Torch.
    sparse_shape, nnz, softmax_dim = plan.builder_args
    indices = torch.stack(
        [
            torch.randint(0, extent, (nnz,), dtype=torch.long, device=device)
            for extent in sparse_shape
        ]
    )
    values = torch.randn(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape, dtype=dtype, device=device
    )
    return inp.coalesce(), {"dim": softmax_dim, "half_to_float": False}


def _uses_external_baseline():
    return bool(os.environ.get("FLAGGEMS_NV_BASELINE")) or flag_gems.vendor_name in {
        "ascend",
        "kunlunxin",
    }


def _build_inputs_fn(plan, dtype, device):
    if not _uses_external_baseline():
        return _build_random_inputs_fn(plan, dtype, device)
    descriptor = tuple(plan.builder_args)
    if descriptor not in SHAPES:
        raise ValueError("External baseline requires an original core descriptor")
    binding = grid()[SHAPES.index(descriptor)]
    expected_params = {
        key: binding[key] for key in binding if key not in {"shape", "ordinal"}
    }
    if plan.shape != {"input": descriptor[0]} or plan.params != expected_params:
        raise ValueError("Benchmark plan changed the original descriptor")
    moved = to_device(_cpu_inputs(binding, dtype), device)
    return moved["input"], {"dim": binding["dim"], "half_to_float": False}


# Bind the official native H800 snapshot to its exact source and ORIGINAL
# per-operator worker environment; do not reuse the last-task environment.
_NV_ARTIFACT_SHA256 = "8c81710abb79d4ac591b514e3efde080aadfa544ad3899d7ca3fa641886e51f5"
_NV_PUBLISHED_REVISION = "277cdd636bbe2cfcf7b54fd45e8116e4eb23f113"
_NV_COLLECTION_SOURCE = "0567cbffdb05aa7987650c55e47738c8183444a7"
_NV_COLLECTION_ENVIRONMENT = {
    "schema_version": 1,
    "collected_at_utc": "2026-10-22T14:06:42.908468+00:00",
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
        "git_commit": "0567cbffdb05aa7987650c55e47738c8183444a7",
        "git_branch": "HEAD",
        "git_dirty": False,
    },
    "run": {
        "repo": "pytorch",
        "device": None,
        "ncu_enabled": False,
        "report_dir": "/data/tj/workspace/ops-baseline-softmax-run-v4-20261022/h800.json.work/_sparse_softmax/d5e00ef0e1ff418ab03a6daed69f6efb/ncu_reports",
        "output_path": "/data/tj/workspace/ops-baseline-softmax-run-v4-20261022/h800.json.work/_sparse_softmax/d5e00ef0e1ff418ab03a6daed69f6efb/result.json",
        "op_timeout_seconds": 1800.0,
        "reference_chip": "H800",
        "only": ["_sparse_softmax"],
        "whitelist": None,
        "blacklist": None,
        "selected_ops": ["_sparse_softmax"],
        "latency_warmup_ms": 25,
        "latency_rep_ms": 100,
        "ncu_warmup_calls": 3,
    },
    "environment_variables": {"CUDA_VISIBLE_DEVICES": "7", "CUDA_DEVICE_ORDER": None},
}


@dataclass
class _CrossMachineMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None
    candidate_timer: dict | None = None


class SparseSoftmaxBenchmark(OperatorBenchmark):
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
                f"benchmark/test_{OP_NAME}.py::test_{OP_NAME}::core::"
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
            path, ("pr29:op_pytorch_sparse_softmax_h800_perf_baseline")
        )
        if hashlib.sha256(payload).hexdigest() != _NV_ARTIFACT_SHA256:
            raise ValueError(
                "Expected the reviewed native H800 sparse softmax snapshot"
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
            or sum(len(dtypes) for dtypes in entry["shapes"].values()) != 12
        ):
            raise ValueError("Incomplete or inconsistent per-op H800 collection")
        return entry["shapes"], {
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
        }

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
            len(args) != 1
            or set(kwargs) != {"dim", "half_to_float"}
            or kwargs != {"dim": binding["dim"], "half_to_float": False}
            or kwargs["half_to_float"] is not False
            or type(kwargs["dim"]) is not int
        ):
            raise ValueError("Expected the original sparse softmax call ABI")
        named = {"input": args[0]}
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
                "External sparse softmax baseline supports forward latency only"
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

    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        if default_shapes is None:
            default_shapes = _SPARSE_SOFTMAX_SHAPES
        super().set_shapes(shape_file_path, default_shapes=default_shapes)


@pytest.mark.underscore_sparse_softmax
def test__sparse_softmax():
    bench = SparseSoftmaxBenchmark(
        op_name="_sparse_softmax",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_softmax.default,
        gems_op=flag_gems._sparse_softmax,
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
