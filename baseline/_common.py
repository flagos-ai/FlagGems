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

"""Shared helpers for the generated ``aten`` baseline modules.

Every module in this directory is one operator from the acceptance list and
exports the contract in ``baseline/__init__.py``. The three things they all
need are collected here so each module stays a short, readable description of
*its own* workload:

``resolve(aten_name)``
    Resolve ``torch.ops.aten.<base>.<overload>`` from the registered name in
    ``conf/operators.yaml`` (the ``for`` field), returning ``None`` plus a
    reason when this PyTorch build does not have it. Operator ids and ATen
    names differ often enough -- ``adaptive_avg_pool2d`` is
    ``aten::_adaptive_avg_pool2d``, ``foreach_mul_tensor_`` is
    ``aten::_foreach_mul_.Tensor`` -- that guessing the name from the id is
    not reliable.

``randn_like_dtype(shape, dtype, device)``
    The same input distribution ``benchmark/base.generate_tensor_input`` uses,
    so a collected record describes the tensor the benchmark will actually
    time. Floating point and complex get ``randn``; integers get a bounded
    ``randint``; bool gets 0/1.

``describe(args, kwargs)``
    Render a call into the ``config.inputs`` structure that
    ``benchmark/baseline.py`` matches against. Keeping the writer and the
    reader of that structure in agreement is the whole point of recording it:
    an ops-baseline record is only usable through ``--base-data`` when its
    description of the inputs matches the live benchmark case exactly.

None of this measures anything; the collector
(``FlagGems-vllm/tools/collect_baseline_nvidia.py``) does the timing and NCU
sampling by importing these modules.
"""

import re

import torch

# Mirrors benchmark/consts.py; kept local so the module imports without the
# benchmark package (the collector adds only this directory to sys.path).
FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16, torch.float64)
INT_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)

# The device a baseline is collected on. A ``device=`` argument is recorded as
# this name even when the tensors themselves are built on ``meta`` for a
# description-only pass: benchmark/baseline.py maps a recorded "cuda" onto the
# benchmark's chosen accelerator (npu, mlu, ...), and matches nothing else.
COLLECTION_DEVICE = "cuda"


def resolve(aten_name):
    """Return (op, None) for an ATen name, or (None, reason) when absent.

    ``aten_name`` is the ``conf/operators.yaml`` ``for`` entry, with or without
    an ``aten::`` prefix: ``"_adaptive_avg_pool2d"``, ``"bitwise_and.Scalar"``,
    ``"_foreach_mul_.Tensor"``. An absent overload is reported rather than
    silently replaced with a different one, because a different overload is a
    different workload and would make the record misleading.
    """
    name = aten_name
    for prefix in ("aten::", "aten."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    base, _, overload = name.partition(".")
    overload = overload or "default"
    packet = getattr(torch.ops.aten, base, None)
    if packet is None:
        return None, f"torch.ops.aten has no operator {base!r}"
    try:
        op = getattr(packet, overload)
    except AttributeError:
        available = ", ".join(sorted(packet.overloads())) or "none"
        return None, (
            f"aten.{base} has no overload {overload!r} "
            f"in torch {torch.__version__} (available: {available})"
        )
    return (op, None) if callable(op) else (None, f"aten.{name} is not callable")


def resolve_path(dotted_path):
    """Return (callable, None) for a dotted torch path, or (None, reason).

    Some benchmarks time a torch-level wrapper (``torch.any``,
    ``torch.nn.functional.adaptive_max_pool3d``) whose signature differs from
    the ATen overload underneath: it may take extra flags, choose an overload
    by argument count, or unpack a tuple return. The grid records that call's
    arguments, so the baseline has to time the same entry point for the
    measurement and the recorded ``config.inputs`` to describe one workload.
    """
    import importlib

    parts = dotted_path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        target = module
        for attribute in parts[split:]:
            target = getattr(target, attribute, None)
            if target is None:
                break
        if callable(target):
            return target, None
    return None, f"cannot resolve {dotted_path!r} in torch {torch.__version__}"


def randn_like_dtype(shape, dtype, device):
    """A value tensor following ``benchmark/base.generate_tensor_input``."""
    shape = tuple(shape)
    if dtype in FLOAT_DTYPES or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=device)
    if dtype in INT_DTYPES:
        info = torch.iinfo(dtype)
        # Bounded well inside the type: the acceptance list has several
        # pointwise ops (pow, lshift, igammac) that overflow or assert on
        # full-range integer inputs, which would cost the whole record.
        low, high = max(info.min, -100), min(info.max, 100)
        return torch.randint(low, high + 1, shape, dtype=dtype, device=device)
    return torch.zeros(shape, dtype=dtype, device=device)


def index_tensor(shape, high, device):
    """An int64 index tensor with values in ``[0, high)``."""
    return torch.randint(0, max(int(high), 1), tuple(shape), dtype=torch.int64, device=device)


# Value constraints a description cannot carry. ``describe`` records a tensor's
# shape, dtype, stride and requires_grad -- never its values -- so an operator
# that asserts on its input range (a probability, a positive weight) needs the
# constraint named in the grid instead. The description is identical either
# way, so constraining the values does not change what the record matches.
_CONSTRAINED = {
    # torch.bernoulli and the BCE losses assert that p lies in [0, 1].
    "unit_interval": lambda shape, dtype, device: torch.rand(
        tuple(shape), dtype=dtype, device=device
    ),
    # Domains requiring a strictly positive input (log, sqrt, gamma).
    "positive": lambda shape, dtype, device: torch.rand(
        tuple(shape), dtype=dtype, device=device
    )
    + 0.5,
    # Gather/scatter indices. The default integer fill spans [-100, 100], and a
    # negative or oversized index is an out-of-bounds device assert that takes
    # the whole collector process down -- not a recoverable per-operator error.
    # Zero is in range for every non-empty dimension, so it is the only width
    # -independent choice; the matcher never compares values.
    "index": lambda shape, dtype, device: torch.zeros(
        tuple(shape), dtype=dtype, device=device
    ),
}


def make_tensor(shape, dtype, device, constraint=None):
    """A tensor for one grid entry, honouring a named value constraint."""
    if constraint:
        try:
            return _CONSTRAINED[constraint](shape, dtype, device)
        except KeyError:
            raise ValueError(f"unknown value constraint {constraint!r}") from None
    return randn_like_dtype(shape, dtype, device)


_TENSOR_MARKER = re.compile(r"^T(?::(?P<dtype>[\w.]+))?(?:!(?P<constraint>\w+))?$")


def _parse_tensor_marker(value):
    """Return (dtype_name, constraint) for a tensor grid entry, else None.

    A tensor entry is ``[marker, shape]`` where marker matches ``T``,
    ``T:<dtype>``, ``T!<constraint>`` or both. Anything else is ordinary data
    the operator takes literally (a list of ints, a list of tensors, ...).
    """
    if not (isinstance(value, list) and len(value) == 2):
        return None
    marker, shape = value
    if not isinstance(marker, str) or not isinstance(shape, (list, tuple)):
        return None
    if not all(isinstance(dim, int) for dim in shape):
        return None
    match = _TENSOR_MARKER.match(marker)
    if match is None:
        return None
    return match.group("dtype"), match.group("constraint")


def build_value(value, dtype, device):
    """Rebuild one argument from its grid entry.

    Grid entries: ``["T", shape]`` a tensor of the case dtype,
    ``["T:<dtype>", shape]`` one pinned to another dtype, either optionally
    carrying ``!<constraint>``; ``{"per_dtype": {...}}`` a scalar the benchmark
    picks from the dtype; ``{"device_arg": True}`` the execution device.
    Everything else is a literal the operator receives unchanged.
    """
    if isinstance(value, dict) and "per_dtype" in value:
        return build_value(value["per_dtype"][str(dtype)], dtype, device)
    if isinstance(value, dict) and value.get("dtype_arg"):
        # The benchmark passes the case dtype straight through (window and cast
        # operators take a dtype= argument), so it follows DTYPES rather than
        # being pinned to one value.
        return dtype
    if isinstance(value, dict) and value.get("device_arg"):
        # The benchmark passes its execution device here, and the record has to
        # describe the call as it runs on the collection device: the matcher's
        # _matches_execution_device bridges only a recorded "cuda" onto the
        # benchmark's chosen accelerator, so a record saying "meta" can never
        # match. A meta build is a description-only build -- the tensors need
        # no storage, but this argument must still name the real device.
        return COLLECTION_DEVICE if str(device) == "meta" else device
    marker = _parse_tensor_marker(value)
    if marker is not None:
        dtype_name, constraint = marker
        tensor_dtype = dtype
        if dtype_name:
            tensor_dtype = getattr(torch, dtype_name.split(".")[-1])
        return make_tensor(value[1], tensor_dtype, device, constraint)
    if isinstance(value, list):
        return [build_value(item, dtype, device) for item in value]
    return value


def _describe_value(value):
    if isinstance(value, torch.Tensor):
        described = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.layout == torch.strided:
            described["stride"] = list(value.stride())
        described["requires_grad"] = value.requires_grad
        return described
    if isinstance(value, (list, tuple)):
        return [_describe_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _describe_value(item) for key, item in value.items()}
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return str(value)
    return value


def describe(args, kwargs):
    """The ``config.inputs`` payload read by ``benchmark/baseline.py``.

    Mirrors ``benchmark.baseline.describe_call``. Tensors are reduced to
    shape/dtype/stride/requires_grad and scalars are kept by value and type,
    which is exactly what the matcher compares.
    """
    return {"args": _describe_value(list(args)), "kwargs": _describe_value(kwargs)}


def config_from_call(args, kwargs, dims=None, outputs=None):
    """A full ``config`` dict: matched inputs plus optional human-facing fields."""
    payload = {"inputs": describe(args, kwargs)}
    if outputs is not None:
        payload["outputs"] = _describe_value(outputs)
    if dims is not None:
        payload["dims"] = dims
    return payload


def shape_key(shape):
    """Primary JSON key for a collection point: the shape as a list."""
    return [int(x) for x in shape]
