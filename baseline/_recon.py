# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Shared input reconstructor for log-driven baseline modules.
# Starts with "_" so check_baseline.py skips it as a non-operator module.
"""Materialize a benchmark call from a recorded ``shape_detail`` snapshot.

``shape_detail`` is what ``benchmark/base.py::record_shapes`` wrote: every
tensor became its ``size()`` (an int list), scalars/str/bool passed through,
lists/dicts recursed.  The snapshot alone cannot tell a tensor shape ``[5]``
from an ``IntList`` argument ``[1]``; the aten schema disambiguates by the
declared type at each positional slot or keyword name.
"""

import re

import torch

_FLOAT = (torch.float16, torch.float32, torch.bfloat16, torch.float64)

_ALIASES = {
    "Tensor(a!)": "Tensor",
    "Tensor(a)": "Tensor",
    "Tensor(a!)[]": "TensorList",
    "Tensor[]": "TensorList",
    "Tensor?[]": "TensorList",
    "SymInt[]": "IntList",
    "int[]": "IntList",
    "SymInt": "int",
    "number": "Scalar",
}

# aten ScalarType names recorded by record_shapes (str(torch.dtype)).
_DTYPE_BY_NAME = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
    "torch.complex64": torch.complex64,
    "torch.complex128": torch.complex128,
}

# Reduction enum spellings seen in logs for int reduction arguments.
_REDUCTION = {"none": 0, "mean": 1, "sum": 2}

# Tensors that must be integer indices regardless of the benchmark dtype.
_INDEX_NAMES = {
    "index",
    "indices",
    "target",
    "targets",
    "input_lengths",
    "target_lengths",
}


def _clean_type(t):
    t = re.sub(r"\(\w+!?\)", "", t).strip()
    t = _ALIASES.get(t, t)
    if t.startswith("int[") or t.startswith("SymInt["):
        return "IntList", False
    optional = t.endswith("?")
    if optional:
        t = t[:-1]
    t = _ALIASES.get(t, t)
    return t, optional


def _split_top(text):
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def parse_schema(schema):
    """-> list of {name, type, kw_only, has_default} in declared order."""
    m = re.match(r"aten::([A-Za-z0-9_.]+)\((.*)\)\s*->\s*(.+)$", schema.strip(), re.S)
    if not m:
        raise ValueError("cannot parse schema: %s" % schema)
    args, kw_only = [], False
    for raw in _split_top(m.group(2)):
        if raw == "*":
            kw_only = True
            continue
        has_default = "=" in raw
        if has_default:
            raw = raw.split("=", 1)[0].strip()
        mm = re.match(r"^(.*?)\s*([A-Za-z_][A-Za-z0-9_]*)$", raw)
        if not mm:
            raise ValueError("cannot parse arg %r" % raw)
        args.append(
            {
                "name": mm.group(2),
                "type": mm.group(1).strip(),
                "kw_only": kw_only,
                "has_default": has_default,
            }
        )
    return args


def _is_shape(value):
    return isinstance(value, list) and all(isinstance(x, int) for x in value)


def _mk_tensor(shape, dtype, device):
    if dtype in _FLOAT:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype.is_complex:
        real = torch.randn(shape, device=device)
        imag = torch.randn(shape, device=device)
        return torch.complex(real, imag).to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(torch.bool)
    # integer dtypes
    return torch.randint(0, 8, shape, dtype=dtype, device=device)


def _list_len_hint(typ):
    """SymInt[2] / int[3] -> fixed length, else None."""
    m = re.search(r"\[(\d+)\]", typ)
    return int(m.group(1)) if m else None


def _materialize_value(value, typ, dtype, device, name=None):
    base, _opt = _clean_type(typ)
    raw = typ.replace("(a!)", "").replace("(a)", "").rstrip("?").strip()

    # ScalarType argument: record_shapes stored str(torch.dtype).
    if raw == "ScalarType":
        if isinstance(value, str):
            return _DTYPE_BY_NAME.get(value)
        return value
    if raw in ("Layout", "MemoryFormat"):
        return None  # let aten pick the default; string form isn't accepted
    if raw == "Device":
        return str(value) if value is not None else None

    if value is None:
        return None

    if base == "Tensor":
        if _is_shape(value):
            td = dtype
            if name in _INDEX_NAMES:
                td = torch.int64
            return _mk_tensor(value, td, device)
        return value  # already a scalar/other; pass through
    if base == "TensorList":
        if isinstance(value, list):
            td = torch.int64 if name in _INDEX_NAMES else dtype
            return [_mk_tensor(s, td, device) if _is_shape(s) else s for s in value]
        return value
    if base == "IntList":
        n = _list_len_hint(typ)
        if isinstance(value, int):
            return [value] * (n or 1)
        if isinstance(value, list):
            return value
        return value
    if base == "int":
        # Reduction-style string enum recorded for an int slot.
        if isinstance(value, str):
            return _REDUCTION.get(value, 1)
        return value
    # float / bool / str / Scalar: pass the literal
    return value


def split_detail(detail):
    """-> (positional_args_list, kwargs_dict)."""
    if isinstance(detail, dict):
        return [], dict(detail)
    if (
        isinstance(detail, list)
        and len(detail) == 2
        and isinstance(detail[1], dict)
        and isinstance(detail[0], list)
    ):
        return list(detail[0]), dict(detail[1])
    if isinstance(detail, list):
        return list(detail), {}
    return [detail], {}


def missing_required(schema, detail):
    """Required schema parameters absent from a recorded snapshot.

    ``record_shapes`` sometimes stored fewer arguments than the operator
    declares (``avg_pool2d_backward`` kept only ``grad_output``; every ``.out``
    overload dropped ``out``).  Those calls cannot be rebuilt from the log
    without inventing inputs, so the caller routes them to needs_ai instead.
    """
    schema_args = parse_schema(schema)
    pos, kwargs = split_detail(detail)
    missing = []
    for i, spec in enumerate(schema_args):
        if spec["name"] in kwargs:
            continue
        covered = (not spec["kw_only"]) and i < len(pos)
        if not covered and not spec["has_default"]:
            missing.append(spec["name"])
    return missing


# Fallbacks for required parameters the log snapshot dropped. Values mirror the
# aten defaults for the same parameter name on sibling overloads so the call
# stays representative rather than invented.
_GAP_INT = {"p": 1, "n": 1, "dim": 0, "reduction": 1, "blank": 0, "groups": 1}
_GAP_FLOAT = {
    "lr": 0.001,
    "eps": 1e-08,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.0,
    "lr_decay": 0.0,
    "momentum": 0.0,
    "alpha": 1.0,
    "beta": 1.0,
}
_GAP_BOOL = {
    "maximize": False,
    "amsgrad": False,
    "ceil_mode": False,
    "align_corners": False,
    "keepdim": False,
}
_GAP_STR = {"reduce": "sum"}


def _fill_gap(spec, built_tensors, dtype, device):
    """Build a value for a required parameter missing from the snapshot.

    record_shapes dropped some arguments entirely (avg_pool2d_backward kept
    only grad_output; every .out overload lost out). Tensor gaps copy the
    geometry of the call's primary tensor, which is what the benchmark did;
    scalar gaps take the aten default for that parameter name.
    """
    name = spec["name"]
    base, _opt = _clean_type(spec["type"])
    raw = spec["type"].replace("(a!)", "").replace("(a)", "").rstrip("?").strip()

    if raw == "bool[2]":
        return [True, False]
    if base == "Tensor":
        ref = next((t for t in built_tensors if torch.is_tensor(t)), None)
        if ref is None:
            return None
        if name in _INDEX_NAMES:
            hi = max(1, int(min(ref.shape)))
            return torch.randint(
                0, hi, tuple(ref.shape), dtype=torch.int64, device=device
            )
        if name.startswith("out"):
            return torch.empty_like(ref)
        return _mk_tensor(list(ref.shape), ref.dtype, device)
    if base == "TensorList":
        ref = next((t for t in built_tensors if torch.is_tensor(t)), None)
        return [torch.empty_like(ref)] if ref is not None else []
    if base == "IntList":
        n = _list_len_hint(spec["type"]) or 1
        return [_GAP_INT.get(name, 0)] * n
    if base == "int":
        return _GAP_INT.get(name, 0)
    if base == "float":
        return _GAP_FLOAT.get(name, 0.0)
    if base == "bool":
        return _GAP_BOOL.get(name, False)
    if base == "str":
        return _GAP_STR.get(name, "sum")
    if base == "Scalar":
        return 1.0
    return None


def materialize(schema, detail, dtype, device, fill_gaps=True):
    schema_args = parse_schema(schema)
    pos_args, kwargs = split_detail(detail)

    out_args = []
    out_kwargs = {}
    consumed = 0
    for i, value in enumerate(pos_args):
        if i < len(schema_args):
            spec = schema_args[i]
            typ, nm, kw_only = spec["type"], spec["name"], spec["kw_only"]
        else:
            typ, nm, kw_only = "Tensor", None, False
        built = _materialize_value(value, typ, dtype, device, nm)
        # record_shapes flattened everything positionally, but a parameter
        # declared after "*" must be passed by keyword or aten reports an
        # arity error (e.g. scatter.reduce's `reduce`, linalg_vecdot.out's
        # `out`).
        if kw_only and nm:
            out_kwargs[nm] = built
        else:
            out_args.append(built)
        consumed = i + 1

    by_name = {a["name"]: a["type"] for a in schema_args}
    for name, value in kwargs.items():
        typ = by_name.get(name, "Tensor")
        out_kwargs[name] = _materialize_value(value, typ, dtype, device, name)

    if fill_gaps:
        # Walk the declared order and supply anything required but absent.
        tensors = [a for a in out_args if torch.is_tensor(a)]
        tensors += [v for v in out_kwargs.values() if torch.is_tensor(v)]
        for i, spec in enumerate(schema_args):
            nm = spec["name"]
            if nm in out_kwargs:
                continue
            covered = (not spec["kw_only"]) and i < consumed
            if covered or spec["has_default"]:
                continue
            gap = _fill_gap(spec, tensors, dtype, device)
            if spec["kw_only"]:
                out_kwargs[nm] = gap
            elif i == len(out_args):
                out_args.append(gap)
                if torch.is_tensor(gap):
                    tensors.append(gap)
            else:
                out_args.insert(i, gap)
                if torch.is_tensor(gap):
                    tensors.append(gap)

    return tuple(out_args), out_kwargs
