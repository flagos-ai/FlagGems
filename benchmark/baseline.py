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

"""Optional ops-baseline matching; no tensor values or device imports needed."""

import json
import math
import re
import warnings


def positive_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def describe_call(args, kwargs):
    """Describe inputs before either implementation can mutate them."""
    import torch

    def describe(value):
        if isinstance(value, torch.Tensor):
            result = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.layout == torch.strided:
                result["stride"] = list(value.stride())
            result["requires_grad"] = value.requires_grad
            return result
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        return value

    return {"args": describe(args), "kwargs": describe(kwargs)}


def _matches(expected, actual):
    if isinstance(expected, dict):
        if "scalar" in expected:
            return _matches(expected["scalar"], actual)
        if not isinstance(actual, dict):
            return False
        if "shape" in expected:
            # Ignore source-only tensor annotations, but compare every available
            # identity field, including stride and dtype for auxiliary tensors.
            return all(
                key not in expected or _matches(expected[key], actual.get(key))
                for key in ("shape", "dtype", "stride", "requires_grad")
            )
        return set(expected) == set(actual) and all(
            _matches(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(expected) == len(actual)
            and all(_matches(a, b) for a, b in zip(expected, actual))
        )
    return type(expected) is type(actual) and expected == actual


def _device_identity(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([a-z][a-z0-9_]*)(?::([0-9]+))?", value)
    if match is None:
        return None
    return match[1], int(match[2] or 0)


def _matches_execution_device(expected, actual, target_device):
    """Map source CUDA's default device only to this benchmark's chosen target.

    CPU placement, non-default source devices, and devices other than the
    chosen target remain significant. This applies only to a named `device`
    parameter, never to arbitrary strings or tensor metadata.
    """
    if isinstance(expected, dict) and "scalar" in expected:
        expected = expected["scalar"]
    source = _device_identity(expected)
    target = _device_identity(target_device)
    return (
        source == ("cuda", 0)
        and target is not None
        and target[0] != "cpu"
        and _device_identity(actual) == target
    )


def _matches_parameters(expected, actual, target_device):
    if (
        not isinstance(expected, dict)
        or not isinstance(actual, dict)
        or set(expected) != set(actual)
    ):
        return False
    return all(
        (
            name == "device"
            and _matches_execution_device(value, actual[name], target_device)
        )
        or _matches(value, actual[name])
        for name, value in expected.items()
    )


def _matches_inputs(inputs, call, named_call, target_device):
    if "args" in inputs or "kwargs" in inputs:
        return (
            set(inputs) == set(call)
            and _matches(inputs["args"], call["args"])
            and _matches_parameters(inputs["kwargs"], call["kwargs"], target_device)
        )
    # Missing parameters can change semantics even when tensor shapes agree.
    # Defaults have already been bound by the benchmark; require full evidence.
    if not inputs or named_call is None:
        return False
    return _matches_parameters(inputs, named_call, target_device)


def _vendor_name(value):
    # ops-baseline and FlagGems use different spellings for this same vendor.
    return "thead" if value == "t-head" else value


def _chip_matches(model, device_name):
    if not isinstance(model, str) or not model or not device_name:
        return False
    # Preserve suffixes such as H20-3e, but never confuse H20 with H200.
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(model)}(?![a-z0-9])", device_name, re.IGNORECASE
        )
        is not None
    )


def _legacy_shape_key(key, call):
    """Only a direct unary tensor call can use shape-only legacy records.

    Extra tensors, scalars, keywords, or nested containers need input metadata
    to distinguish incompatible operator ABIs (for example packed vs split).
    """
    args = call.get("args", [])
    if len(args) != 1 or call.get("kwargs"):
        return False
    tensor = args[0]
    if not isinstance(tensor, dict) or "shape" not in tensor:
        return False
    try:
        return json.loads(key) == tensor["shape"]
    except (TypeError, ValueError):
        return False


class BaselineData:
    def __init__(self, data, source):
        self.data = data
        self.source = str(source)
        self._records = {}

    @classmethod
    def load(cls, path):
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("top-level baseline must be a JSON object")
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Cannot load --base-data {path}: {exc}; using native baseline"
            )
            return None
        return cls(data, path)

    def lookup(self, op_name, dtype, call, named_call=None, *, target_device=None):
        """Return (record, shape key, miss reason), never guess an ambiguous match.

        Supports ops-baseline PyTorch args/kwargs metadata, named SGLang/vLLM
        config.inputs when parameter binding is available, and legacy simple
        shape keys without input metadata. Leading-underscore ops are valid.
        """
        entry = self.data.get(op_name)
        if not isinstance(entry, dict):
            return None, None, "operator_missing"
        shapes = entry.get("shapes")
        if not isinstance(shapes, dict):
            return None, None, "shape_missing"
        index_key = (op_name, str(dtype))
        if index_key not in self._records:
            self._records[index_key] = [
                (key, per_dtype[str(dtype)])
                for key, per_dtype in shapes.items()
                if isinstance(per_dtype, dict)
                and isinstance(per_dtype.get(str(dtype)), dict)
            ]
        candidates = []
        for key, record in self._records[index_key]:
            config = record.get("config")
            if config is not None and not isinstance(config, dict):
                continue
            inputs = config.get("inputs") if isinstance(config, dict) else None
            if (
                isinstance(config, dict)
                and "inputs" in config
                and not isinstance(inputs, dict)
            ):
                continue
            if isinstance(inputs, dict):
                matched = _matches_inputs(inputs, call, named_call, target_device)
            else:
                matched = _legacy_shape_key(key, call)
            if matched:
                candidates.append((record, key))
        if len(candidates) != 1:
            reason = "ambiguous_match" if candidates else "shape_or_dtype_missing"
            return None, None, reason
        record, key = candidates[0]
        latency = record.get("latency_ms")
        if latency is None and positive_number(record.get("T_us")):
            latency = record["T_us"] / 1000
        if not positive_number(latency):
            return None, None, "invalid_latency"
        return {**record, "latency_ms": latency}, key, None

    def scaling_factor(self, vendor, chip, unit, dtype):
        scaling = self.data.get("_scaling_factors")
        if not isinstance(scaling, dict) or not isinstance(scaling.get("chips"), list):
            return None
        candidates = [
            c
            for c in scaling["chips"]
            if isinstance(c, dict)
            and _vendor_name(c.get("vendor")) == _vendor_name(vendor)
        ]
        matched = [c for c in candidates if _chip_matches(c.get("chip"), chip)]
        if len(matched) == 1:
            selected = matched[0]
        elif not chip and len(candidates) == 1:
            selected = candidates[0]
        else:
            return None
        if not isinstance(unit, str):
            return None
        resource = {"mem": "bandwidth_gbps", "cuda": "vector", "tensor": "tensor"}.get(
            unit
        )
        dtype_key = {
            "torch.float16": "fp16",
            "torch.bfloat16": "bf16",
            "torch.float32": "fp32",
            "torch.float64": "fp64",
            "torch.int8": "int8",
            "torch.float8_e4m3fn": "fp8",
            "torch.float8_e5m2": "fp8",
        }.get(str(dtype))
        if resource is None:
            return None
        for preference in ("measured", "nominal"):
            block = selected.get(preference)
            if not isinstance(block, dict):
                continue
            value = block.get(resource)
            if resource != "bandwidth_gbps":
                value = value.get(dtype_key) if isinstance(value, dict) else None
            if positive_number(value):
                return value
        return None

    def apply(
        self,
        metric,
        op_name,
        dtype,
        call,
        named_call,
        vendor,
        chip,
        *,
        target_device=None,
    ):
        if not positive_number(metric.latency):
            metric.base_miss_reason = "invalid_candidate_latency"
            return
        record, key, reason = self.lookup(
            op_name, dtype, call, named_call, target_device=target_device
        )
        metric.base_miss_reason = reason
        if record is None:
            return
        factor = self.scaling_factor(
            vendor, chip, record.get("bottle_neck_unit"), dtype
        )
        raw_ratio = record["latency_ms"] / metric.latency
        compared = raw_ratio / (factor or 1.0)
        if not positive_number(compared):
            metric.base_miss_reason = "invalid_speedup"
            return
        metric.compared_speedup = compared
        metric.base_raw_speedup = raw_ratio
        metric.base_data_source = self.source
        metric.base_shape_key = key
        metric.base_latency_ms = record["latency_ms"]
        metric.base_scaling_factor = factor
