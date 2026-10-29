import hashlib
import json
from pathlib import Path

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base, consts
from .baseline import describe_call, positive_number

_LOG2E = 1.4426950408889634
_OVERRIDEABLE_H800_SHA256 = (
    "57292fff3991a06e5b39fc131ff66213db2a556227433b61a3147144d052d97c"
)
_OVERRIDEABLE_H800_SOURCE = "cd263d21c3ad5bfc541ba3f68afa5a76a9c453cd"
_OVERRIDEABLE_CORE_SHAPES = [
    (4, 32, 1024, 64),
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
    (4, 32, 4096, 128),
    (4, 32, 8192, 128),
]


def _native_dense_p0_nine_tuple(
    query,
    key,
    value,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    assert attn_bias is None and dropout_p == 0.0 and return_debug_mask is False
    assert scale is None and query.shape == key.shape == value.shape
    out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention.default(
        query, key, value, None, True, 0.0, is_causal
    )
    batch, _, length, _ = query.shape
    return (
        out.contiguous(),
        (lse[..., :length] * _LOG2E).contiguous(),
        torch.tensor(
            [0, length] * batch, dtype=torch.int32, device=query.device
        ).reshape(batch, 2),
        torch.tensor([0, length] * batch, dtype=torch.int32, device=key.device).reshape(
            batch, 2
        ),
        length,
        length,
        torch.tensor([0], dtype=torch.int64, device=query.device),
        torch.tensor([0], dtype=torch.int64, device=query.device),
        torch.empty(0, dtype=query.dtype, device=query.device),
    )


def torch_ref(
    query,
    key,
    value,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    scale=None,
):
    if flag_gems.vendor_name == "nvidia":
        return _native_dense_p0_nine_tuple(
            query,
            key,
            value,
            attn_bias,
            dropout_p,
            is_causal,
            return_debug_mask,
            scale=scale,
        )
    if flag_gems.vendor_name in {"hygon", "metax"}:
        raise ValueError(
            "Backend uses exact static H800 reference; native timer must be skipped"
        )
    return flag_gems.ops.scaled_dot_product_attention_forward(
        query, key, value, is_causal=is_causal
    )


def _overrideable_h800_config(binding, dtype):
    shape = list(binding["shape"])
    batch, heads, length, _ = shape
    stride = [1] * 4
    for i in range(2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    tensor = dict(
        shape=shape,
        stride=stride,
        storage_offset=0,
        dtype=str(dtype),
        requires_grad=False,
    )
    return {
        "native_expression": "native_ATen_efficient_attention_dense_dropout0_nine_tuple",
        "literal_overrideable_stub": False,
        "math_API": "aten::_scaled_dot_product_efficient_attention",
        "logsumexp_convention": "log2 = native natural LSE * LOG2E; trim padding; contiguous",
        "metadata_recipe": (
            "existing dense p0 API cum_seq[0,L]*B int32, maxQ=maxK=L, "
            "seed/offset target int64[0], debugempty"
        ),
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "source_benchmark_sha256": "47e8b5cc73898f03a300e1dce5892c1333d3eab3cd26a40490a8f13e11241799",
        "source_builder_ast_sha256": "43f272587d2de9c060cac48a2ef2bd50e4b9557632f37da2f073c3c37326f5e8",
        "benchmark": "benchmark/test_scaled_dot_product_fused_attention_overrideable.py",
        "builder": "_build_inputs_fn",
        "scope": "core",
        "descriptor": binding,
        "recipe": {
            "query": "torch.randn",
            "key": "torch.randn",
            "value": "torch.randn",
            "on_requested_device": True,
            "seed": "original unseeded builder",
        },
        "inputs": {
            "args": [
                tensor.copy(),
                tensor.copy(),
                tensor.copy(),
                None,
                0.0,
                binding["is_causal"],
                False,
            ],
            "kwargs": {"scale": None},
        },
        "outputs": [
            {"name": "output", "shape": shape, "dtype": str(dtype)},
            {
                "name": "logsumexp",
                "shape": [batch, heads, length],
                "dtype": "torch.float32",
                "base": 2,
            },
            {"name": "cum_seq_q", "shape": [batch, 2], "dtype": "torch.int32"},
            {"name": "cum_seq_k", "shape": [batch, 2], "dtype": "torch.int32"},
            {"name": "max_q", "value": length},
            {"name": "max_k", "value": length},
            {"name": "philox_seed", "shape": [1], "dtype": "torch.int64"},
            {"name": "philox_offset", "shape": [1], "dtype": "torch.int64"},
            {"name": "debug_attn_mask", "shape": [0], "dtype": str(dtype)},
        ],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }


class _OverrideableH800Reference:
    def _require_composite_static(self, args, kwargs, case_id):
        if not base.Config.skip_native or base.Config.base_data is None:
            raise ValueError(
                "Backend overrideable attention requires exact native-composite H800 --base-data"
            )
        if (
            self.op_name != "_scaled_dot_product_fused_attention_overrideable"
            or self.torch_op is not torch_ref
        ):
            raise ValueError("Original operator and reviewed native composite required")
        if (
            base.Config.bench_level != consts.BenchLevel.CORE
            or getattr(self, "fresh_inputs", False)
            or self.is_backward
        ):
            raise ValueError("Original forward shared-input core required")
        if len(args) != 7 or kwargs != {"scale": None}:
            raise ValueError("Original seven positional inputs and scale=None required")
        query, key, value, bias, dropout, causal, debug = args
        shape = tuple(query.shape)
        if (
            shape not in _OVERRIDEABLE_CORE_SHAPES
            or query.dtype not in consts.FLOAT_DTYPES
            or bias is not None
            or type(dropout) is not float
            or dropout != 0.0
            or type(causal) is not bool
            or debug is not False
        ):
            raise ValueError("Original dense dropout0 attention recipe required")
        target = torch.device(self.device)
        if target.index is None:
            target = torch.device(target.type, torch_device_fn.current_device())
        for tensor in (query, key, value):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != query.dtype
                or tensor.device != target
                or not tensor.is_contiguous()
                or tensor.storage_offset() != 0
                or tensor.requires_grad
            ):
                raise ValueError("Original separate contiguous target q/k/v required")
        if len({x.untyped_storage().data_ptr() for x in (query, key, value)}) != 3:
            raise ValueError("Original q/k/v must not alias")
        dtype = str(query.dtype).split(".")[-1]
        expected_id = (
            "benchmark/test_scaled_dot_product_fused_attention_overrideable.py::"
            "test_scaled_dot_product_fused_attention_overrideable"
            f"[{causal}]::core::{dtype}::{_OVERRIDEABLE_CORE_SHAPES.index(shape)}"
        )
        if case_id != expected_id:
            raise ValueError("Original ordered joint core identity required")
        baseline = base.Config.base_data
        if (
            hashlib.sha256(Path(baseline.source).read_bytes()).hexdigest()
            != _OVERRIDEABLE_H800_SHA256
        ):
            raise ValueError("Exact reviewed native-composite H800 artifact required")
        data = baseline.data
        op = self.op_name
        task = data["_collection"]["tasks"][op]
        env = data["_environments"][str(task["device"])]
        if (
            data[op]["native_api"] != "ops." + op
            or task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
            or env["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or env["source"]["git_commit"] != _OVERRIDEABLE_H800_SOURCE
            or env["source"]["git_dirty"] is not False
            or env["run"]["repo"] != "pytorch"
        ):
            raise ValueError(
                "Completed clean native ATen composite H800 provenance required"
            )
        expected = _overrideable_h800_config(
            dict(shape=list(shape), is_causal=causal), query.dtype
        )
        call = describe_call(args, kwargs)
        expected_call = {
            "args": [
                (
                    {k: v for k, v in x.items() if k != "storage_offset"}
                    if isinstance(x, dict)
                    else x
                )
                for x in expected["inputs"]["args"]
            ],
            "kwargs": {"scale": None},
        }
        record, key, reason = baseline.lookup(
            op, query.dtype, call, target_device=self.device
        )
        if (
            record is None
            or reason is not None
            or record["config"] != expected
            or call != expected_call
            or not positive_number(record.get("T_us"))
            or not positive_number(record.get("latency_ms"))
            or abs(record["T_us"] / 1000 - record["latency_ms"]) > 1e-12
        ):
            raise ValueError(
                "Actual call must select its exact original native-composite H800 record"
            )
        return baseline, json.dumps(call, sort_keys=True), record["latency_ms"], key

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name in {"hygon", "metax"} and op is self.torch_op:
            raise ValueError("Backend requires explicit supported CPU-oracle candidate")
        return op, dispatch, overridden

    def _measure_input(self, input, case_id=None):
        if flag_gems.vendor_name in {"hygon", "metax"}:
            self._composite_checked = self._require_composite_static(
                *self.unpack_to_args_kwargs(input), case_id
            )
        return super()._measure_input(input, case_id=case_id)

    def _apply_external_baseline(self, metric, dtype, context):
        if flag_gems.vendor_name not in {"hygon", "metax"}:
            return super()._apply_external_baseline(metric, dtype, context)
        baseline, captured, latency, key = self._composite_checked
        if (
            context is None
            or context[0] is not baseline
            or base.Config.base_data is not baseline
            or json.dumps(context[1], sort_keys=True) != captured
            or hashlib.sha256(Path(baseline.source).read_bytes()).hexdigest()
            != _OVERRIDEABLE_H800_SHA256
        ):
            raise ValueError(
                "Native-composite H800 source/call changed around candidate timer"
            )
        super()._apply_external_baseline(metric, dtype, context)
        if (
            metric.error_msg is not None
            or metric.latency_base is not None
            or metric.base_miss_reason is not None
            or metric.base_data_source != baseline.source
            or metric.base_shape_key != key
            or metric.base_latency_ms != latency
        ):
            raise ValueError("Native-composite H800 baseline consumption failed")


class AttentionBenchmark(_OverrideableH800Reference, base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (2, 8, 64, 64),
        (4, 8, 128, 64),
        (2, 8, 128, 128),
        (1, 16, 256, 64),
    ]

    def set_more_shapes(self):
        return None


def _case_fn_for(is_causal):
    def case_fn(shape, dtype):
        del dtype
        dims = tuple(shape)
        yield base.BenchmarkCasePlan(
            shape={"query": list(dims), "key": list(dims), "value": list(dims)},
            params={"is_causal": is_causal},
            builder_args=(dims,),
        )

    return case_fn


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    query, key, value = (
        torch.randn(shape, device=device, dtype=dtype) for _ in range(3)
    )
    return (
        query,
        key,
        value,
        None,
        0.0,
        plan.params["is_causal"],
        False,
        {"scale": None},
    )


@pytest.mark.scaled_dot_product_fused_attention_overrideable
@pytest.mark.underscore_scaled_dot_product_fused_attention_overrideable
@pytest.mark.skip_native(
    vendors=["hygon", "metax"],
    reason="Absent overrideable hook; consume exact native ATen composite H800 records",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_fused_attention_overrideable(is_causal):
    bench = AttentionBenchmark(
        op_name="_scaled_dot_product_fused_attention_overrideable",
        case_fn=_case_fn_for(is_causal),
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
