import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch

from . import base, consts

_MKLDNN_RNN_LAYER_SHAPES = [
    (16, 4, 32),
    (32, 8, 64),
    (64, 16, 128),
]

# mkldnn_rnn_layer is LSTM (mode=2), single-layer, unidirectional.
_MODE = 2
_NUM_LAYERS = 1
_BIDIRECTIONAL = False
_BATCH_FIRST = False


@dataclass
class _MkldnnMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _H800Latency:
    @staticmethod
    def _input_config(args, kwargs):
        def value(x):
            if isinstance(x, torch.Tensor):
                return dict(
                    shape=list(x.shape),
                    dtype=str(x.dtype),
                    stride=list(x.stride()),
                    requires_grad=x.requires_grad,
                )
            if isinstance(x, (tuple, list)):
                return [value(y) for y in x]
            if isinstance(x, dict):
                return {k: value(v) for k, v in x.items()}
            return x

        return dict(
            inputs=dict(args=value(args), kwargs=value(kwargs)),
            source_commit="9d57a23b4386bd669152a935fe88a96db907da7c",
            benchmark="benchmark/test_mkldnn_rnn_layer.py",
            descriptor=list(args[0].shape),
            scope="core",
            variant=_MKLDNN_RNN_LAYER_SHAPES.index(tuple(args[0].shape)),
        )

    @cached_property
    def _h800(self):
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique(pairs):
            result = {}
            for key, val in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline key: {key}")
                result[key] = val
            return result

        def reject(value):
            raise ValueError(f"Non-finite baseline constant: {value}")

        data = json.loads(payload, object_pairs_hook=unique, parse_constant=reject)
        op = data["mkldnn_rnn_layer"]
        if op["native_api"] != "ops.mkldnn_rnn_layer":
            raise ValueError("Expected ops-baseline native LSTM composition")
        task = data["_collection"]["tasks"]["mkldnn_rnn_layer"]
        if (
            task["status"] != "completed"
            or isinstance(task["returncode"], bool)
            or not isinstance(task["returncode"], int)
            or task["returncode"] != 0
        ):
            raise ValueError("Expected a completed H800 LSTM collection")
        return (
            data,
            op["shapes"],
            task,
            dict(
                kind="external_nvidia",
                source_artifact=data.get("_snapshot_source"),
                artifact=str(path),
                artifact_sha256=hashlib.sha256(payload).hexdigest(),
                reference_implementation="pytorch_native_lstm_composition",
                input_identity="same distribution/configuration; not identical random payloads",
                ratio="raw H800 native-composition latency / target candidate latency",
                collection_task_status=task["status"],
            ),
        )

    def _import_latency(self, args, kwargs):
        data, shapes, task, provenance = self._h800
        config = self._input_config(args, kwargs)
        rows = [
            row
            for group in shapes.values()
            for dtype, row in group.items()
            if dtype == str(args[0].dtype)
            and json.dumps(row.get("config"), sort_keys=True, allow_nan=False)
            == json.dumps(config, sort_keys=True, allow_nan=False)
        ]
        if len(rows) != 1:
            raise ValueError(
                "Expected exactly one matching H800 LSTM input configuration"
            )
        row = rows[0]
        origin = row.get("collection_origin")
        environment = (
            origin["environment"]
            if origin
            else data["_environments"][str(task["device"])]
        )
        source = environment["source"]
        commit = source["git_commit"]
        validity = row.get("metric_validity")
        measured = (
            validity.get("T_us") == "measured"
            and validity.get("latency_ms") == "measured"
            if validity is not None
            else environment["run"].get("ncu_enabled") is False
            and environment["run"].get("selected_ops") == ["mkldnn_rnn_layer"]
        )
        if (
            environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["run"]["repo"] != "pytorch"
            or source["git_dirty"] is not False
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or not environment["software"]["packages"]["torch"]
            or not measured
        ):
            raise ValueError(
                "H800 latency needs measured timing and its actual environment"
            )
        for field in ("T_us", "latency_ms"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("H800 latency must be finite and positive")
        latency = row["T_us"] / 1000
        if not math.isclose(row["latency_ms"], latency, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("H800 latency units disagree")
        self._imported_baseline = dict(
            **provenance,
            environment=environment,
            config=config,
            collection_origin=origin,
            ncu_status=row.get("ncu_status"),
            ncu_reason=row.get("ncu_reason"),
            ncu_enabled=environment["run"]["ncu_enabled"],
            latency_validity="measured",
            target_timing_mode=base.Config.mode.value,
        )
        return latency

    def get_latency(self, op, *args, **kwargs):
        if getattr(self, "_native_measurement_pending", False):
            if op is not self.torch_op:
                raise ValueError("External baseline must replace the reference timer")
            self._native_measurement_pending = False
            return self._import_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external and (
            self.is_backward
            or "tflops" in self.to_bench_metrics
            or base.Config.skip_native
            or base.Config.reference_only
        ):
            raise ValueError(
                "External LSTM baseline requires ordinary forward latency timing"
            )
        self._imported_baseline = None
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if self._imported_baseline is not None:
                if (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError(
                        "Target candidate latency must be finite and positive"
                    )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _MkldnnMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False


class MkldnnRnnLayerBenchmark(_H800Latency, base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _MKLDNN_RNN_LAYER_SHAPES

    def set_more_shapes(self):
        return None


def mkldnn_rnn_layer_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def mkldnn_rnn_layer_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return mkldnn_rnn_layer_input_tuple(shape, dtype, device)


def mkldnn_rnn_layer_input_tuple(shape, dtype, device):
    seq_len, batch_size, input_size = shape
    hidden_size = input_size
    inp = torch.randn(seq_len, batch_size, input_size, dtype=dtype, device=device)
    lstm = torch.nn.LSTM(input_size, hidden_size, 1).to(dtype=dtype, device=device)
    w_ih = lstm.weight_ih_l0
    w_hh = lstm.weight_hh_l0
    b_ih = lstm.bias_ih_l0
    b_hh = lstm.bias_hh_l0
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    # All arguments are positional in the aten schema:
    # (input, w0, w1, w2, w3, hx, cx, reverse, batch_sizes, mode, hidden_size,
    #  num_layers, has_biases, bidirectional, batch_first, train)
    return (
        inp,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hx,
        cx,
        False,
        [],
        _MODE,
        hidden_size,
        _NUM_LAYERS,
        True,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
        False,
    )


def _torch_lstm_layer_baseline(inp, w_ih, w_hh, b_ih, b_hh, hx, cx, *rest):
    """Same-device LSTM baseline.

    The aten mkldnn_rnn_layer (oneDNN) is CPU-only, so it cannot serve as an
    on-device torch baseline. Reproduce the identical single-layer LSTM with
    native PyTorch on the input device instead (gate order i, f, g, o)."""
    reverse = rest[0]
    seq_len = inp.shape[0]
    h = hx
    c = cx
    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outputs = [None] * seq_len
    for t in steps:
        gates = torch.addmm(b_ih, inp[t], w_ih.t()) + torch.addmm(b_hh, h, w_hh.t())
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c = f_g * c + i_g * g_g
        h = o_g * torch.tanh(c)
        outputs[t] = h
    return torch.stack(outputs, dim=0), h, c


@pytest.mark.mkldnn_rnn_layer
def test_mkldnn_rnn_layer():
    bench = MkldnnRnnLayerBenchmark(
        case_fn=mkldnn_rnn_layer_case_fn,
        build_inputs_fn=mkldnn_rnn_layer_build_inputs_fn,
        op_name="mkldnn_rnn_layer",
        torch_op=_torch_lstm_layer_baseline,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
