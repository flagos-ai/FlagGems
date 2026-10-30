import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts, utils


def _fractional_backward_forward_fixture(
    inp, kernel_size, output_size, random_samples=None
):
    if flag_gems.vendor_name != "ascend":
        return flag_gems.fractional_max_pool2d(
            inp,
            kernel_size=kernel_size,
            output_size=output_size,
            _random_samples=random_samples,
        )
    # Build canonical flat indices from the actual input and original samples.
    # Ascend's native frontend also uses a CPU fallback for this forward API.
    if random_samples is None:
        random_samples = torch.rand(
            inp.shape[0], inp.shape[1], 2, dtype=torch.float32, device=inp.device
        )
    output, indices = torch.nn.functional.fractional_max_pool2d(
        inp.detach().to(device="cpu", dtype=torch.float64),
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=random_samples.to(device="cpu", dtype=torch.float64),
    )
    return output.to(device=inp.device, dtype=inp.dtype), indices.to(device=inp.device)


class FractionalMaxPool2dBenchmark(base.GenericBenchmark):
    pass


@dataclass
class _FractionalBackwardMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class _FractionalBackwardH800Mixin:
    @cached_property
    def _h800_records(self):
        path = Path(
            os.environ["FLAGGEMS_FRACTIONAL_POOL2D_BACKWARD_H800_BASELINE"]
        ).resolve(strict=True)
        raw = path.read_bytes()
        digest = "e39cfbee78e702c43fee2e8361abd95e7c8d0895e02ec30c0824b1f25c5ea91b"
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(
                "Expected the exact native H800 fractional backward artifact"
            )
        data = json.loads(raw)
        op = "fractional_max_pool2d_backward"
        task = data["_collection"]["tasks"][op]
        environment = data["_environments"][str(task["device"])]
        records = data[op]["shapes"]
        shapes = {
            "shared": (
                (4, 3, 224, 224),
                (16, 64, 56, 56),
                (32, 128, 28, 28),
                (64, 256, 14, 14),
                (128, 512, 7, 7),
            ),
            "split": ((4, 3, 32, 32), (8, 16, 28, 28), (2, 8, 16, 20)),
        }
        expected = {
            name + "_" + "x".join(map(str, shape))
            for name, items in shapes.items()
            for shape in items
        }
        if (
            task["status"] != "completed"
            or task["returncode"] != 0
            or data["_collection"]["configuration"]["repo"] != "pytorch"
            or data[op]["native_api"] != "ops.fractional_max_pool2d_backward"
            or environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
            or environment["source"]["git_commit"]
            != "f83d67187e11d6f09dbaeedc8a9763e4b1032981"
            or environment["source"]["git_dirty"] is not False
            or set(records) != expected
            or any(
                set(values) != {"torch.float16", "torch.float32", "torch.bfloat16"}
                for values in records.values()
            )
        ):
            raise ValueError("Expected all 24 original shared/split native H800 cases")
        return records, dict(
            artifact=str(path),
            artifact_sha256=digest,
            collection_task=task,
            environment=environment,
            producer_source="f83d67187e11d6f09dbaeedc8a9763e4b1032981",
            input_identity="Original input/sample/gradient recipe, shared caller RNG and split seed0; no cross-device random-byte identity",
            ratio="Raw native H800 latency / locally timed candidate; no scaling",
        )

    def get_latency(self, op, *args, **kwargs):
        if not getattr(self, "_h800_pending", False):
            return super().get_latency(op, *args, **kwargs)
        self._h800_pending = False
        if (
            op is not self.torch_op
            or self.is_backward
            or len(args) != 2
            or set(kwargs) != {"kernel_size", "output_size", "indices"}
        ):
            raise ValueError(
                "Static baseline requires the original standalone backward call"
            )
        grad_output, inp = args
        indices = kwargs["indices"]
        namespace = self._h800_namespace
        shapes = {
            "shared": (
                (4, 3, 224, 224),
                (16, 64, 56, 56),
                (32, 128, 28, 28),
                (64, 256, 14, 14),
                (128, 512, 7, 7),
            ),
            "split": ((4, 3, 32, 32), (8, 16, 28, 28), (2, 8, 16, 20)),
        }
        size = tuple(inp.shape[-2:])
        output_size = (size[0] // 2, size[1] // 2)
        if (
            tuple(inp.shape) not in shapes[namespace]
            or kwargs["kernel_size"] != (2 if namespace == "shared" else [2, 2])
            or tuple(kwargs["output_size"]) != output_size
            or inp.dtype not in consts.FLOAT_DTYPES
            or inp.device != torch.empty(0, device=self.device).device
            or inp.requires_grad != (namespace == "shared")
            or grad_output.requires_grad
            or indices.requires_grad
            or grad_output.device != inp.device
            or indices.device != inp.device
            or grad_output.dtype != inp.dtype
            or indices.dtype != torch.int64
            or tuple(grad_output.shape) != (*inp.shape[:-2], *output_size)
            or indices.shape != grad_output.shape
            or any(
                not value.is_contiguous() or value.storage_offset() != 0
                for value in (grad_output, inp, indices)
            )
        ):
            raise ValueError(
                "Runtime inputs differ from the original H800 backward recipe"
            )
        records, provenance = self._h800_records
        key = namespace + "_" + "x".join(map(str, inp.shape))
        record = records[key][str(inp.dtype)]
        us = record["T_us"]
        if (
            isinstance(us, bool)
            or not isinstance(us, (int, float))
            or not math.isfinite(us)
            or us <= 0
        ):
            raise ValueError("Native H800 latency must be finite and positive")
        ms = us / 1000
        if not math.isclose(record["latency_ms"], ms, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Native H800 latency units disagree")
        self._h800_source = dict(
            provenance,
            key=key,
            dtype=str(inp.dtype),
            configuration=dict(
                namespace=namespace,
                shape=list(inp.shape),
                kernel_size=kwargs["kernel_size"],
                output_size=list(output_size),
            ),
        )
        return ms

    def _measure_input(self, input, case_id=None):
        needs_native = (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if flag_gems.vendor_name != "ascend" or not needs_native:
            return super()._measure_input(input, case_id)
        if not os.environ.get("FLAGGEMS_FRACTIONAL_POOL2D_BACKWARD_H800_BASELINE"):
            raise ValueError(
                "Ascend native fractional backward uses CPU fallback; set FLAGGEMS_FRACTIONAL_POOL2D_BACKWARD_H800_BASELINE"
            )
        if self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "Static baseline requires standalone backward latency measurement"
            )
        self._h800_pending = True
        self._h800_source = None
        try:
            metric = super()._measure_input(input, case_id)
            if self._h800_source is not None:
                if metric.latency is not None and (
                    isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Candidate must have a real positive latency")
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _FractionalBackwardMetrics(
                    **asdict(metric), baseline_source=self._h800_source
                )
            return metric
        finally:
            self._h800_pending = False


class _FractionalSharedBackwardBenchmark(
    _FractionalBackwardH800Mixin, FractionalMaxPool2dBenchmark
):
    _h800_namespace = "shared"


def fractional_max_pool2d_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"kernel_size": 2, "output_size": (shape[2] // 2, shape[3] // 2)},
        builder_args=(shape, 2),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        if shape[-2] > 5 and shape[-1] > 5:
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={
                    "kernel_size": 2,
                    "output_size": (shape[2] // 4, shape[3] // 4),
                },
                builder_args=(shape, 4),
            )


def fractional_max_pool2d_build_inputs_fn(plan, dtype, device):
    shape, divisor = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {
        "kernel_size": 2,
        "output_size": (shape[2] // divisor, shape[3] // divisor),
    }


@pytest.mark.fractional_max_pool2d
def test_fractional_max_pool2d():
    bench = FractionalMaxPool2dBenchmark(
        case_fn=fractional_max_pool2d_case_fn,
        build_inputs_fn=fractional_max_pool2d_build_inputs_fn,
        op_name="fractional_max_pool2d",
        torch_op=torch.nn.functional.fractional_max_pool2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.fractional_max_pool2d)
    bench.run()


def fractional_max_pool2d_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"kernel_size": 2, "output_size": (shape[2] // 2, shape[3] // 2)},
        builder_args=(shape,),
    )


def fractional_max_pool2d_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    inp.requires_grad_(True)
    output_size = (shape[2] // 2, shape[3] // 2)
    output, indices = _fractional_backward_forward_fixture(
        inp, kernel_size=2, output_size=output_size
    )
    grad_output = torch.randn_like(output)
    return (
        grad_output,
        inp,
        {
            "kernel_size": 2,
            "output_size": output_size,
            "indices": indices,
        },
    )


def torch_fractional_max_pool2d_backward_wrapper(grad_output, input, **kwargs):
    return torch.ops.aten.fractional_max_pool2d_backward(
        grad_output,
        input,
        kwargs["kernel_size"],
        kwargs["output_size"],
        kwargs["indices"],
    )


@pytest.mark.fractional_max_pool2d_backward
def test_fractional_max_pool2d_backward():
    bench = _FractionalSharedBackwardBenchmark(
        case_fn=fractional_max_pool2d_backward_case_fn,
        build_inputs_fn=fractional_max_pool2d_backward_build_inputs_fn,
        op_name="fractional_max_pool2d_backward",
        torch_op=torch_fractional_max_pool2d_backward_wrapper,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.fractional_max_pool2d_backward)
    bench.run()
