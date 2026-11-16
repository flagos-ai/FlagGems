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

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import triton
import yaml

import flag_gems

from . import base, consts


# TODO(Qiming): Extract this to a base class
class NormBenchmark(base.GenericBenchmark):
    # TODO: add new metric

    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
            # Medium normalized dimensions across representative row counts
            (256, 512),
            (4096, 256),
            (4096, 512),
            (2048, 1024),
            (1024, 2048),
        ]


def input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = shape[1:]
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    bias = torch.randn(layer_shape, dtype=dtype, device=device)
    yield inp, layer_shape, weight, bias


def native_layer_norm_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    normalized_shape = shape[1:]
    weight = torch.randn(normalized_shape, dtype=dtype, device=device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=device)
    yield inp, normalized_shape, weight, bias, 1e-5


@pytest.mark.native_layer_norm
def test_native_layer_norm():
    bench = NormBenchmark(
        op_name="native_layer_norm",
        input_fn=native_layer_norm_input_fn,
        torch_op=torch.ops.aten.native_layer_norm.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.layer_norm
def test_layer_norm():
    bench = NormBenchmark(
        op_name="layer_norm",
        input_fn=input_fn,
        torch_op=torch.layer_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def input_fn_backward(shape, dtype, device):
    grad_out = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    normalized_shape = shape[1:]
    # mean and rstd are always float32 for native_layer_norm_backward
    mean = torch.randn(shape[0], dtype=torch.float32, device=device)
    rstd = torch.randn(shape[0], dtype=torch.float32, device=device)
    weight = torch.randn(normalized_shape, dtype=dtype, device=device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=device)
    output_mask = [True, True, True]
    yield grad_out, inp, normalized_shape, mean, rstd, weight, bias, output_mask


@pytest.mark.layer_norm_backward
def test_layer_norm_backward():
    bench = NormBenchmark(
        op_name="layer_norm_backward",
        input_fn=input_fn_backward,
        torch_op=torch.ops.aten.native_layer_norm_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class LayernormBenchmark(base.Benchmark):
    def get_input_iter(self, dtype):
        torch.manual_seed(20261103)
        for shape, normalized_shape, eps in self.shapes:
            tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            weight = torch.randn(normalized_shape, dtype=dtype, device=tensor.device)
            bias = torch.randn_like(weight)
            yield tensor, normalized_shape, weight, bias, float(eps)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX C550")
@pytest.mark.layer_norm
def test_layernorm():
    bench = LayernormBenchmark(
        op_name="layernorm", torch_op=F.layer_norm, dtypes=consts.FLOAT_DTYPES
    )
    bench.set_gems(lambda *args, **kwargs: flag_gems.layer_norm(*args, **kwargs)[0])
    bench.run()


def capture_layernorm_graph(function, nodes):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            output = function()
    return graph, output


def measure_layernorm_graph(graph, nodes, replays):
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / nodes / replays


def make_manifest_tensor(metadata):
    dtypes = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    tensor = torch.empty_strided(
        metadata["shape"],
        metadata["stride"],
        dtype=dtypes[metadata["dtype"]],
        device=flag_gems.device,
    )
    return tensor.normal_()


def benchmark_manifest_case(record, nodes, replays, samples):
    extra = json.loads(record["extra"])
    tensor = make_manifest_tensor(
        {
            "shape": record["shape"],
            "stride": extra["input_stride"],
            "dtype": extra["input_dtype"],
        }
    )
    weight = make_manifest_tensor(extra["weight"]) if extra["weight"] else None
    bias = make_manifest_tensor(extra["bias"]) if extra["bias"] else None
    normalized_shape, eps = tuple(extra["normalized_shape"]), extra["eps"]
    native = lambda: F.layer_norm(tensor, normalized_shape, weight, bias, eps)
    candidate = lambda: flag_gems.layer_norm(
        tensor, normalized_shape, weight, bias, eps
    )[0]
    tolerance = {
        torch.float32: 1e-6,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }[tensor.dtype]
    expected, actual = native(), candidate()
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    native_graph, native_output = capture_layernorm_graph(native, nodes)
    candidate_graph, candidate_output = capture_layernorm_graph(candidate, nodes)
    native_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        candidate_output, native_output, atol=tolerance, rtol=tolerance
    )
    native_samples, candidate_samples = [], []
    for index in range(samples):
        ordered = [
            (native_graph, native_samples),
            (candidate_graph, candidate_samples),
        ]
        if index % 2:
            ordered.reverse()
        for graph, timings in ordered:
            timings.append(measure_layernorm_graph(graph, nodes, replays))
    native_us = statistics.median(native_samples)
    candidate_us = statistics.median(candidate_samples)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "normalized_shape": normalized_shape,
        "weight": extra["weight"],
        "bias": extra["bias"],
        "eps": eps,
        "count": record["count"],
        "sources": record["sources"],
        "native_us": native_us,
        "layernorm_us": candidate_us,
        "speedup": native_us / candidate_us,
        "samples_us": {"native": native_samples, "layernorm": candidate_samples},
        "max_abs": max_abs,
        "atol": tolerance,
        "rtol": tolerance,
        "eager_accuracy": "pass",
        "graph_accuracy": "pass",
    }


def core_layernorm_records(dtypes, shape_file):
    with shape_file.open() as stream:
        cases = yaml.safe_load(stream)["layernorm"]["shapes"]
    records = []
    for shape, normalized_shape, eps in cases:
        for dtype in dtypes:
            affine = {
                "shape": normalized_shape,
                "stride": [
                    math.prod(normalized_shape[index + 1 :])
                    for index in range(len(normalized_shape))
                ],
                "dtype": f"torch.{dtype}",
            }
            extra = {
                "normalized_shape": normalized_shape,
                "input_dtype": f"torch.{dtype}",
                "input_stride": [
                    math.prod(shape[index + 1 :]) for index in range(len(shape))
                ],
                "weight": affine,
                "bias": affine,
                "eps": float(eps),
            }
            records.append(
                {
                    "shape": shape,
                    "extra": json.dumps(extra),
                    "count": 0,
                    "sources": {"core_shapes": 0},
                }
            )
    return records


def run_manifest_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark captured LayerNorm shapes")
    suite = parser.add_mutually_exclusive_group(required=True)
    suite.add_argument("--shape-file", type=Path)
    suite.add_argument("--core-shapes", action="store_true")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["float16", "bfloat16", "float32"],
        help="Dtypes for --core-shapes; external manifest dtypes are preserved",
    )
    parser.add_argument(
        "--shapes-yaml", type=Path, default=Path(__file__).with_name("core_shapes.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if min(args.nodes, args.replays, args.samples) < 1:
        parser.error("nodes, replays and samples must be positive")
    records = (
        core_layernorm_records(args.dtypes, args.shapes_yaml)
        if args.core_shapes
        else json.loads(args.shape_file.read_text())
    )
    if not records:
        parser.error("shape manifest must not be empty")
    if flag_gems.vendor_name != "metax":
        raise NotImplementedError("This benchmark targets MetaX")
    torch.manual_seed(20261103)
    with torch.inference_mode():
        results = [
            benchmark_manifest_case(record, args.nodes, args.replays, args.samples)
            for record in records
        ]
    implementation = Path(__file__).resolve().parents[1] / (
        "src/flag_gems/runtime/backend/_metax/ops/layernorm.py"
    )
    report = {
        "operator": "layernorm",
        "baseline": "device-native torch.nn.functional.layer_norm",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "shape_file_sha256": (
            hashlib.sha256(args.shape_file.read_bytes()).hexdigest()
            if args.shape_file
            else None
        ),
        "suite": "core_shapes" if args.core_shapes else "external_manifest",
        "implementation_sha256": hashlib.sha256(
            implementation.read_bytes()
        ).hexdigest(),
        "seed": 20261103,
        "graph_nodes": args.nodes,
        "replays_per_sample": args.replays,
        "samples": args.samples,
        "unique_shapes": len(
            {(tuple(row["shape"]), tuple(row["normalized_shape"])) for row in results}
        ),
        "case_count": len(results),
        "captured_calls": sum(record["count"] for record in records),
        "arithmetic_mean_speedup": statistics.mean(row["speedup"] for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_manifest_benchmark()
