# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete-call MetaX Linear comparisons.

Pass unmodified MetaX and generic Linear source files from
860538c69bfb6457d3c2eb858feca35c352a00cf via --native-source and
--generic-source. Their exact hashes are checked before benchmarking.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import flag_gems
import torch
import triton


def paired_calls(native, candidate, rounds=7):
    def capture(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [fn() for _ in range(8)]
        graph.replay()
        torch.cuda.synchronize()
        return graph, outputs

    ng, no = capture(native)
    cg, co = capture(candidate)

    def gpu(graph):
        samples = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 8)
        return statistics.median(samples)

    def eager(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            output = fn()
        torch.cuda.synchronize()
        _ = output  # Keep the last result alive through synchronization.
        return (time.perf_counter() - start) * 1e6 / 20

    samples = []
    for i in range(rounds):
        row = {}
        order = [("native", native, ng), ("candidate", candidate, cg)]
        if i % 2:
            order.reverse()
        for name, fn, graph in order:
            row[name + "_graph_us"] = gpu(graph)
            row[name + "_eager_us"] = eager(fn)
        samples.append(row)
    med = lambda key: statistics.median(row[key] for row in samples)
    result = {key: med(key) for key in samples[0]}
    result["graph_speedup"] = result["native_graph_us"] / result["candidate_graph_us"]
    result["eager_speedup"] = result["native_eager_us"] / result["candidate_eager_us"]
    result["graph_wins"] = sum(
        row["native_graph_us"] > row["candidate_graph_us"] for row in samples
    )
    result["rounds"] = samples
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--generic-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    expected_sha = "9838150a35e7f0ef23c7d255bf22e459aac7c1b8eceeecfb00c5b0e630a33efe"
    if hashlib.sha256(args.native_source.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Expected the pinned, unmodified MetaX Linear source")
    spec = importlib.util.spec_from_file_location(
        "linear_native_reference", args.native_source
    )
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    if (
        hashlib.sha256(args.generic_source.read_bytes()).hexdigest()
        != "564f52ffba1f50da4c47fe07c01b90d63164a326779078f8234571b4e7fbb0b7"
    ):
        raise ValueError("Expected the pinned generic Linear source")
    generic_spec = importlib.util.spec_from_file_location(
        "generic_linear_native_reference", args.generic_source
    )
    generic = importlib.util.module_from_spec(generic_spec)
    sys.modules[generic_spec.name] = generic
    generic_spec.loader.exec_module(generic)
    rows = []
    for m, n, k, qualified in [
        (1, 384, 6144, True),
        (1, 768, 6144, True),
        (8192, 1024, 6144, True),
        (2, 384, 6144, False),
        (1, 1024, 6144, False),
        (8192, 384, 6144, False),
    ]:
        torch.manual_seed(560)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=x.dtype)
        candidate = lambda: flag_gems.linear(x, w)
        expected = torch.nn.functional.linear(x.cpu().float(), w.cpu().float())
        for name, fn in [
            ("metax_native", lambda: reference.linear(x, w)),
            ("shared_generic", lambda: generic.linear(x, w)),
        ]:
            actual = candidate().cpu().float()
            native = fn().cpu().float()
            for value in (actual, native):
                delta = value - expected
                assert (delta.norm() / expected.norm().clamp_min(1e-20)) <= 0.004
                assert (
                    delta.abs().max() / expected.abs().max().clamp_min(1e-20)
                ) <= 0.01
            result = paired_calls(fn, candidate)
            row = dict(
                shape=[m, n, k],
                dtype="bfloat16",
                qualified=qualified,
                comparator=name,
                **result,
            )
            rows.append(row)
            print(
                json.dumps(
                    {key: value for key, value in row.items() if key != "rounds"}
                ),
                flush=True,
            )
    args.output.write_text(
        json.dumps(
            dict(
                rows=rows,
                torch=torch.__version__,
                triton=triton.__file__,
                source_reference_sha256=expected_sha,
                scope="Graph GPU time includes all operator kernels; eager also includes allocation/dispatch.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
