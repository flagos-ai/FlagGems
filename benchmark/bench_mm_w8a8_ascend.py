# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

new = importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm_w8a8_fp8")


def quant(x, dim):
    xf = x.float()
    scale = xf.abs().amax(dim=dim).clamp_min(1e-10) / 127.0
    view = scale[:, None] if dim == 1 else scale[None, :]
    return (xf / view).round().clamp(-128, 127).to(torch.int8), scale


def bench(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        g.capture_begin()
        for _ in range(10):
            fn()
        g.capture_end()
    torch.npu.current_stream().wait_stream(stream)
    for _ in range(5):
        g.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(3):
        st, en = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        st.record()
        for _ in range(20):
            g.replay()
        en.record()
        torch.npu.synchronize()
        times.append(st.elapsed_time(en) * 1000 / 200)
    return {"median_us": statistics.median(times), "samples_us": times}


def prepared(mod, a, b):
    aq, sa = mod._quantize_int8_rows(a)
    bq, sb = mod._quantize_int8_cols(b)
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    call, metadata = mod._prepare_mm_w8a8_kernel(
        aq, bq, sa, sb, out, a.shape[0], b.shape[1], a.shape[1]
    )
    return call, out, metadata


def summarize(samples):
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    shapes = [
        tuple(map(int, v.split("x")))
        for v in os.environ.get(
            "SHAPES",
            "1x16x16,2x32x32,8x64x64,16x128x64,32x128x128,64x256x128,"
            "128x256x256,192x512x512,256x768x1024,512x1024x1024,2048x2048x2048,"
            "16384x2048x2048,64x2048x4096,16384x12288x2048,1x248320x2048,"
            "8x248320x2048,16384x1024x2048,2048x2048x4096",
        ).split(",")
    ]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[
        os.environ.get("DTYPE", "bf16")
    ]
    results = {
        "candidate_sha256": hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape_order": "M,N,K",
        "dtype": str(dtype),
        "device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "current"),
        "method": (
            "Kernel only: matmul + both output scales + final cast. A/B quantization, padding, packing "
            "and explicit allocations excluded. NPUGraph 10 calls/capture, 5 warmup replays, 3x20 "
            "measured replays, two passes in opposite order; median of 6 samples. Torch uses the faster "
            "of row-major and column-major B (identical values; conversion excluded). Times in us."
        ),
        "stable_faster_rule": "max(kernel samples) < min(all samples of both Torch layouts), within this run only",
        "results": [],
    }
    for m, n, k in shapes:
        new.clear_mm_w8a8_fp8_caches()
        torch.manual_seed(123)
        a = torch.randn((m, k), device="npu", dtype=dtype)
        b = torch.randn((k, n), device="npu", dtype=dtype)
        fn, output, metadata = prepared(new, a, b)
        aq, sa = quant(a, 1)
        bq, sb = quant(b, 0)
        ref = ((aq.float() @ bq.float()) * sa[:, None] * sb[None, :]).to(dtype)
        fn()
        torch.npu.synchronize()
        torch.testing.assert_close(output, ref, rtol=0.016, atol=0.0512)
        torch_out = torch.empty_like(output)
        # Give Torch the same opportunity to choose a prepared weight layout.
        b_column = b.t().contiguous().t()

        def torch_row_call():
            torch.mm(a, b, out=torch_out)

        def torch_column_call():
            torch.mm(a, b_column, out=torch_out)

        functions = {
            "kernel": fn,
            "torch_row": torch_row_call,
            "torch_column": torch_column_call,
        }
        samples = {key: [] for key in functions}
        for order in [
            ("kernel", "torch_row", "torch_column"),
            ("torch_column", "torch_row", "kernel"),
        ]:
            for key in order:
                samples[key].extend(bench(functions[key])["samples_us"])
        best = min(
            ("torch_row", "torch_column"),
            key=lambda key: statistics.median(samples[key]),
        )
        row = {
            "shape": [m, n, k],
            **metadata,
            "max_abs_quant": (output.float() - ref.float()).abs().max().item(),
            **{key: summarize(values) for key, values in samples.items()},
            "torch_best_layout": best,
        }
        row["speedup_vs_torch"] = row[best]["median_us"] / row["kernel"]["median_us"]
        row["stable_faster"] = max(samples["kernel"]) < min(
            min(samples["torch_row"]), min(samples["torch_column"])
        )
        if os.environ.get("GEMS") == "1":
            gems = importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm")
            row["gems"] = bench(lambda: gems.mm_out(a, b, out=torch_out))
        results["results"].append(row)
        Path(sys.argv[1]).write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
