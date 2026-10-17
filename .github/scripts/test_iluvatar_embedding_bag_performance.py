# Copyright 2026 FlagOS Contributors
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

"""Isolated valid-input performance diagnostics; never a production bypass."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

CHILD = "IX_PYTEST_ERROR_CHILD"


def _isolated(request):
    if os.environ.get(CHILD) == "1":
        return False
    # A fresh process keeps one failed device assertion from poisoning later probes.
    # Keep each parametrized case isolated, including the successful controls.
    node = str(Path(__file__).resolve()) + "::" + request.node.name
    env = dict(os.environ, **{CHILD: "1", "TRITON_DEBUG": "0"})
    command = [sys.executable, "-m", "pytest", "-c", "/dev/null", "-s", "-vv", node]
    with subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            pytest.fail("Child pytest timed out:\n" + output)
    print(output, flush=True)
    assert process.returncode == 0, f"Child pytest exit={process.returncode}\n{output}"
    if "1 skipped" in output:
        pytest.skip("Isolated device capability skip; see child stdout")
    assert "1 passed" in output, output
    return True


SHAPES = [(8, 16, 128, 4), (1024, 64, 16384, 64)]


@pytest.mark.parametrize("variant", ["_embedding_bag", "_embedding_bag_forward_only"])
@pytest.mark.parametrize("dtype_name", ["float16", "float32", "bfloat16"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("shape", SHAPES, ids=["small", "large"])
def test_forward_performance(request, variant, dtype_name, mode, shape):
    if _isolated(request):
        return
    import cProfile
    import importlib
    import inspect
    import io
    import json
    import pstats
    import statistics
    import time
    import warnings

    import torch
    import triton
    import triton.language as tl

    import flag_gems

    if dtype_name == "bfloat16" and not flag_gems.runtime.device.support_bf16:
        pytest.skip("Device does not support bfloat16")
    backend = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
    )
    path = Path("../ix-results/performance-probes") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    bags, dim, rows, length = shape
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(1234)
    weight = torch.randn((rows, dim), dtype=dtype, device="cuda")
    indices = torch.randint(1, rows, (bags * length,), device="cuda")
    offsets = torch.arange(0, bags * length, length, device="cuda")
    args = (weight, indices, offsets, False, mode, False, None, False, 0)
    native = getattr(torch.ops.aten, variant).default
    gems = getattr(backend, variant)
    report = {
        "variant": variant,
        "dtype": dtype_name,
        "mode": mode,
        "shape": shape,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "measurements": {},
    }

    def save():
        (path / "results.json").write_text(json.dumps(report, indent=2))

    (path / "triton-timers.py.txt").write_text(
        inspect.getsource(triton.testing.do_bench)
        + "\n"
        + inspect.getsource(triton.testing.do_bench_cudagraph)
    )
    # Measure first call separately; it includes lazy setup/compilation and is
    # never mixed into steady-state comparisons.
    for name, function in (("native", native), ("candidate", gems)):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            function(*args)
            torch.cuda.synchronize()
        report[name + "_first_call_ms"] = (time.perf_counter() - begin) * 1000
        report[name + "_warnings"] = [str(w.message) for w in caught]
        if any(
            "fallback" in str(w.message).lower().replace("fall back", "fallback")
            and "cpu" in str(w.message).lower()
            for w in caught
        ):
            report["status"] = "CPU_FALLBACK_UNMEASURED"
            save()
            pytest.skip("Native/device CPU fallback warning; saved warning evidence")
    report["runtime_flags"] = {
        key: os.environ.get(key)
        for key in (
            "CUDA_LAUNCH_BLOCKING",
            "TRITON_DEBUG",
            "TRITON_INTERPRET",
            "TORCH_USE_CUDA_DSA",
        )
    }
    report["device"] = torch.cuda.get_device_name()
    for name, obj in (
        ("jit-run", triton.runtime.JITFunction.run),
        ("forward-impl", backend._generic_impl),
    ):
        (path / f"{name}-source.py.txt").write_text(inspect.getsource(obj))
    save()

    @triton.jit
    def empty_launch(buffer):
        value = tl.load(buffer)
        tl.store(buffer, value)

    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(bags, dtype=torch.int32, device="cuda")
    functions = {
        "native": lambda: native(*args),
        "candidate": lambda: gems(*args),
        "checker_only": lambda: backend._check_native_error(flags, bags),
        "empty_debug_false": lambda: empty_launch[(1,)](zero, debug=False),
        "empty_debug_true": lambda: empty_launch[(1,)](zero, debug=True),
    }

    def measure(function):
        for _ in range(5):
            function()
        torch.cuda.synchronize()
        values = {"triton_ms": [], "batch_event_ms": [], "wall_ms": [], "submit_ms": []}
        for _ in range(3):
            values["triton_ms"].append(
                triton.testing.do_bench(
                    function, warmup=20, rep=60, return_mode="median"
                )
            )
            count = 50
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start.record()
            enqueue = time.perf_counter()
            for _ in range(count):
                function()
            submitted = time.perf_counter()
            end.record()
            end.synchronize()
            finish = time.perf_counter()
            values["batch_event_ms"].append(start.elapsed_time(end) / count)
            values["wall_ms"].append((finish - begin) * 1000 / count)
            values["submit_ms"].append((submitted - enqueue) * 1000 / count)
        return {
            "raw": values,
            "median": {key: statistics.median(v) for key, v in values.items()},
        }

    for name, function in functions.items():
        report["measurements"][name] = measure(function)
        save()
    # Diagnostic ablations use only in-range, prevalidated inputs. Neither is
    # installed into operator-source or used as an accepted implementation.
    for name, attribute in (
        ("without_check", "_check_native_error"),
        ("without_prepare", "_prepare_native_assert"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(backend, attribute, lambda *args: None)
            report["measurements"][name] = measure(functions["candidate"])
        save()

    # Audit Python-visible synchronizations/readbacks only during the op call;
    # timer synchronization is outside this region. Profiler traces below cover
    # lower-level runtime activity that these wrappers cannot observe.
    calls = []

    def audit(name, original):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return wrapped

    with pytest.MonkeyPatch.context() as patch:
        for name in ("item", "tolist", "cpu", "numpy"):
            patch.setattr(
                torch.Tensor, name, audit("Tensor." + name, getattr(torch.Tensor, name))
            )
        patch.setattr(
            torch.cuda, "synchronize", audit("cuda.synchronize", torch.cuda.synchronize)
        )
        patch.setattr(
            torch.cuda.Event,
            "synchronize",
            audit("Event.synchronize", torch.cuda.Event.synchronize),
        )
        patch.setattr(
            torch.cuda.Stream,
            "synchronize",
            audit("Stream.synchronize", torch.cuda.Stream.synchronize),
        )
        functions["candidate"]()
    report["candidate_python_sync_readback_calls"] = calls
    torch.cuda.synchronize()
    save()

    for name in ("native", "candidate", "checker_only"):
        function = functions[name]
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(30):
            function()
        profile.disable()
        torch.cuda.synchronize()
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(60)
        (path / f"{name}-python-profile.txt").write_text(stream.getvalue())
        # Optional profiler capability is reported explicitly, never confused
        # with a successful device trace when CoreX lacks Kineto support.
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(5):
                    function()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(path / f"{name}-trace.json"))
            (path / f"{name}-profiler.txt").write_text(
                prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60)
            )
        except Exception as error:
            report[name + "_profiler_error"] = repr(error)
            save()
            raise

    # Graph replay separates device work from repeated Python dispatch. Run it
    # last: a failed capture can invalidate the device context on some runtimes.
    for name in (
        "native",
        "candidate",
        "checker_only",
        "empty_debug_false",
        "empty_debug_true",
    ):
        try:
            report["measurements"][name]["graph_ms"] = (
                triton.testing.do_bench_cudagraph(
                    functions[name], rep=60, return_mode="median"
                )
            )
        except Exception as error:
            report["graph_error"] = {"stage": name, "error": repr(error)}
            save()
            break
        save()
    print(json.dumps(report, indent=2), flush=True)


@pytest.mark.parametrize(
    "kind",
    ["loop_store", "loop_assert", "scalar_assert", "guarded_assert", "tiled_assert"],
)
@pytest.mark.parametrize("count", [1, 513])
@pytest.mark.parametrize("phase", ["timing", "invalid"])
def test_assertion_components(request, kind, count, phase):
    """Separate reduction work from assertion runtime machinery in fresh processes."""
    if phase == "invalid" and kind == "loop_store":
        pytest.skip("Reduction-only control deliberately has no assertion")
    if _isolated(request):
        return
    import importlib
    import json
    import statistics
    import time

    import torch
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["count"])
    def check(flags, result, count, kind: tl.constexpr, block: tl.constexpr):
        lane = tl.arange(0, block)
        if kind == "scalar_assert":
            # One program per flag removes both the loop and the reduction.
            index = tl.program_id(0)
            bad = tl.load(flags + index)
            tl.device_assert(bad == 0, "IX component probe: invalid flag")
        elif kind == "tiled_assert":
            index = tl.program_id(0) * block + lane
            bad = tl.load(flags + index, index < count, other=0)
            # Elementwise assertion checks all flags without scalar reduction.
            tl.device_assert(bad == 0, "IX component probe: invalid flag")
        else:
            bad = tl.full((), 0, tl.int32)
            for base in range(0, count, block):
                values = tl.load(flags + base + lane, base + lane < count, other=0)
                bad |= tl.max(values, 0)
            if kind == "loop_store":
                tl.store(result, bad)
            elif kind == "guarded_assert":
                # Test whether a device-side cold error branch avoids overhead.
                if bad != 0:
                    tl.device_assert(bad == 0, "IX component probe: invalid flag")
            else:
                tl.device_assert(bad == 0, "IX component probe: invalid flag")

    path = Path("../ix-results/assertion-components") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    flags = torch.zeros(count, dtype=torch.int32, device="cuda")
    result = torch.empty(1, dtype=torch.int32, device="cuda")
    grid = (
        count
        if kind == "scalar_assert"
        else triton.cdiv(count, 256) if kind == "tiled_assert" else 1
    )

    def launch():
        return check[(grid,)](flags, result, count, kind, 256, debug=True)

    # Warm the exact assertion specialization before any embedding-bag kernels.
    # No host wait or readback is inserted into this candidate sequence.
    compiled = launch()
    for name, code in compiled.asm.items():
        if isinstance(code, str):
            (path / (name + ".txt")).write_text(code)
    report = {
        "kind": kind,
        "count": count,
        "phase": phase,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "kernel_hash": compiled.hash,
    }

    def save():
        (path / "results.json").write_text(json.dumps(report, indent=2))

    save()
    if phase == "invalid":
        backend = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
        )
        backward = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag_backward"
        )
        weight = torch.ones((16, 8), device="cuda")
        indices = torch.tensor([1, 2, 3, 4], device="cuda")
        offsets = torch.tensor([0, 2], device="cuda")
        output, mapping, sizes, maxima = backend._embedding_bag(
            weight, indices, offsets
        )
        backward._embedding_bag_backward(
            torch.ones_like(output),
            indices,
            offsets,
            mapping,
            sizes,
            maxima,
            16,
            False,
            0,
            False,
        )
        # Preserve the asynchronous forward/backward-to-check sequence.
        flags[-1:].fill_(1)
        try:
            launch()
            torch.cuda.synchronize()
        except Exception as error:
            report["exception_type"] = type(error).__name__
            report["exception"] = str(error)
            report["caught"] = "assert" in str(error).lower()
        else:
            report["caught"] = False
        save()
        assert report["caught"], report
        return

    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    report["triton_ms"] = [
        triton.testing.do_bench(launch, warmup=20, rep=60, return_mode="median")
        for _ in range(3)
    ]
    report["median_ms"] = statistics.median(report["triton_ms"])
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    begin = time.perf_counter()
    for _ in range(50):
        launch()
    report["submit_ms"] = (time.perf_counter() - begin) * 1000 / 50
    end.record()
    end.synchronize()
    report["batch_event_ms"] = start.elapsed_time(end) / 50
    save()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path / "trace.json"))
    (path / "profiler.txt").write_text(prof.key_averages().table(row_limit=30))


@pytest.mark.parametrize("kind", ["printf_cold", "printf_hot"])
@pytest.mark.parametrize("count", [1, 513])
@pytest.mark.parametrize("phase", ["timing"])
def test_printf_isolation(request, kind, count, phase):
    """Separate reduction work from assertion runtime machinery in fresh processes."""
    if phase == "invalid" and kind not in ("assert_unsigned", "assert_empty_message"):
        pytest.skip("This control is not an error-reporting candidate")
    if _isolated(request):
        return
    import importlib
    import inspect
    import json
    import statistics
    import time

    import torch
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["count"])
    def reduce_only(flags, result, count, block: tl.constexpr):
        lane = tl.arange(0, block)
        bad = tl.full((), 0, tl.int32)
        for base in range(0, count, block):
            values = tl.load(flags + base + lane, base + lane < count, other=0)
            bad |= tl.max(values, 0)
        tl.store(result, bad)

    @triton.jit(do_not_specialize=["count"])
    def instrumented(flags, result, count, kind: tl.constexpr, block: tl.constexpr):
        # Unsigned bounded indices avoid debug's signed int32 overflow checks.
        lane = tl.arange(0, block).to(tl.uint32)
        limit = count.to(tl.uint32)
        bad = tl.full((), 0, tl.int32)
        for base in range(0, count, block):
            index = base.to(tl.uint32) + lane
            values = tl.load(flags + index, index < limit, other=0)
            bad |= tl.max(values, 0)
        if kind == "printf_cold":
            if bad != 0:
                tl.device_print("IX printf cold", bad)
        elif kind == "printf_hot":
            tl.device_print("IX printf hot", bad)
        elif kind == "assert_unsigned":
            tl.device_assert(bad == 0, "IX invalid flag")
        elif kind == "assert_empty_message":
            tl.device_assert(bad == 0, "")
        tl.store(result, bad)

    path = Path("../ix-results/printf-isolation") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    import triton.backends.iluvatar.driver as ix_driver

    (path / "driver-source.py.txt").write_text(inspect.getsource(ix_driver))
    (path / "jit-source.py.txt").write_text(
        inspect.getsource(triton.runtime.JITFunction.run)
    )
    flags = torch.zeros(count, dtype=torch.int32, device="cuda")
    result = torch.empty(1, dtype=torch.int32, device="cuda")

    def launch():
        if kind == "reduce_debug_false":
            return reduce_only[(1,)](flags, result, count, 256, debug=False)
        # Print-only controls must not inherit debug overflow assertions.
        return instrumented[(1,)](flags, result, count, kind, 256, debug=False)

    # Warm the exact assertion specialization before any embedding-bag kernels.
    # No host wait or readback is inserted into this candidate sequence.
    compiled = launch()
    for name, code in compiled.asm.items():
        if isinstance(code, str):
            (path / (name + ".txt")).write_text(code)
    report = {
        "kind": kind,
        "count": count,
        "phase": phase,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "kernel_hash": compiled.hash,
    }

    def save():
        (path / "results.json").write_text(json.dumps(report, indent=2))

    ir = compiled.asm.get("ttir", "")
    assert ir, "Compiler did not expose TTIR for control verification"
    report["ttir_assert_lines"] = [
        line for line in ir.splitlines() if "tt.assert" in line
    ]
    report["asm_instrumentation"] = [
        line
        for line in compiled.asm.get("asm", "").splitlines()
        if "has_called_assert" in line or "has_called_printf" in line
    ]
    save()
    if kind in ("reduce_debug_false", "reduce_unsigned", "printf_cold", "printf_hot"):
        assert not report[
            "ttir_assert_lines"
        ], "Control contaminated by compiler assertions"
    if kind in ("reduce_debug_false", "reduce_unsigned"):
        assert "tt.print" not in ir, "Reduction control contains device printf"
    if kind in ("printf_cold", "printf_hot"):
        assert "tt.print" in ir, "Printf control was compiled away"
        metadata = "\n".join(report["asm_instrumentation"])
        assert "has_called_assert = 0" in metadata, metadata
        assert "has_called_printf = 1" in metadata, metadata
    if kind.startswith("assert_"):
        assert report["ttir_assert_lines"], "Required assertion was compiled away"
    if phase == "invalid":
        backend = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
        )
        backward = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag_backward"
        )
        weight = torch.ones((16, 8), device="cuda")
        indices = torch.tensor([1, 2, 3, 4], device="cuda")
        offsets = torch.tensor([0, 2], device="cuda")
        output, mapping, sizes, maxima = backend._embedding_bag(
            weight, indices, offsets
        )
        backward._embedding_bag_backward(
            torch.ones_like(output),
            indices,
            offsets,
            mapping,
            sizes,
            maxima,
            16,
            False,
            0,
            False,
        )
        # Preserve the asynchronous forward/backward-to-check sequence.
        flags[-1:].fill_(1)
        try:
            launch()
            torch.cuda.synchronize()
        except Exception as error:
            report["exception_type"] = type(error).__name__
            report["exception"] = str(error)
            report["caught"] = "assert" in str(error).lower()
        else:
            report["caught"] = False
        save()
        assert report["caught"], report
        return

    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    if kind == "printf_hot":
        # Bound stdout: ten launches per sample rather than adaptive thousands.
        samples = []
        for _ in range(3):
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            begin_event.record()
            for _ in range(10):
                launch()
            end_event.record()
            end_event.synchronize()
            samples.append(begin_event.elapsed_time(end_event) / 10)
        report["timing_method"] = "10-launch event batches"
        report["triton_ms"] = samples
    else:
        report["timing_method"] = "triton do_bench"
        report["triton_ms"] = [
            triton.testing.do_bench(launch, warmup=20, rep=60, return_mode="median")
            for _ in range(3)
        ]
    report["median_ms"] = statistics.median(report["triton_ms"])
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    begin = time.perf_counter()
    for _ in range(50):
        launch()
    report["submit_ms"] = (time.perf_counter() - begin) * 1000 / 50
    end.record()
    end.synchronize()
    report["batch_event_ms"] = start.elapsed_time(end) / 50
    save()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path / "trace.json"))
    (path / "profiler.txt").write_text(prof.key_averages().table(row_limit=30))


@pytest.mark.parametrize("count", [1, 513])
@pytest.mark.parametrize("phase", ["timing", "invalid"])
def test_sdk_direct(request, count, phase):
    """Separate reduction work from assertion runtime machinery in fresh processes."""
    if _isolated(request):
        return
    import importlib
    import json
    import statistics
    import time

    import torch
    import triton
    import triton.language as tl

    backend = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
    )
    library = backend._corex_assert_library()

    @tl.core.builtin
    def sdk_assert(code, _semantic=None):
        # Build a real Python list outside JIT AST lowering. CoreX Triton 3.6
        # extern_elementwise calls args.copy(), which a TL tuple cannot provide.
        return tl.core.extern_elementwise(
            "",
            "",
            [code],
            {(tl.int32,): ("flaggems_corex_assert", tl.int32)},
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit(do_not_specialize=["count"])
    def check(flags, count, block: tl.constexpr):
        lane = tl.arange(0, block)
        bad = tl.full((), 0, tl.int32)
        for base in range(0, count, block):
            values = tl.load(flags + base + lane, base + lane < count, other=0)
            bad |= tl.max(values, 0)
        sdk_assert(bad)

    path = Path("../ix-results/sdk-direct") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    flags = torch.zeros(count, dtype=torch.int32, device="cuda")
    (path / "sdk-adapter.ll").write_text(Path(library).read_text())

    def launch():
        return check[(1,)](
            flags,
            count,
            256,
            debug=False,
            extern_libs={"embedding_bag_assert": library},
        )

    # Warm the exact assertion specialization before any embedding-bag kernels.
    # No host wait or readback is inserted into this candidate sequence.
    compiled = launch()
    for name, code in compiled.asm.items():
        if isinstance(code, str):
            (path / (name + ".txt")).write_text(code)
    report = {
        "kind": "sdk_direct",
        "count": count,
        "phase": phase,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "kernel_hash": compiled.hash,
    }

    def save():
        (path / "results.json").write_text(json.dumps(report, indent=2))

    report["asm_instrumentation"] = [
        line
        for line in compiled.asm.get("asm", "").splitlines()
        if "has_called_assert" in line or "has_called_printf" in line
    ]
    save()
    if phase == "invalid":
        backend = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
        )
        backward = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag_backward"
        )
        weight = torch.ones((16, 8), device="cuda")
        indices = torch.tensor([1, 2, 3, 4], device="cuda")
        offsets = torch.tensor([0, 2], device="cuda")
        output, mapping, sizes, maxima = backend._embedding_bag(
            weight, indices, offsets
        )
        backward._embedding_bag_backward(
            torch.ones_like(output),
            indices,
            offsets,
            mapping,
            sizes,
            maxima,
            16,
            False,
            0,
            False,
        )
        # Preserve the asynchronous forward/backward-to-check sequence.
        flags[-1:].fill_(1)
        try:
            launch()
            torch.cuda.synchronize()
        except Exception as error:
            report["exception_type"] = type(error).__name__
            report["exception"] = str(error)
            report["caught"] = "assert" in str(error).lower()
        else:
            report["caught"] = False
        save()
        assert report["caught"], report
        return

    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    report["triton_ms"] = [
        triton.testing.do_bench(launch, warmup=20, rep=60, return_mode="median")
        for _ in range(3)
    ]
    report["median_ms"] = statistics.median(report["triton_ms"])
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    begin = time.perf_counter()
    for _ in range(50):
        launch()
    report["submit_ms"] = (time.perf_counter() - begin) * 1000 / 50
    end.record()
    end.synchronize()
    report["batch_event_ms"] = start.elapsed_time(end) / 50
    save()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path / "trace.json"))
    (path / "profiler.txt").write_text(prof.key_averages().table(row_limit=30))


@pytest.mark.parametrize("variant", ["_embedding_bag", "_embedding_bag_forward_only"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("invalid", ["index", "offset"])
def test_native_error_path(request, variant, mode, invalid):
    """Measure native valid launches before observing one isolated invalid call."""
    if _isolated(request):
        return
    import json
    import statistics
    import time
    import warnings

    import torch
    import triton

    path = Path("../ix-results/native-error-path") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    rows, dim, bags, length = 128, 16, 8, 4
    weight = torch.randn((rows, dim), device="cuda", dtype=torch.float32)
    indices = torch.randint(1, rows, (bags * length,), device="cuda")
    offsets = torch.arange(0, bags * length, length, device="cuda")
    native = getattr(torch.ops.aten, variant).default

    def call(index, offset):
        return native(weight, index, offset, False, mode, False, None, False, 0)

    report = {
        "variant": variant,
        "mode": mode,
        "invalid": invalid,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "shape": [rows, dim, bags, length],
        "dtype": "float32",
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }

    def save():
        (path / "results.json").write_text(json.dumps(report, indent=2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            call(indices, offsets)
        torch.cuda.synchronize()
    report["warnings"] = [str(w.message) for w in caught]
    save()
    if any(
        "fallback" in msg.lower().replace("fall back", "fallback")
        and "cpu" in msg.lower()
        for msg in report["warnings"]
    ):
        pytest.skip("Native CPU fallback: performance comparison is not applicable")
    report["valid_triton_ms"] = [
        triton.testing.do_bench(
            lambda: call(indices, offsets), warmup=20, rep=60, return_mode="median"
        )
        for _ in range(3)
    ]
    report["valid_median_ms"] = statistics.median(report["valid_triton_ms"])
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    begin = time.perf_counter()
    for _ in range(50):
        call(indices, offsets)
    report["valid_submit_ms"] = (time.perf_counter() - begin) * 1000 / 50
    end.record()
    end.synchronize()
    report["valid_batch_event_ms"] = start.elapsed_time(end) / 50
    save()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(5):
            call(indices, offsets)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path / "valid-trace.json"))
    (path / "valid-profiler.txt").write_text(prof.key_averages().table(row_limit=40))

    bad_indices, bad_offsets = indices.clone(), offsets.clone()
    if invalid == "index":
        bad_indices[-1:].fill_(rows)
    else:
        bad_offsets[-1:].fill_(indices.numel() + 1)
    # Establish a clean test boundary before timing the invalid operator itself.
    # This synchronization belongs only to the diagnostic harness.
    torch.cuda.synchronize()
    report["error_stage"] = None
    stage = "operator_call"
    begin = time.perf_counter()
    try:
        call(bad_indices, bad_offsets)
        report["invalid_call_ms"] = (time.perf_counter() - begin) * 1000
        stage = "synchronize"
        sync_begin = time.perf_counter()
        torch.cuda.synchronize()
        report["invalid_sync_ms"] = (time.perf_counter() - sync_begin) * 1000
    except Exception as error:
        finish = time.perf_counter()
        report["error_stage"] = stage
        report["exception_type"] = type(error).__name__
        report["exception"] = str(error)
        report["invalid_total_ms"] = (finish - begin) * 1000
        if stage == "operator_call":
            report["invalid_call_ms"] = report["invalid_total_ms"]
        else:
            report["invalid_sync_ms"] = (finish - sync_begin) * 1000
    save()
    # A missing native check is evidence, not an accepted validation baseline.
    assert (
        report["error_stage"] is not None
    ), "Native accepted invalid input; see results.json"


@pytest.mark.parametrize("kind", ["flag_store", "flag_eop"])
@pytest.mark.parametrize("count", [1, 513])
@pytest.mark.parametrize("phase", ["timing", "invalid"])
def test_sdk_silent(request, count, phase, kind):
    """Separate reduction work from assertion runtime machinery in fresh processes."""
    if _isolated(request):
        return
    import importlib
    import json
    import statistics
    import time

    import torch
    import triton
    import triton.language as tl

    backend = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
    )
    import hashlib

    from triton.runtime.cache import get_cache_manager

    # Extracted from CoreX 4.5 SDK __assertfail LLVM in run 38697425454.
    # Diagnostic only: this private runtime ABI is not a production contract.
    ending = 'call void asm sideeffect "sl_eop;", ""()' if kind == "flag_eop" else ""
    adapter = r"""
 target triple = "bi-iluvatar-ilurt"
 @gAssertReason = weak dso_local addrspace(1) externally_initialized global ptr null, align 8
 @llvm.compiler.used = appending global [1 x ptr]
   [ptr addrspacecast (ptr addrspace(1) @gAssertReason to ptr)], section "llvm.metadata"
 define i32 @flaggems_corex_assert(i32 %code) alwaysinline nounwind {
 entry:
   %ok = icmp eq i32 %code, 0
   br i1 %ok, label %end, label %invalid
 invalid:
   %reason = load ptr, ptr addrspace(1) @gAssertReason, align 8
   %missing = icmp eq ptr %reason, null
   br i1 %missing, label %end, label %signal
 signal:
   store i32 -2006555000, ptr %reason, align 4
   ENDING
   br label %end
 end:
   ret i32 %code
 }
 """.replace("ENDING", ending)
    cache = get_cache_manager(hashlib.sha256(adapter.encode()).hexdigest())
    library = cache.get_file("silent-assert.ll")
    if library is None:
        library = cache.put(adapter, "silent-assert.ll", binary=False)

    @tl.core.builtin
    def sdk_assert(code, _semantic=None):
        # Build a real Python list outside JIT AST lowering. CoreX Triton 3.6
        # extern_elementwise calls args.copy(), which a TL tuple cannot provide.
        return tl.core.extern_elementwise(
            "",
            "",
            [code],
            {(tl.int32,): ("flaggems_corex_assert", tl.int32)},
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit(do_not_specialize=["count"])
    def check(flags, count, block: tl.constexpr):
        lane = tl.arange(0, block)
        bad = tl.full((), 0, tl.int32)
        for base in range(0, count, block):
            values = tl.load(flags + base + lane, base + lane < count, other=0)
            bad |= tl.max(values, 0)
        sdk_assert(bad)

    path = Path("../ix-results/sdk-silent") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    flags = torch.zeros(count, dtype=torch.int32, device="cuda")
    (path / "sdk-adapter.ll").write_text(Path(library).read_text())

    def launch():
        return check[(1,)](
            flags,
            count,
            256,
            debug=False,
            extern_libs={"embedding_bag_assert": library},
        )

    # Warm the exact assertion specialization before any embedding-bag kernels.
    # No host wait or readback is inserted into this candidate sequence.
    compiled = launch()
    for name, code in compiled.asm.items():
        if isinstance(code, str):
            (path / (name + ".txt")).write_text(code)
    report = {
        "kind": kind,
        "count": count,
        "phase": phase,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "kernel_hash": compiled.hash,
    }

    def save():
        (path / "results.json").write_text(json.dumps(report, indent=2))

    report["asm_instrumentation"] = [
        line
        for line in compiled.asm.get("asm", "").splitlines()
        if "has_called_assert" in line or "has_called_printf" in line
    ]
    save()
    assert "has_called_printf = 0" in "\n".join(
        report["asm_instrumentation"]
    ), "Unexpected printf instrumentation; inspect compiler artifacts"
    if phase == "invalid":
        backend = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
        )
        backward = importlib.import_module(
            "flag_gems.runtime.backend._iluvatar.ops.embedding_bag_backward"
        )
        weight = torch.ones((16, 8), device="cuda")
        indices = torch.tensor([1, 2, 3, 4], device="cuda")
        offsets = torch.tensor([0, 2], device="cuda")
        output, mapping, sizes, maxima = backend._embedding_bag(
            weight, indices, offsets
        )
        backward._embedding_bag_backward(
            torch.ones_like(output),
            indices,
            offsets,
            mapping,
            sizes,
            maxima,
            16,
            False,
            0,
            False,
        )
        # Preserve the asynchronous forward/backward-to-check sequence.
        flags[-1:].fill_(1)
        try:
            launch()
            torch.cuda.synchronize()
        except Exception as error:
            report["exception_type"] = type(error).__name__
            report["exception"] = str(error)
            report["caught"] = "assert" in str(error).lower()
        else:
            report["caught"] = False
        save()
        assert report["caught"], report
        return

    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    report["triton_ms"] = [
        triton.testing.do_bench(launch, warmup=20, rep=60, return_mode="median")
        for _ in range(3)
    ]
    report["median_ms"] = statistics.median(report["triton_ms"])
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    begin = time.perf_counter()
    for _ in range(50):
        launch()
    report["submit_ms"] = (time.perf_counter() - begin) * 1000 / 50
    end.record()
    end.synchronize()
    report["batch_event_ms"] = start.elapsed_time(end) / 50
    save()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path / "trace.json"))
    (path / "profiler.txt").write_text(prof.key_averages().table(row_limit=30))


def test_native_forensics_build(request):
    """Collect installed build/source evidence without claiming upstream identity."""
    if _isolated(request):
        return
    import hashlib
    import json
    import shutil
    import time

    import torch

    path = Path("../ix-results/native-forensics/build")
    path.mkdir(parents=True, exist_ok=True)
    root = Path(torch.__file__).parent
    report = {
        "torch": torch.__version__,
        "git_version": torch.version.git_version,
        "torch_file": str(torch.__file__),
        "cuda_version": torch.version.cuda,
        "config": torch.__config__.show(),
        "sources": [],
        "binaries": [],
        "tools": {
            name: shutil.which(name)
            for name in (
                "nm",
                "readelf",
                "objdump",
                "cuobjdump",
                "ixobjdump",
                "llvm-objdump",
            )
        },
    }
    (path / "dispatch.txt").write_text(
        "\n".join(
            torch._C._dispatch_dump_table("aten::" + name)
            for name in ("_embedding_bag", "_embedding_bag_forward_only")
        )
    )
    # Bounded search: never traverse /proc, /sys, devices or home credentials.
    deadline = time.monotonic() + 25
    roots = [root, Path("/opt"), Path("/usr/local/corex"), Path("/workspace")]
    targets = {
        "EmbeddingBag.cu",
        "EmbeddingBag.cpp",
        "CUDADeviceAssertion.h",
        "Macros.h",
    }
    for search_root in roots:
        if not search_root.exists():
            continue
        for base, dirs, files in os.walk(search_root):
            dirs[:] = [
                d
                for d in dirs
                if d not in (".git", "__pycache__", ".cache", "node_modules")
            ]
            if time.monotonic() > deadline:
                report["source_search_time_limited"] = True
                break
            for name in files:
                if name not in targets:
                    continue
                source = Path(base) / name
                if source.stat().st_size > 2_000_000 or len(report["sources"]) >= 40:
                    continue
                data = source.read_bytes()
                dest = f"source-{len(report['sources'])}-{name}.txt"
                (path / dest).write_bytes(data)
                report["sources"].append(
                    {
                        "path": str(source),
                        "artifact": dest,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
    # Filter potentially large symbol tables while streaming, keeping bounded logs.
    import shlex

    candidates = sorted((root / "lib").glob("*.so*"))
    for binary in candidates:
        if not any(
            x in binary.name for x in ("torch_cuda", "torch_corex", "torch_cpu")
        ):
            continue
        if len(report["binaries"]) >= 3:
            break
        info = {"path": str(binary), "bytes": binary.stat().st_size}
        for label, command in (
            ("symbols", "nm -D -C"),
            ("sections", "readelf -SW"),
            ("build-id", "readelf -n"),
        ):
            expression = "EmbeddingBag|embedding_bag|assertfail|\\.debug|\\.text|Build ID|corex|iluvatar"
            script = f"{command} {shlex.quote(str(binary))} | grep -E {shlex.quote(expression)} | head -n 160"
            try:
                result = subprocess.run(
                    script,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                (path / f"{binary.name}-{label}.txt").write_text(
                    result.stdout + result.stderr
                )
                info[label + "_returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                info[label + "_timeout"] = True
        report["binaries"].append(info)
    (path / "build.json").write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize("variant", ["_embedding_bag", "_embedding_bag_forward_only"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("invalid", ["upper", "negative"])
def test_native_forensics_indices(request, variant, mode, invalid):
    """Use guard rows in one allocation to distinguish masking from raw row reads."""
    if _isolated(request):
        return
    import json

    import torch

    path = Path("../ix-results/native-forensics") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    # Invalid logical indices still address allocated guard rows. This avoids
    # intentionally accessing another allocation while testing native behavior.
    backing = torch.tensor([100.0, 200.0, 30.0, 40.0, 50.0, 60.0], device="cuda")
    weight = backing[2:4].view(2, 1)
    bad = 2 if invalid == "upper" else -2
    indices = torch.tensor([0, bad, 1], device="cuda")
    offsets = torch.tensor([0], device="cuda")
    native = getattr(torch.ops.aten, variant).default
    report = {
        "variant": variant,
        "mode": mode,
        "indices": [0, bad, 1],
        "weight_rows": [30.0, 40.0],
        "guard_value": 50.0 if bad == 2 else 100.0,
        "storage_offset": weight.storage_offset(),
        "torch": torch.__version__,
    }
    stage = "operator_call"
    try:
        output = native(weight, indices, offsets, False, mode, False, None, False, -1)
        stage = "synchronize"
        torch.cuda.synchronize()
        stage = "diagnostic_output_readback"
        # Test-only readback records semantics; never used in an operator path.
        report["outputs"] = [tensor.cpu().tolist() for tensor in output]
        report["error_stage"] = None
    except Exception as error:
        report["error_stage"] = stage
        report["exception_type"] = type(error).__name__
        report["exception"] = str(error)
    report["masked_expected"] = [70.0, 35.0, 40.0][mode]
    guard = report["guard_value"]
    report["raw_row_read_expected"] = [
        70.0 + guard,
        (70.0 + guard) / 3,
        max(40.0, guard),
    ][mode]
    (path / "results.json").write_text(json.dumps(report, indent=2))
    # This is observation, not an assertion that invalid native behavior is valid.


@pytest.mark.parametrize("dtype_name", ["float16", "float32", "float64", "bfloat16"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("padding", [-1, 0])
def test_masked_indices(request, dtype_name, index_dtype, mode, sparse, padding):
    """Invalid rows act like padding, including all-invalid and empty bags."""
    if _isolated(request):
        return
    import torch

    import flag_gems

    if dtype_name == "bfloat16" and not flag_gems.runtime.device.support_bf16:
        pytest.skip("Device does not support bfloat16")
    if dtype_name == "float64" and not flag_gems.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    dtype = getattr(torch, dtype_name)
    idtype = getattr(torch, index_dtype)
    device = flag_gems.device
    weight = torch.arange(24, device=device, dtype=dtype).reshape(6, 4)
    indices = torch.tensor([1, -2, 6, 2, -100, 100], device=device, dtype=idtype)
    offsets = torch.tensor([0, 4, 6, 6], device=device, dtype=idtype)
    # Reference uses an explicit padding row, not native's unsafe OOB behavior.
    safe = torch.tensor([1, 0, 0, 2, 0, 0], device=device, dtype=idtype)
    for variant in ("_embedding_bag", "_embedding_bag_forward_only"):
        op = getattr(flag_gems, variant)
        actual = op(weight, indices, offsets, mode=mode, padding_idx=padding)
        expected = op(weight, safe, offsets, mode=mode, padding_idx=0)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref)
        grad = torch.ones_like(actual[0])
        backward = flag_gems._embedding_bag_backward
        got = backward(
            grad, indices, offsets, *actual[1:], 6, False, mode, sparse, None, padding
        )
        ref = backward(
            grad, safe, offsets, *expected[1:], 6, False, mode, sparse, None, 0
        )
        torch.testing.assert_close(
            got.to_dense() if sparse else got, ref.to_dense() if sparse else ref
        )
    flag_gems.runtime.torch_device_fn.synchronize()


@pytest.mark.parametrize(
    "error", ["offset-negative", "offset-large", "mapping", "size", "maximum"]
)
def test_masked_device_metadata(request, error):
    if _isolated(request):
        return
    import torch

    import flag_gems

    device = flag_gems.device
    weight = torch.ones((4, 8), device=device)
    indices = torch.tensor([1, 2, 1], device=device)
    offsets = torch.tensor([0, 2], device=device)
    mode = 2 if error == "maximum" else 1
    output, mapping, sizes, maximum = flag_gems._embedding_bag(
        weight, indices, offsets, mode=mode
    )
    if error.startswith("offset"):
        offsets[1] = -100 if error.endswith("negative") else 100
        for op in (flag_gems._embedding_bag, flag_gems._embedding_bag_forward_only):
            result = op(weight, indices, offsets, mode=mode)[0]
            assert torch.isfinite(result).all().item()
    elif error == "mapping":
        mapping[0] = 100
    elif error == "size":
        sizes[0] = -100
    else:
        maximum[0] = 100
    grad = torch.ones_like(output)
    result = flag_gems._embedding_bag_backward(
        grad, indices, offsets, mapping, sizes, maximum, 4, False, mode, False
    )
    flag_gems.runtime.torch_device_fn.synchronize()
    assert torch.isfinite(result).all().item()


@pytest.mark.parametrize("dtype_name", ["float16", "float32", "bfloat16"])
@pytest.mark.parametrize(
    "shape",
    [
        (8, 16, 128, 4),
        (32, 64, 1024, 8),
        (128, 128, 4096, 16),
        (256, 256, 4096, 32),
        (1024, 64, 16384, 64),
        (128, 512, 4096, 8),
        (5, 33, 8, 7),
        (4, 0, 8, 3),
    ],
)
def test_max_candidate_accuracy(dtype_name, shape):
    """Only the changed MAX paths: strides, ties, NaNs and masked indirect loads."""
    import torch

    import flag_gems

    if dtype_name == "bfloat16" and not flag_gems.runtime.device.support_bf16:
        pytest.skip("Device does not support bfloat16")
    dtype = getattr(torch, dtype_name)
    bags, dim, rows, length = shape
    device = flag_gems.device
    torch.manual_seed(5764)
    weight = torch.randn((rows, dim * 2), dtype=dtype)[:, ::2]
    values = torch.randint(1, rows, (bags * length,), dtype=torch.int64)
    # Invalid rows and padding must have exactly the same masking as before.
    values[::11] = rows + 100
    values[1::13] = -2
    values[2::17] = 0
    values[:length] = rows
    offsets = torch.arange(0, values.numel() + 1, length, dtype=torch.int64)
    if dim == 33:
        values[length : 2 * length] = torch.tensor([1, 2, 3, 4, 4, 5, 6])
        values[2 * length : 3 * length] = torch.tensor([2, 1, 3, 4, 4, 5, 6])
        offsets[4] = offsets[3]  # An empty bag before the final bag.
        weight[1, :4] = float("nan")
        weight[2, :4] = float("inf")
        weight[3, :4] = float("-inf")
        weight[4:, 4:8] = 0
    expected = torch.zeros((bags, dim), dtype=dtype)
    winners = torch.full((bags, dim), -1, dtype=torch.int64)
    counts = torch.zeros(bags, dtype=torch.int64)
    for bag in range(bags):
        valid = values[offsets[bag].item() : offsets[bag + 1].item()]
        valid = valid[(valid > 0) & (valid < rows)]
        counts[bag] = valid.numel()
        for position, row in enumerate(valid.tolist()):
            replace = (
                torch.ones(dim, dtype=torch.bool)
                if position == 0
                else weight[row] > expected[bag]
            )
            expected[bag] = torch.where(replace, weight[row], expected[bag])
            winners[bag] = torch.where(replace, row, winners[bag])
    # Transfer the backing storage then slice, retaining noncontiguous strides.
    weight_gpu = torch.empty((rows, dim * 2), dtype=dtype, device=device)[:, ::2]
    weight_gpu.copy_(weight)
    indices_gpu, offsets_gpu = values.to(device), offsets.to(device)
    grad = torch.randn((bags, dim * 2), dtype=dtype)[:, ::2]
    grad_gpu = torch.empty((bags, dim * 2), dtype=dtype, device=device)[:, ::2]
    grad_gpu.copy_(grad)
    for variant in ("_embedding_bag", "_embedding_bag_forward_only"):
        out, mapping, sizes, maximum = getattr(flag_gems, variant)(
            weight_gpu,
            indices_gpu,
            offsets_gpu,
            mode=2,
            include_last_offset=True,
            padding_idx=0,
        )
        torch.testing.assert_close(out.cpu(), expected, equal_nan=True, rtol=0, atol=0)
        torch.testing.assert_close(maximum.cpu(), winners, rtol=0, atol=0)
        # IX keeps one bag_size entry per offset, including a zero terminal
        # entry when include_last_offset=True. Check the complete ABI tensor.
        expected_sizes = torch.cat((counts, counts.new_zeros(1)))
        torch.testing.assert_close(sizes.cpu(), expected_sizes, rtol=0, atol=0)
        # Include malformed saved rows; every atomic destination remains masked.
        invalid_max = winners.clone()
        if dim:
            invalid_max[1, 0] = rows + 100
            invalid_max[2, 0] = -2
        expected_grad = torch.zeros((rows, dim), dtype=torch.float32)
        feature = torch.arange(dim)[None, :].expand(bags, dim)
        valid = (invalid_max > 0) & (invalid_max < rows) & (counts[:, None] > 0)
        destination = invalid_max * dim + feature
        expected_grad.view(-1).index_add_(0, destination[valid], grad.float()[valid])
        actual_grad = flag_gems._embedding_bag_backward(
            grad_gpu,
            indices_gpu,
            offsets_gpu,
            mapping,
            sizes,
            invalid_max.to(device),
            rows,
            True,
            2,
            False,
            None,
            0,
        )
        torch.testing.assert_close(actual_grad.cpu(), expected_grad.to(dtype))
    flag_gems.runtime.torch_device_fn.synchronize()
