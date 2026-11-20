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

import ctypes
import logging
import os
import sys
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from io import StringIO
from tempfile import TemporaryFile
from types import SimpleNamespace

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from tests.test_scaled_mm import (
    METAX_ERROR,
    _device_key,
    _load_modules,
    _pytest_environment,
    _unsupported_reason,
    case_id,
    cases,
    comparison_tolerance,
    golden,
    make_inputs,
    require_case,
)

from . import base, conftest, consts

logger = logging.getLogger(__name__)


@contextmanager
def _native_probe_output():
    state = SimpleNamespace(unsupported=False)
    if flag_gems.vendor_name != "metax":
        yield state
        return

    # MACA writes directly to native stdout/stderr, bypassing Python logging.
    # Keep unknown diagnostics even when the exception is a capability rejection.
    expected = (
        b"invalid matrix datatype, matC->type(2), matD->type(28)",
        b"invalid matrix datatype, matC->type(2), matD->type(29)",
        b"narrow-precision only supports block scaling_1D1D or 2D1D or tensorwide",
    )
    libc = ctypes.CDLL(None)
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int

    def flush():
        sys.stdout.flush()
        sys.stderr.flush()
        libc.fflush(None)

    with ExitStack() as resources:
        streams = [
            (
                fd,
                resources.enter_context(TemporaryFile()),
                resources.enter_context(os.fdopen(os.dup(fd), "wb", buffering=0)),
            )
            for fd in (1, 2)
        ]
        flush()
        try:
            with ExitStack() as redirects:
                for fd, captured, original in streams:
                    redirects.callback(os.dup2, original.fileno(), fd)
                    os.dup2(captured.fileno(), fd)
                try:
                    yield state
                finally:
                    flush()
        finally:
            # Descriptors are restored before replay, including on exceptions.
            for _, captured, original in streams:
                captured.seek(0)
                for line in captured:
                    _, marker, message = line.partition(b"[MCBLASLT][E]")
                    if state.unsupported and marker and message.strip() in expected:
                        continue
                    original.write(line)


@lru_cache(None)
def _native_reason(case, shape, use_out, device_key, reference):
    if not hasattr(torch, "_scaled_mm"):
        return "torch._scaled_mm is unavailable"
    args, kwargs = make_inputs(case, shape)
    with _native_probe_output() as output:
        try:
            if use_out:
                out = torch.empty(
                    (shape[0], shape[1]),
                    dtype=case[2] or case[0],
                    device=flag_gems.device,
                )
                actual = torch.ops.aten._scaled_mm.out(*args, **kwargs, out=out)
            else:
                actual = torch._scaled_mm(*args, **kwargs)
            torch_device_fn.synchronize()
            tol = comparison_tolerance(actual.dtype)
            torch.testing.assert_close(
                actual.cpu().float(),
                reference(args, kwargs).float(),
                rtol=tol,
                atol=tol,
            )
        except AssertionError as exc:
            return f"Native probe disagrees with FP32 golden: {str(exc)[:300]}"
        except Exception as exc:
            reason = _unsupported_reason(
                exc, dtype=case[0], other_dtype=case[1], stage="native", scaling=case[3]
            )
            output.unsupported = bool(reason)
            return reason
    return None


def native_reason(case, shape, use_out=False, *, reference):
    return _native_reason(case, shape, use_out, _device_key(), reference)


CORE_SHAPES = [(16, 16, 16), (128, 128, 128), (512, 512, 512)]


# Do not compile kernels or call native operators while collecting benchmarks.
CASES = tuple(cases())
DTYPES = tuple(dict.fromkeys(case[0] for case in CASES))


def _prepare_benchmark(case, shape, use_out, request, monkeypatch):
    config = conftest.Config
    if config.query or config.list_cases:
        return
    require_case(case)
    if config.preflight_only or config.profile_only or config.skip_native:
        return
    reason = native_reason(case, shape, use_out, reference=golden)
    if reason:
        if request is not None:
            request.node.add_marker(
                pytest.mark.skip_native(vendors=flag_gems.vendor_name, reason=reason)
            )
        # The autouse baseline fixture has already run; update its per-case
        # configuration as well as the reporting marker, then restore it.
        monkeypatch.setattr(config, "skip_native", True)
        monkeypatch.setattr(config, "native_baseline_skip_reason", reason)


class ScaledMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, dtype, use_out):
        self.use_out = use_out
        op_name = "scaled_mm_out" if use_out else "scaled_mm"
        op = torch.ops.aten._scaled_mm.out if use_out else torch._scaled_mm
        super().__init__(op_name, op, dtypes=[dtype])
        self.set_gems(flag_gems.scaled_mm_out if use_out else flag_gems.scaled_mm)

    def set_shapes(self, shape_file_path=None):
        self.shapes = CORE_SHAPES
        self.shape_desc = "M, N, K"

    def _build_input(self):
        args, kwargs = make_inputs(self.case, self.shape)
        if self.use_out:
            kwargs["out"] = torch.empty(
                (self.shape[0], self.shape[1]),
                dtype=self.case[2] or self.case[0],
                device=flag_gems.device,
            )
        return (*args, kwargs)

    def _run_legacy(self):
        # A pytest item/table groups one A dtype, with B/output dtypes and all
        # other parameters retained as rows. Restore baseline state per row.
        results = []
        failed = skipped = total = 0
        for dtype in self.to_bench_dtypes:
            metrics = []
            skipped_rows = native_skipped_rows = 0
            for case in CASES:
                if case[0] != dtype:
                    continue
                for shape in self.shapes:
                    self.case, self.shape = case, shape
                    total += 1
                    with pytest.MonkeyPatch.context() as patch:
                        try:
                            _prepare_benchmark(case, shape, self.use_out, None, patch)
                            native_skipped_rows += int(conftest.Config.skip_native)
                            metric = self._measure_input(
                                self._build_input(), dtype=dtype
                            )
                        except pytest.skip.Exception:
                            skipped += 1
                            skipped_rows += 1
                            continue
                        except (Exception, pytest.fail.Exception) as exc:
                            failed += 1
                            detail = self.record_shapes()
                            detail["error"] = str(exc)[-300:]
                            metric = consts.BenchmarkMetrics(
                                shape_detail=detail, error_msg=str(exc)
                            )
                        metrics.append(metric)
            if metrics:
                results.append(self._emit_result(dtype, metrics))
            logger.info(
                "%s dtype=%s: %d result rows, %d skipped rows, %d native baselines skipped",
                self.op_name,
                dtype,
                len(metrics),
                skipped_rows,
                native_skipped_rows,
            )
        if failed:
            pytest.fail(
                f"{failed}/{total} scaled_mm benchmark rows failed; see table/JSON"
            )
        if total and skipped == total:
            pytest.skip(f"All {total} benchmark rows skipped")
        return results

    def record_shapes(self, *args, **kwargs):
        return {
            "fp8_case": case_id(self.case),
            "M_N_K": self.shape,
        }

    def get_tflops(self, op, *args, **kwargs):
        return 2 * args[0].shape[0] * args[0].shape[1] * args[1].shape[1]


@pytest.mark.scaled_mm
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_scaled_mm_benchmark(dtype):
    ScaledMMBenchmark(dtype, False).run()


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_scaled_mm_out_benchmark(dtype):
    ScaledMMBenchmark(dtype, True).run()


# Host-only regression checks: no capability probe may run during collection.
pytest_plugins = ["pytester"]


@pytest.fixture
def modules(monkeypatch):
    return _load_modules(monkeypatch)


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    "outcome", ["unsupported", "unknown", "oom", "mismatch", "success", "interrupt"]
)
def test_native_probe_output_preserves_unexpected_diagnostics(
    modules, monkeypatch, capfd, use_out, outcome
):
    module = modules.benchmark
    case = (
        "torch.float8_e4m3fn",
        "torch.float8_e4m3fn",
        "torch.bfloat16",
        "scalar",
        True,
    )
    known = (
        b"[05:57:42.335][MCBLASLT][E]invalid matrix datatype, matC->type(2), matD->type(28)\n"
        b"[05:57:42.335][MCBLASLT][E]invalid matrix datatype, matC->type(2), matD->type(29)\n",
        b"[MCBLASLT][E]narrow-precision only supports block scaling_1D1D or 2D1D or tensorwide\n",
    )
    other = b"[MCBLASLT][E]unexpected diagnostic must survive\n"
    error = {
        "unsupported": RuntimeError(METAX_ERROR),
        "unknown": RuntimeError("unexpected native failure"),
        "oom": RuntimeError("out of memory"),
        "interrupt": KeyboardInterrupt(),
    }.get(outcome)
    value = SimpleNamespace(dtype="torch.bfloat16")
    value.cpu = value.float = lambda: value
    libc = ctypes.CDLL(None)
    libc.puts.argtypes = [ctypes.c_char_p]
    libc.puts.restype = ctypes.c_int

    def native(*args, **kwargs):
        assert ("out" in kwargs) is use_out
        # Exercise buffered C stdio as well as direct fd writes; redirect_stdout
        # alone cannot capture either. Unknown messages must survive a rejection.
        libc.puts(known[0].rstrip(b"\n"))
        os.write(2, known[1])
        os.write(2, other)
        if error is not None:
            raise error
        return value

    def check(*args, **kwargs):
        if outcome == "mismatch":
            raise AssertionError("incorrect native result")

    monkeypatch.setattr(module, "make_inputs", lambda *args: ((), {}))
    monkeypatch.setattr(module, "comparison_tolerance", lambda dtype: 1e-3)
    monkeypatch.setattr(module.torch, "_scaled_mm", native, raising=False)
    monkeypatch.setattr(module.torch, "empty", lambda *a, **kw: value, raising=False)
    monkeypatch.setattr(
        module.torch,
        "ops",
        SimpleNamespace(aten=SimpleNamespace(_scaled_mm=SimpleNamespace(out=native))),
        raising=False,
    )
    monkeypatch.setattr(
        module.torch, "testing", SimpleNamespace(assert_close=check), raising=False
    )
    monkeypatch.setattr(
        module.torch_device_fn, "synchronize", lambda: None, raising=False
    )
    # Flush pending output before redirecting, and restore both descriptors after
    # the probe even if classification, correctness checking, or interruption fails.
    libc.puts(b"before probe")
    if outcome in ("unknown", "oom", "interrupt"):
        with pytest.raises(type(error)) as caught:
            module._native_reason(
                case, (16, 16, 16), use_out, "device", lambda *a: value
            )
        assert caught.value is error
    else:
        reason = module._native_reason(
            case, (16, 16, 16), use_out, "device", lambda *a: value
        )
        if outcome == "unsupported":
            assert "MetaX native FP8 scalar baseline unavailable" in reason
        elif outcome == "mismatch":
            assert "Native probe disagrees with FP32 golden" in reason
        else:
            assert reason is None
    os.write(1, b"candidate stdout\n")
    os.write(2, b"candidate stderr\n")
    captured = capfd.readouterr()
    assert (
        captured.out
        == "before probe\n"
        + ("" if outcome == "unsupported" else known[0].decode())
        + "candidate stdout\n"
    )
    assert (
        captured.err
        == ("" if outcome == "unsupported" else known[1].decode())
        + other.decode()
        + "candidate stderr\n"
    )


@pytest.mark.parametrize("out", [False, True])
@pytest.mark.parametrize(
    "a,b,scaling,message",
    [
        ("float8_e4m3fn", "float8_e4m3fn", "scalar", METAX_ERROR),
        ("float8_e4m3fn", "float8_e5m2", "scalar", METAX_ERROR),
        ("float8_e5m2", "float8_e4m3fn", "scalar", METAX_ERROR),
        (
            "float8_e4m3fn",
            "float8_e4m3fn",
            "rowwise",
            "Rowwise scaling is not currently supported on your device",
        ),
    ],
)
def test_native_rejection_still_runs_candidate(
    modules, monkeypatch, out, a, b, scaling, message
):
    module = modules.benchmark
    case = (f"torch.{a}", f"torch.{b}", "torch.bfloat16", scaling, True)
    calls = []

    def rejected(*args, **kwargs):
        assert ("out" in kwargs) is out
        calls.append("native rejected")
        raise RuntimeError(message)

    monkeypatch.setattr(module, "require_case", lambda case: None)
    monkeypatch.setattr(module, "_device_key", lambda: ("cuda", 0))
    monkeypatch.setattr(module, "make_inputs", lambda *args: ((), {}))
    monkeypatch.setattr(module.torch, "_scaled_mm", rejected, raising=False)
    monkeypatch.setattr(
        module.torch, "empty", lambda *args, **kwargs: object(), raising=False
    )
    monkeypatch.setattr(
        module.torch,
        "ops",
        SimpleNamespace(aten=SimpleNamespace(_scaled_mm=SimpleNamespace(out=rejected))),
        raising=False,
    )
    monkeypatch.setattr(module, "CASES", (case,))
    bench = object.__new__(module.ScaledMMBenchmark)
    bench.use_out = out
    bench.op_name = "scaled_mm_out" if out else "scaled_mm"
    bench.to_bench_dtypes = [case[0]]
    bench.shapes = [(16, 16, 16)]
    monkeypatch.setattr(bench, "_build_input", lambda: ())
    candidate_fails = False

    def measure(input, *, dtype):
        assert dtype == case[0]
        assert modules.config.skip_native is True
        reason = modules.config.native_baseline_skip_reason
        assert f"native FP8 {scaling}" in reason and message in reason
        calls.append("candidate ran")
        if candidate_fails:
            raise RuntimeError("candidate kernel failure")
        return module.consts.BenchmarkMetrics(
            shape_detail=bench.record_shapes(), latency=1.0
        )

    monkeypatch.setattr(bench, "_measure_input", measure, raising=False)
    with redirect_stdout(StringIO()):
        results = bench._run_legacy()
    assert calls == ["native rejected", "candidate ran"]
    assert results[0].result[0].latency_base is None
    assert results[0].result[0].speedup is None
    assert modules.config.skip_native is False
    candidate_fails = True
    with redirect_stdout(StringIO()), pytest.raises(pytest.fail.Exception, match="1/1"):
        bench._run_legacy()
    assert calls == ["native rejected", "candidate ran", "candidate ran"]
    assert (
        modules.conftest.results[-1]["result"][0]["error_msg"]
        == "candidate kernel failure"
    )


def test_runtime_skips_errors_and_baseline_state(pytester, monkeypatch):
    _pytest_environment(pytester, monkeypatch)
    pytester.makepyfile("""
import pytest
from conftest import modules

def check(value):
    if value == "unsupported":
        return "known capability limitation"
    if value == "broken":
        raise RuntimeError("unknown probe failure")
    return None
modules.accuracy.case_reason = check
modules.benchmark.native_reason = lambda case, shape, use_out, **kw: (
    "native unavailable" if case == "no-native" else None
)
@pytest.mark.parametrize("case", ["supported", "unsupported", "broken"])
def test_accuracy(case):
    modules.accuracy.require_case(case)
    assert case == "supported"
@pytest.mark.parametrize("case", ["no-native", "supported", "unsupported", "broken", "supported"])
@pytest.mark.parametrize("shape", [(16, 16, 16)])
def test_benchmark(case, shape, request, monkeypatch):
    modules.benchmark._prepare_benchmark(case, shape, False, request, monkeypatch)
    assert modules.config.skip_native == (case == "no-native")
    assert modules.config.native_baseline_skip_reason == ("native unavailable" if case == "no-native" else None)
def test_final_reset():
    assert modules.config.skip_native is False
    assert modules.config.native_baseline_skip_reason is None
""")
    # Expected child failures must not look like failures of this regression.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        result = pytester.runpytest_subprocess("-q", "--tb=short")
    diagnostics = result.stdout.str() + result.stderr.str()
    assert result.ret == pytest.ExitCode.TESTS_FAILED, diagnostics
    assert result.parseoutcomes() == {
        "passed": 5,
        "skipped": 2,
        "failed": 2,
    }, diagnostics
    assert "unknown probe failure" in result.stdout.str()


@pytest.mark.parametrize(
    "mode", ["query", "list_cases", "preflight_only", "profile_only", "skip_native"]
)
def test_metadata_and_candidate_modes_do_not_probe_native(modules, monkeypatch, mode):
    monkeypatch.setattr(modules.config, mode, True)
    calls = []
    monkeypatch.setattr(
        modules.benchmark, "require_case", lambda case: calls.append(case)
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("unexpected native probe")

    monkeypatch.setattr(modules.benchmark, "native_reason", unexpected)
    modules.benchmark._prepare_benchmark("case", (16, 16, 16), False, None, monkeypatch)
    assert calls == ([] if mode in ("query", "list_cases") else ["case"])


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("inject_failure", [False, True])
def test_grouped_tables_preserve_rows_and_baseline_state(
    modules, monkeypatch, tmp_path, use_out, inject_failure
):
    from .summary_for_plot import all_benchshape_passed, parse_log

    module = modules.benchmark
    attempted = set()

    def require(case):
        attempted.add((case, bench.shape))
        if any("fnuz" in dtype for dtype in case[:2]):
            pytest.skip("input FNUZ unsupported")

    monkeypatch.setattr(module, "require_case", require)
    monkeypatch.setattr(
        module,
        "native_reason",
        lambda case, *args, **kwargs: (
            "native unavailable" if case[1].endswith("e4m3fn") else None
        ),
    )
    visited = set()
    tables = []
    for dtype in module.DTYPES:
        bench = object.__new__(module.ScaledMMBenchmark)
        bench.op_name = "scaled_mm_out" if use_out else "scaled_mm"
        bench.use_out = use_out
        bench.to_bench_dtypes = [dtype]
        bench.shapes = module.CORE_SHAPES
        monkeypatch.setattr(bench, "_build_input", lambda: ())

        def measure(input, *, dtype):
            assert dtype == bench.case[0]
            case, shape = bench.case, bench.shape
            visited.add((case, shape))
            skip_native = case[1].endswith("e4m3fn")
            assert modules.config.skip_native is skip_native
            assert modules.config.native_baseline_skip_reason == (
                "native unavailable" if skip_native else None
            )
            if (
                inject_failure
                and case == module.CASES[0]
                and shape == module.CORE_SHAPES[0]
            ):
                raise RuntimeError("candidate failed on first row")
            return module.consts.BenchmarkMetrics(
                shape_detail=bench.record_shapes(),
                latency=1.0,
                latency_base=None if skip_native else 2.0,
                speedup=None if skip_native else 2.0,
                tflops=1.0,
            )

        monkeypatch.setattr(bench, "_measure_input", measure, raising=False)
        output = StringIO()
        with redirect_stdout(output):
            if "fnuz" in dtype:
                with pytest.raises(pytest.skip.Exception, match="All 60"):
                    bench._run_legacy()
            elif inject_failure and dtype == module.DTYPES[0]:
                with pytest.raises(pytest.fail.Exception, match="1/60"):
                    bench._run_legacy()
            else:
                bench._run_legacy()
        table = output.getvalue()
        assert table.count("Performance Test") == (0 if "fnuz" in dtype else 1)
        assert "input FNUZ unsupported" not in table
        assert "native unavailable" not in table
        assert "SKIP" not in table
        tables.append(table)
        assert modules.config.skip_native is False
        assert modules.config.native_baseline_skip_reason is None

    assert len(visited) == 60  # Two executable A dtypes x two B dtypes x 5 x 3.
    assert attempted == {
        (case, shape) for case in module.CASES for shape in module.CORE_SHAPES
    }
    assert len(modules.conftest.results) == 2
    rows = [row for report in modules.conftest.results for row in report["result"]]
    assert len(rows) == 60
    assert {
        (row["shape_detail"]["fp8_case"], tuple(row["shape_detail"]["M_N_K"]))
        for row in rows
    } == {(module.case_id(case), shape) for case, shape in visited}
    assert sum(row["error_msg"] is not None for row in rows) == int(inject_failure)
    assert sum(row["latency"] is not None for row in rows) == 60 - int(inject_failure)
    # Only executed rows reach the unchanged reporting parser; a fully
    # skipped dtype must not emit an empty, apparently successful result.
    log = tmp_path / "result.log"
    log.write_text("\n".join("[INFO] " + record for record in modules.conftest.records))
    reports = parse_log(log)
    assert [all_benchshape_passed(report.result) for report in reports] == [
        not inject_failure,
        True,
    ]
