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

import ast
import importlib.util
import logging
import sys
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from io import StringIO
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

from .conftest import QUICK_MODE

logger = logging.getLogger(__name__)
FP8_NAMES = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
# output dtype, scaling scheme, bias. Matrix inputs are always FP8.
CONFIGS = (
    (torch.float16, "scalar", True),
    (torch.bfloat16, "scalar", True),
    (torch.float32, "scalar", False),
    (torch.bfloat16, "rowwise", True),
    (None, "scalar", False),
)


@triton.jit
def _convert_probe(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0.0).to(tl.float32)
    tl.store(Y + i, x, i < N)


@triton.jit
def _dot_probe(A, B, C):
    m = tl.arange(0, 16)
    n = tl.arange(0, 16)
    k = tl.arange(0, 32)
    a = tl.load(A + m[:, None] * 32 + k[None, :]).to(tl.float16)
    b = tl.load(B + k[:, None] * 16 + n[None, :]).to(tl.float16)
    tl.store(C + m[:, None] * 16 + n[None, :], tl.dot(a, b))


def _device_key():
    return str(flag_gems.device), torch_device_fn.current_device()


def _unsupported_reason(exc, *, dtype=None, other_dtype=None, stage=None, scaling=None):
    text = str(exc)
    # Resource exhaustion and device faults are not evidence of dtype support.
    if any(
        s in text.lower()
        for s in (
            "out of memory",
            "device-side assert",
            "illegal memory",
            "no space left on device",
            "unspecified launch failure",
        )
    ):
        raise exc
    # Match the unsupported FNUZ conversion, not arbitrary compiler failures.
    if (
        flag_gems.vendor_name == "ascend"
        and stage in ("input_conversion", "output_conversion")
        and str(dtype) in ("torch.float8_e4m3fnuz", "torch.float8_e5m2fnuz")
        and type(exc).__name__ == "MLIRCompilationError"
        and "'hivm.hir.vbrc' op failed to verify" in text
        and "operand at idx 0 should have element type f8E4M3FN" in text
        and "f8E5M2 type" in text
    ):
        return f"Ascend {stage} does not support {dtype}: {text[-1600:]}"
    # MACA rejects these generated, aligned native FP8 inputs. Record an
    # unavailable baseline, not a skipped Gems benchmark or a claim that the
    # device cannot run FP8. Require both operand types and the native stage;
    # the same error from a conversion, dot probe, or candidate must still fail.
    native_dtypes = ("torch.float8_e4m3fn", "torch.float8_e5m2")
    if (
        flag_gems.vendor_name == "metax"
        and stage == "native"
        and isinstance(exc, RuntimeError)
        and str(dtype) in native_dtypes
        and str(other_dtype) in native_dtypes
    ):
        if (
            scaling in ("scalar", "rowwise")
            and "torch.float8_e4m3fn" in (str(dtype), str(other_dtype))
            and "CUBLAS_STATUS_INVALID_VALUE when calling cublasLtMatmul" in text
        ) or (
            scaling == "rowwise"
            and "Rowwise scaling is not currently supported on your device" in text
        ):
            return f"MetaX native FP8 {scaling} baseline unavailable: {text[-1600:]}"
    if not any(
        s in text.lower()
        for s in (
            "not support",
            "doesn't support",
            "unsupported",
            "not implement",
            "cannot convert",
            "cannot cast",
            "only support",
            "has no attribute",
            "divisible",
            "expected b.dtype",
            "not_supported",
        )
    ):
        raise exc
    return f"{type(exc).__name__}: {text[-1600:]}"


@lru_cache(None)
def _dtype_reason(dtype, device_key, output=False):
    device, index = device_key
    stage = "allocation"
    try:
        with torch_device_fn.device(index):
            # CPU quantization avoids requiring an unrelated vendor cast operator.
            cpu = ((torch.arange(512).float() % 13 - 6) / 16).reshape(16, 32)
            a = cpu.to(dtype).to(device)
            y = torch.empty_like(cpu, device=device)
            stage = "input_conversion"
            _convert_probe[(2,)](a, y, 511, 256)
            torch_device_fn.synchronize()
            torch.testing.assert_close(
                y.cpu().flatten()[:511],
                cpu.to(dtype).float().flatten()[:511],
                rtol=0,
                atol=0,
            )
            if output:
                stage = "output_conversion"
                _convert_probe[(2,)](y, a, 511, 256)
                torch_device_fn.synchronize()
                torch.testing.assert_close(
                    a.cpu().float().flatten()[:511],
                    cpu.to(dtype).float().flatten()[:511],
                    rtol=0,
                    atol=0,
                )
            else:
                b = cpu.t().contiguous().to(dtype).to(device)
                c = torch.empty((16, 16), dtype=torch.float32, device=device)
                stage = "dot"
                _dot_probe[(1,)](a, b, c)
                torch_device_fn.synchronize()
                torch.testing.assert_close(
                    c.cpu(),
                    cpu.to(dtype).float() @ cpu.t().to(dtype).float(),
                    rtol=1e-4,
                    atol=1e-4,
                )
    except Exception as exc:
        reason = _unsupported_reason(exc, dtype=dtype, stage=stage)
        logger.info(
            "FP8 feature check failed: device=%s dtype=%s output=%s reason=%s",
            device_key,
            dtype,
            output,
            reason,
        )
        return reason
    return None


def dtype_reason(dtype, output=False):
    return _dtype_reason(dtype, _device_key(), output)


def cases():
    dtypes = [getattr(torch, name) for name in FP8_NAMES if hasattr(torch, name)]
    return [
        (a, b, out, mode, bias)
        for a, b in product(dtypes, repeat=2)
        for out, mode, bias in CONFIGS
    ]


def case_id(case):
    return "-".join(str(x).replace("torch.", "") for x in case)


def case_reason(case):
    a, b, out, _, _ = case
    reason = dtype_reason(a) or dtype_reason(b)
    if reason is None and out is None:
        reason = dtype_reason(a, output=True)
    return reason


def make_inputs(case, shape, layout="column", fast=False, scale_result=None):
    a_dtype, b_dtype, out_dtype, mode, has_bias = case
    M, N, K = shape
    # Seed locally, without changing the application's global RNG state.
    gen = torch.Generator().manual_seed(2026)
    a = (torch.randn((M, K), generator=gen) * 0.25).to(a_dtype).to(flag_gems.device)
    b = (torch.randn((K, N), generator=gen) * 0.25).to(b_dtype)
    if layout == "column":
        b = b.t().contiguous().t()
    b = b.to(flag_gems.device)
    if mode == "scalar":
        sa = torch.tensor([0.75], device=flag_gems.device)
        sb = torch.tensor([1.25], device=flag_gems.device)
    else:
        sa = torch.linspace(0.5, 1.0, M).reshape(M, 1).to(flag_gems.device)
        sb = torch.linspace(1.0, 1.5, N).reshape(1, N).to(flag_gems.device)
    bias = None
    if has_bias:
        bias = (
            (torch.randn((N,), generator=gen) * 0.1).to(out_dtype).to(flag_gems.device)
        )
    return (a, b, sa, sb), dict(
        bias=bias, scale_result=scale_result, out_dtype=out_dtype, use_fast_accum=fast
    )


def comparison_tolerance(dtype):
    if str(dtype).removeprefix("torch.") in FP8_NAMES:
        return 0.125
    return 0.02 if dtype == torch.bfloat16 else 0.002


SHAPES = (
    [(16, 16, 16)]
    if QUICK_MODE
    else [
        (16, 16, 16),
        (32, 32, 32),
        (17, 31, 80),
        (64, 128, 128),
        (128, 128, 128),
        (512, 512, 512),
    ]
)


# Collection only enumerates metadata; device probes run for selected cases.
CASES = [pytest.param(case, id=case_id(case)) for case in cases()]


def require_case(case):
    reason = case_reason(case)
    if reason:
        pytest.skip(f"FP8 capability: {reason}")


def golden(args, kwargs):
    a, b, sa, sb = args
    result = a.cpu().float() @ b.cpu().float()
    result = result * sa.cpu() * sb.cpu()
    if kwargs["bias"] is not None and a.shape[1] > 0:
        result += kwargs["bias"].cpu().float()
    # ATen GPU _scaled_mm currently ignores scale_result (unlike its CPU kernel).
    return result.to(kwargs["out_dtype"] or a.dtype)


def _check(case, shape, layout, fast, use_out):
    require_case(case)
    args, kwargs = make_inputs(case, shape, layout=layout, fast=fast)
    expected = golden(args, kwargs)
    if use_out:
        out = torch.empty(
            (shape[1], shape[0]), dtype=expected.dtype, device=flag_gems.device
        ).t()
        actual = flag_gems.scaled_mm_out(*args, **kwargs, out=out)
        assert actual is out
    else:
        actual = flag_gems.scaled_mm(*args, **kwargs)
    assert actual.dtype == expected.dtype
    tol = comparison_tolerance(expected.dtype)
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=tol, atol=tol
    )


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout,fast", [("column", False), ("row", True)])
def test_scaled_mm(case, shape, layout, fast):
    _check(case, shape, layout, fast, False)


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", SHAPES)
def test_scaled_mm_out(case, shape):
    _check(case, shape, "column", False, True)


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", [(0, 16, 32), (16, 0, 32), (16, 16, 0)])
def test_scaled_mm_empty(case, shape):
    require_case(case)
    args, kwargs = make_inputs(case, shape)
    actual = flag_gems.scaled_mm(*args, **kwargs)
    torch.testing.assert_close(
        actual.cpu().float(), golden(args, kwargs).float(), rtol=0.02, atol=0.02
    )


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case", CASES)
def test_scaled_mm_schema(case):
    require_case(case)
    args, kwargs = make_inputs(case, (16, 16, 32))
    with pytest.raises(RuntimeError, match="Float32"):
        flag_gems.scaled_mm(args[0], args[1], args[2].half(), args[3], **kwargs)
    with pytest.raises(RuntimeError, match="float scalar"):
        flag_gems.scaled_mm(
            *args,
            **{**kwargs, "scale_result": torch.ones(2, device=flag_gems.device)},
        )
    actual = flag_gems.scaled_mm(
        *args,
        **{**kwargs, "scale_result": torch.tensor([2.0], device=flag_gems.device)},
    )
    tol = comparison_tolerance(actual.dtype)
    torch.testing.assert_close(
        actual.cpu().float(), golden(args, kwargs).float(), rtol=tol, atol=tol
    )


@pytest.mark.scaled_mm
def test_scaled_mm_fast_route(monkeypatch):
    import importlib

    module = importlib.import_module("flag_gems.ops.scaled_mm")
    case = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16, "scalar", True)
    reason = case_reason(case)
    if reason:
        pytest.skip(reason)
    calls = []
    original = module._csmm

    def tracked(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(module, "_csmm", tracked)
    for layout in ("column", "row"):
        args, kwargs = make_inputs(case, (32, 32, 32), layout)
        expected_fast = (
            layout == "column"
            and flag_gems.vendor_name == "nvidia"
            and torch.cuda.get_device_capability()[0] == 9
        )
        calls.clear()
        actual = module.scaled_mm(*args, **kwargs)
        assert bool(calls) == expected_fast
        torch.testing.assert_close(
            actual.cpu().float(), golden(args, kwargs).float(), rtol=0.02, atol=0.002
        )


@pytest.mark.scaled_mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads", reason="MThreads descriptor cache"
)
def test_scaled_mm_descriptor_dtype_switch():
    dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    for dtype in dtypes:
        reason = dtype_reason(dtype)
        if reason:
            pytest.skip(reason)
    # Keep shape, strides and output dtype identical while switching input
    # encodings, including switching back to a previously compiled encoding.
    for dtype in (*dtypes, dtypes[0]):
        case = (dtype, dtype, torch.bfloat16, "scalar", True)
        args, kwargs = make_inputs(case, (512, 512, 512))
        expected = golden(args, kwargs)
        actual = flag_gems.scaled_mm(*args, **kwargs)
        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=0.02, atol=0.02
        )

        out = torch.empty_like(actual)
        actual = flag_gems.scaled_mm_out(*args, **kwargs, out=out)
        assert actual is out
        torch.testing.assert_close(
            actual.cpu().float(), expected.float(), rtol=0.02, atol=0.02
        )


# Host-only regression checks: no capability probe may run during collection.
pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]

ASCEND_ERROR = (
    "'hivm.hir.vbrc' op failed to verify that operand at idx 0 should have "
    "element type f8E4M3FN type or f8E5M2 type or 32-bit float"
)

METAX_ERROR = (
    "CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasLtMatmul "
    "with transpose_mat1 t transpose_mat2 n m 16 n 16 k 16"
)


def _load_modules(monkeypatch):
    from dataclasses import asdict, dataclass
    from typing import List, Optional, Tuple

    def unexpected(*args, **kwargs):
        raise AssertionError("device operation during collection")

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("benchmark", __path__=[str(ROOT / "benchmark")])
    names = (
        "float16",
        "bfloat16",
        "float32",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    module("torch", **{name: f"torch.{name}" for name in names}, arange=unexpected)
    language = module("triton.language", constexpr=object())
    module("triton", language=language, jit=lambda fn: fn)
    gems = module("flag_gems", device="cuda", vendor_name="metax")
    module(
        "flag_gems.runtime", torch_device_fn=SimpleNamespace(current_device=unexpected)
    )
    module("tests.conftest", QUICK_MODE=False)
    base = module("benchmark.base", Benchmark=type("Benchmark", (), {}))
    consts = module("benchmark.consts", DEFAULT_METRICS=[])
    # Load the real report dataclasses/renderer without importing device config.
    tree = ast.parse((ROOT / "benchmark/consts.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("BenchmarkMetrics", "BenchmarkResult", "custom_json_encoder")
    ]
    consts.__dict__.update(
        dataclass=dataclass,
        asdict=asdict,
        List=List,
        Optional=Optional,
        Tuple=Tuple,
        torch=sys.modules["torch"],
    )
    exec(compile(tree, "benchmark/consts.py", "exec"), consts.__dict__)
    config = SimpleNamespace(
        query=False,
        list_cases=False,
        preflight_only=False,
        profile_only=False,
        skip_native=False,
        native_baseline_skip_reason=None,
        bench_level=SimpleNamespace(value="core"),
        mode=SimpleNamespace(value="operator"),
    )
    conftest = module("benchmark.conftest", Config=config)
    # Exercise the real baseline fixture, including its teardown ordering.
    tree = ast.parse((ROOT / "benchmark/conftest.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("configure_native_baseline", "_get_native_baseline_skip_reason")
    ]
    conftest.__dict__.update(pytest=pytest, vendor_name="metax")
    exec(compile(tree, "benchmark/conftest.py", "exec"), conftest.__dict__)
    conftest.results, conftest.records = [], []
    tree = ast.parse((ROOT / "benchmark/base.py").read_text())
    benchmark_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Benchmark"
    )
    tree.body = [
        node
        for node in benchmark_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_emit_result"
    ]
    namespace = dict(
        Config=config,
        BenchmarkResult=consts.BenchmarkResult,
        asdict=asdict,
        update_result=lambda name, result: conftest.results.append(result),
        emit_record_logger=conftest.records.append,
    )
    exec(compile(tree, "benchmark/base.py", "exec"), namespace)
    base.Benchmark._emit_result = namespace["_emit_result"]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        value = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, value)
        spec.loader.exec_module(value)
        return value

    accuracy = load("tests.test_scaled_mm", "tests/test_scaled_mm.py")
    benchmark = load("benchmark.test_scaled_mm", "benchmark/test_scaled_mm.py")
    return SimpleNamespace(
        accuracy=accuracy,
        benchmark=benchmark,
        config=config,
        conftest=conftest,
        gems=gems,
    )


@pytest.fixture
def modules(monkeypatch):
    return _load_modules(monkeypatch)


def test_import_enumerates_all_cases_without_device_calls(modules):
    accuracy, benchmark = modules.accuracy, modules.benchmark
    assert len(accuracy.CASES) == 80
    assert len(benchmark.CASES) == 80
    assert {p.values[0] for p in accuracy.CASES} == set(accuracy.cases())
    assert set(benchmark.CASES) == set(accuracy.cases())
    assert len(benchmark.DTYPES) == 4
    assert 2 * len(benchmark.CASES) * len(benchmark.CORE_SHAPES) == 480
    assert all(p.id == accuracy.case_id(p.values[0]) for p in accuracy.CASES)


@pytest.mark.parametrize("dtype", ["float8_e4m3fnuz", "float8_e5m2fnuz"])
@pytest.mark.parametrize("stage", ["input_conversion", "output_conversion"])
def test_ascend_known_fnuz_conversion_is_unsupported(modules, dtype, stage):
    modules.gems.vendor_name = "ascend"
    error = type("MLIRCompilationError", (RuntimeError,), {})(ASCEND_ERROR)
    reason = modules.accuracy._unsupported_reason(
        error, dtype=f"torch.{dtype}", stage=stage
    )
    assert dtype in reason and stage in reason


@pytest.mark.parametrize(
    "vendor,dtype,stage,error_type",
    [
        ("metax", "float8_e4m3fnuz", "input_conversion", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fn", "input_conversion", "MLIRCompilationError"),
        ("ascend", "float8_e5m2", "input_conversion", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fnuz", "dot", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fnuz", "allocation", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fnuz", "input_conversion", "RuntimeError"),
    ],
)
def test_unrelated_compiler_errors_are_not_skipped(
    modules, vendor, dtype, stage, error_type
):
    modules.gems.vendor_name = vendor
    error = type(error_type, (RuntimeError,), {})(ASCEND_ERROR)
    with pytest.raises(RuntimeError) as caught:
        modules.accuracy._unsupported_reason(error, dtype=f"torch.{dtype}", stage=stage)
    assert caught.value is error


def test_metax_known_native_rowwise_rejection(modules):
    reason = modules.accuracy._unsupported_reason(
        RuntimeError(METAX_ERROR),
        dtype="torch.float8_e4m3fn",
        other_dtype="torch.float8_e4m3fn",
        stage="native",
        scaling="rowwise",
    )
    assert "native FP8 rowwise" in reason


@pytest.mark.parametrize(
    "vendor,dtype,other_dtype,stage,scaling,message",
    [
        ("nvidia", "float8_e4m3fn", "float8_e4m3fn", "native", "rowwise", METAX_ERROR),
        ("metax", "float32", "float8_e4m3fn", "native", "rowwise", METAX_ERROR),
        ("metax", "float8_e4m3fn", "float32", "native", "scalar", METAX_ERROR),
        ("metax", "float8_e4m3fn", "float8_e4m3fn", "dot", "scalar", METAX_ERROR),
        ("metax", "float8_e5m2", "float8_e5m2", "native", "scalar", METAX_ERROR),
        ("metax", "float8_e4m3fn", None, "native", "scalar", METAX_ERROR),
        (
            "metax",
            "float8_e4m3fn",
            "float8_e4m3fn",
            "native",
            "rowwise",
            "CUBLAS_STATUS_INVALID_VALUE",
        ),
        (
            "metax",
            "float8_e4m3fn",
            "float8_e4m3fn",
            "native",
            "scalar",
            "Rowwise scaling is not currently supported on your device",
        ),
    ],
)
def test_other_native_errors_are_not_skipped(
    modules, vendor, dtype, other_dtype, stage, scaling, message
):
    modules.gems.vendor_name = vendor
    error = RuntimeError(message)
    with pytest.raises(RuntimeError) as caught:
        modules.accuracy._unsupported_reason(
            error,
            dtype=f"torch.{dtype}",
            other_dtype=f"torch.{other_dtype}" if other_dtype else None,
            stage=stage,
            scaling=scaling,
        )
    assert caught.value is error


@pytest.mark.parametrize(
    "message",
    [
        "out of memory",
        "device-side assert",
        "illegal memory access",
        "No space left on device",
        "unspecified launch failure",
        "unknown compiler regression",
    ],
)
def test_resource_and_unknown_errors_propagate(modules, message):
    # A device fault must take precedence even when the diagnostic also
    # contains the native rejection that would otherwise allow baseline skip.
    diagnostic = message
    if message != "unknown compiler regression":
        diagnostic += "\n" + METAX_ERROR
    with pytest.raises(RuntimeError, match=message):
        modules.accuracy._unsupported_reason(
            RuntimeError(diagnostic),
            dtype="torch.float8_e4m3fn",
            other_dtype="torch.float8_e4m3fn",
            stage="native",
            scaling="scalar",
        )


def _pytest_environment(pytester, monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest(f"""
import ast
import pytest
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import importlib.util
source = Path({str(Path(__file__))!r}).read_text()
tree = ast.parse(source)
def helper_node(node):
    if isinstance(node, ast.FunctionDef):
        return node.name == "_load_modules"
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ("ROOT", "ASCEND_ERROR", "METAX_ERROR")
        for t in node.targets
    )
tree.body = [node for node in tree.body if helper_node(node)]
namespace = dict(globals(), __file__={str(Path(__file__))!r})
exec(compile(tree, "regression_helpers", "exec"), namespace)
helpers = SimpleNamespace(**namespace)
patch = pytest.MonkeyPatch()
modules = helpers._load_modules(patch)
configure_native_baseline = modules.conftest.configure_native_baseline

def pytest_unconfigure(config):
    patch.undo()
""")
    pytester.makeini("""[pytest]
markers =
    scaled_mm
    scaled_mm_out
    unrelated
    skip_native(vendors, reason)
""")


def test_unselected_tests_do_not_probe(pytester, monkeypatch):
    _pytest_environment(pytester, monkeypatch)
    pytester.makepyfile("""
import pytest
from conftest import modules
# Import and collect the real tests, with all device calls prohibited.
for module in (modules.accuracy, modules.benchmark):
    globals().update({k: v for k, v in vars(module).items() if k.startswith("test_scaled_mm")})
@pytest.mark.unrelated
def test_unrelated():
    pass
""")
    # Pytester echoes child output even under -s; show it only on a failed check.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        result = pytester.runpytest_subprocess("-q", "-m", "unrelated")
    diagnostics = result.stdout.str() + result.stderr.str()
    assert result.ret == pytest.ExitCode.OK, diagnostics
    assert result.parseoutcomes() == {"passed": 1, "deselected": 1770}, diagnostics
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        collected = pytester.runpytest_subprocess("--collect-only", "-qq")
    assert collected.ret == pytest.ExitCode.OK, (
        collected.stdout.str() + collected.stderr.str()
    )
    assert "test_unselected_tests_do_not_probe.py: 1771" in collected.stdout.str()
