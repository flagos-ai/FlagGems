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

import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import accuracy_utils as utils


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


# Shapes covering 0-dim, 1-D, multi-D and large flatten spans.
PUT_SHAPES = [
    (),
    (1,),
    (16,),
    (64, 64),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# `put` is dtype-agnostic in ATen: it accepts every integer, floating-point, bool
# and complex dtype. Cover all of them, not just the float set.
PUT_INT_DTYPES = utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]
# `complex32` is excluded on purpose: ATen has no `put_cuda` for ComplexHalf
# (`NotImplementedError: "put_cuda" not implemented for 'ComplexHalf'`), so there
# is no reference to compare against.
PUT_COMPLEX_DTYPES = [torch.complex64, torch.complex128]
PUT_ALL_DTYPES = (
    utils.ALL_FLOAT_DTYPES + PUT_INT_DTYPES + utils.BOOL_TYPES + PUT_COMPLEX_DTYPES
)


def _reference(tensor):
    if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
        return utils.to_reference(tensor, True)
    return utils.to_reference(tensor, False)


def _gems_put(*args, **kwargs):
    result = flag_gems.put(*args, **kwargs)
    inp = args[0]
    assert result.device == inp.device
    assert result.dtype == inp.dtype
    assert result.shape == inp.shape
    if inp.numel():
        assert result.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    return result


def _gems_put_out(*args, **kwargs):
    with flag_gems.use_gems():
        return torch.ops.aten.put.out(*args, **kwargs)


# Accumulating many low-precision values via reordered atomic adds accumulates
# rounding error; use a looser tolerance than the default `1e-4` for these cases.
_ACCUMULATE_ATOL = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-1,
    torch.float32: 1e-4,
}


def accumulate_atol(dtype):
    return _ACCUMULATE_ATOL.get(dtype, 1e-4)


def is_exact(dtype):
    """Integer, bool and complex-integer results must match bit-for-bit."""
    return not dtype.is_floating_point and not dtype.is_complex


def _put_randperm(numel, *, device):
    # Enflame's native int64 randperm aborts before the operator under test.
    # Build the same unique-index fixture on CPU, then preserve its target ABI.
    factory_device = "cpu" if flag_gems.vendor_name == "enflame" else device
    return torch.randperm(numel, device=factory_device).to(device)


def gen_input(shape, dtype, device):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.complex128:
        pytest.skip("Kunlunxin target storage does not retain complex128 dtype")
    if flag_gems.vendor_name in ("ascend", "enflame") and dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device="cpu").to(device)
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(dtype)
    # Keep the magnitudes small so an accumulate over many duplicates stays inside
    # the narrow integer dtypes and the comparison is not about overflow wrap.
    return torch.randint(-8, 8, shape, device=device).to(dtype)


def gen_source(count, dtype, device):
    if flag_gems.vendor_name in ("ascend", "enflame") and dtype.is_complex:
        return torch.randn(count, dtype=dtype, device="cpu").to(device)
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(count, dtype=dtype, device=device, requires_grad=False)
    if dtype == torch.bool:
        return torch.randint(0, 2, (count,), device=device).to(dtype)
    return torch.randint(-8, 8, (count,), device=device).to(dtype)


def gen_index_and_source(inp, dtype, device, accumulate, count=None):
    numel = inp.numel()
    if count is None:
        # Use a subset of positions to exercise partial writes.
        count = max(1, numel // 3)
    if accumulate:
        # Duplicate indices are only meaningful with `accumulate=True`, where
        # atomic-add makes the result order-independent. Draw with replacement so
        # duplicates really occur and the sum has to be right.
        index = torch.randint(0, numel, (count,), dtype=torch.int64, device=device)
    else:
        # With `accumulate=False` and duplicate indices ATen's result is
        # unspecified -- the last writer wins and which one that is depends on
        # thread scheduling (measured: not bit-stable across 200 identical runs).
        # Draw unique indices so the expected answer is well defined.
        index = _put_randperm(numel, device=device)[:count].to(torch.int64)
    source = gen_source(count, dtype, device)
    return index, source


def _put_assert_metadata(actual, expected, dtype):
    target = torch.device(flag_gems.device)
    if target.type != "cpu" and target.index is None:
        target = torch.device(target.type, torch_device_fn.current_device())
    assert actual.device == target
    assert actual.dtype == dtype
    assert actual.shape == expected.shape


def _put_assert_close(actual, expected, dtype, **kwargs):
    _put_assert_metadata(actual, expected, dtype)
    if (
        (flag_gems.vendor_name == "ascend" and expected.device.type == "cpu")
        or flag_gems.vendor_name == "enflame"
        or (flag_gems.vendor_name in ("ascend", "kunlunxin") and actual.is_complex())
    ):
        actual, expected = actual.cpu(), expected.cpu()
    utils.gems_assert_close(actual, expected, dtype, **kwargs)


def _put_assert_exact(actual, expected, dtype):
    _put_assert_metadata(actual, expected, dtype)
    assert expected.dtype == dtype
    if flag_gems.vendor_name == "enflame":
        actual, expected = actual.cpu(), expected.cpu()
    utils.gems_assert_equal(actual, expected)


def _put_same_values(actual, expected):
    _put_assert_metadata(actual, expected, expected.dtype)
    if flag_gems.vendor_name == "enflame" or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and actual.is_complex()
    ):
        actual, expected = actual.cpu(), expected.cpu()
    return torch.equal(actual, expected)


def assert_put_close(res, ref, dtype, accumulate=False):
    if is_exact(dtype):
        _put_assert_exact(res, ref, dtype)
    elif accumulate:
        _put_assert_close(res, ref, dtype, atol=accumulate_atol(dtype))
    else:
        _put_assert_close(res, ref, dtype)


# ---------------------------------------------------------------------------
# put (out-of-place)
# ---------------------------------------------------------------------------
@pytest.mark.put
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put(shape, dtype, accumulate):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = _reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = _gems_put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize("shape", [(16,), (64, 64)])
@pytest.mark.parametrize("dtype", PUT_ALL_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_dtypes(shape, dtype, accumulate):
    # ATen accepts integer, floating-point, bool and complex `put`; check parity
    # for all of them under both accumulate modes.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = _reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = _gems_put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int32, torch.int8, torch.uint8, torch.bool]
)
def test_put_accumulate_duplicates(dtype):
    # Every slot is written many times over: a plain store would silently keep one
    # contribution and drop the rest, so this pins the atomic-add path. The narrow
    # integer dtypes have no `tl.atomic_add` lowering and go through an int32
    # staging buffer; ATen wraps on overflow and so must we.
    numel = 17
    repeats = 512
    inp = gen_input((numel,), dtype, flag_gems.device)
    index = torch.arange(numel, device=flag_gems.device).repeat(repeats)
    source = torch.ones(index.numel(), dtype=dtype, device=flag_gems.device)

    ref_out = torch.put(
        _reference(inp),
        utils.to_reference(index, False),
        _reference(source),
        accumulate=True,
    )
    res_out = _gems_put(inp, index, source, accumulate=True)

    assert_put_close(res_out, ref_out, dtype, accumulate=True)


@pytest.mark.put
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put_negative_index(shape, dtype):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = _reference(inp)

    numel = inp.numel()
    count = max(1, numel // 2)
    # Unique indices so the non-accumulating overwrite is order-independent;
    # shift half of them to the negative range to exercise negative indexing.
    # Enflame lacks the int64 remainder fixture operation, so form the same
    # signed indices on CPU and transfer the finished index tensor.
    index_device = "cpu" if flag_gems.vendor_name == "enflame" else flag_gems.device
    perm = _put_randperm(numel, device=index_device)[:count].to(torch.int64)
    neg_mask = torch.arange(count, device=index_device) % 2 == 0
    index = torch.where(neg_mask, perm - numel, perm).to(flag_gems.device)
    source = gen_source(count, dtype, flag_gems.device)

    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    if flag_gems.vendor_name == "ascend":
        ref_inp, ref_index, ref_source = (
            value.cpu() for value in (ref_inp, ref_index, ref_source)
        )
    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = _gems_put(inp, index, source)

    _put_assert_close(res_out, ref_out, dtype)


@pytest.mark.put
@pytest.mark.parametrize("numel", [1, 2, 3, 5, 6, 7, 8, 64])
@pytest.mark.parametrize("index_at", ["first", "last", "neg_first", "neg_last"])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_index_bounds(numel, index_at, accumulate):
    # The exact ends of the valid range: 0, numel-1, -numel and -1. Each is put in
    # its own call because -numel and 0 alias the same slot (as do -1 and
    # numel-1), which would otherwise make this a duplicate-index case.
    dtype = torch.float32
    value = {"first": 0, "last": numel - 1, "neg_first": -numel, "neg_last": -1}[
        index_at
    ]
    inp = gen_input((numel,), dtype, flag_gems.device)
    index = torch.tensor([value], dtype=torch.int64, device=flag_gems.device)
    source = gen_source(1, dtype, flag_gems.device)

    if flag_gems.vendor_name == "ascend" and index_at in ("neg_first", "neg_last"):
        ref_out = torch.put(
            _reference(inp).cpu(),
            utils.to_reference(index, False).cpu(),
            _reference(source).cpu(),
            accumulate=accumulate,
        )
    else:
        ref_out = torch.put(
            _reference(inp),
            utils.to_reference(index, False),
            _reference(source),
            accumulate=accumulate,
        )
    res_out = _gems_put(inp, index, source, accumulate=accumulate)

    _put_assert_close(res_out, ref_out, dtype)


def _put_child_override(request):
    # Match the public CLI precedence: config first, then command-line overrides.
    # Only forward this operator; unrelated overrides must not be marked unused.
    from flag_gems.cli_override import load_override_config, parse_override_spec

    options = request.config.option
    overrides = {}
    config_file = getattr(options, "override_config", None)
    if config_file:
        overrides.update(load_override_config(config_file))
    for spec in getattr(options, "override", None) or ():
        name, filename, function = parse_override_spec(spec)
        overrides[name] = (filename, function)
    selected = overrides.get("put")
    if selected is None:
        return None
    filename, function = selected
    return {"file": str(Path(filename).resolve()), "function": function}


def _put_callable_binding(function):
    function = inspect.unwrap(function)
    code = getattr(function, "__code__", None)
    filename = getattr(code, "co_filename", None)
    source = Path(filename) if filename else None
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "code_file": filename,
        "first_line": getattr(code, "co_firstlineno", None),
        "source_sha256": (
            hashlib.sha256(source.read_bytes()).hexdigest()
            if source is not None and source.is_file()
            else None
        ),
    }


def _run_put_bounds_child(numel, bad, accumulate, override, ref_cpu=False):
    # A device fault poisons its process. Recreate the *same* injected callable
    # in a fresh interpreter instead of accidentally running the original Gems
    # kernel, and keep failure phase separate from compiler/setup errors.
    code = textwrap.dedent("""
        import hashlib
        import inspect
        import json
        import sys
        from pathlib import Path

        import torch
        import flag_gems
        from flag_gems.dynamic_registry import DynamicOpOverride
        from flag_gems.runtime import torch_device_fn

        args = json.loads(sys.argv[1])
        report = {
            "phase": "setup", "override": args["override"],
            "numel": args["numel"], "bad": args["bad"],
            "accumulate": args["accumulate"], "valid_completed": False,
        }
        from tests import conftest as test_config
        test_config.TO_CPU = args["ref_cpu"]
        from tests import accuracy_utils
        accuracy_utils.TO_CPU = args["ref_cpu"]
        registry = None
        try:
            if args["override"] is not None:
                registry = DynamicOpOverride()
                selected = args["override"]
                if not registry.override_from_file(
                    "put", selected["file"], selected["function"]
                ):
                    raise RuntimeError("Could not recreate the put override")
            actual = inspect.unwrap(flag_gems.put)
            actual_code = getattr(actual, "__code__", None)
            filename = getattr(actual_code, "co_filename", None)
            source_file = Path(filename) if filename else None
            report["binding"] = {
                "module": getattr(actual, "__module__", None),
                "qualname": getattr(actual, "__qualname__", None),
                "code_file": filename,
                "first_line": getattr(actual_code, "co_firstlineno", None),
                "source_sha256": (
                    hashlib.sha256(source_file.read_bytes()).hexdigest()
                    if source_file is not None and source_file.is_file() else None
                ),
            }
            if report["binding"] != args["binding"]:
                raise RuntimeError("The put child did not load its parent's callable")
            report["ref_cpu"] = test_config.TO_CPU
            inp = torch.randn(
                args["numel"], dtype=torch.float32, device=flag_gems.device
            )
            index = torch.tensor(
                [0, args["bad"]], dtype=torch.int64, device=flag_gems.device
            )
            source = torch.randn(2, dtype=torch.float32, device=flag_gems.device)

            def check_host_result(result, target, idx, values, before):
                expected = torch.put(before, idx.cpu(), values.cpu(),
                                     accumulate=args["accumulate"])
                assert result.device == target.device and result.dtype == target.dtype
                assert result.shape == target.shape
                assert result.untyped_storage().data_ptr() != target.untyped_storage().data_ptr()
                torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(target.cpu(), before, rtol=0, atol=0)

            def tensor_abi(value):
                return {"device": str(value.device), "dtype": str(value.dtype),
                        "shape": list(value.shape), "stride": list(value.stride()),
                        "offset": value.storage_offset(), "pointer": value.data_ptr(),
                        "storage_pointer": value.untyped_storage().data_ptr()}

            before_valid_abi = [tensor_abi(value) for value in (inp, index, source)]
            before_valid_inp = inp.cpu().clone()
            before_valid_index = index.cpu().clone()
            before_valid_source = source.cpu().clone()
            actual_indices = before_valid_index.tolist()
            assert actual_indices == [0, args["bad"]]
            assert args["bad"] < -inp.numel() or args["bad"] >= inp.numel()
            report.update(invalid_index_values=actual_indices, actual_oob_verified=True,
                          before_valid_abi=before_valid_abi,
                          before_valid_values={"inp": before_valid_inp.tolist(),
                                              "index": before_valid_index.tolist(),
                                              "source": before_valid_source.tolist()})
            report["phase"] = "valid"
            original_sync = torch_device_fn.synchronize
            sync_code = getattr(inspect.unwrap(original_sync), "__code__", None)
            sync_file = getattr(sync_code, "co_filename", None)
            report["sync_binding"] = {
                "module": getattr(original_sync, "__module__", None),
                "qualname": getattr(original_sync, "__qualname__", None),
                "file": sync_file,
                "sha256": (hashlib.sha256(Path(sync_file).read_bytes()).hexdigest()
                           if sync_file and Path(sync_file).is_file() else None),
            }
            def unexpected_sync(*unused_args, **unused_kwargs):
                raise AssertionError("A valid put candidate must not synchronize")
            torch_device_fn.synchronize = unexpected_sync
            try:
                valid = flag_gems.put(inp, index[:1], source[:1], accumulate=args["accumulate"])
            finally:
                torch_device_fn.synchronize = original_sync
            # This is the test helper's synchronization, after the candidate's
            # no-sync monitor has ended. An asynchronous valid-call fault must
            # not be mistaken for an expected out-of-range failure below.
            original_sync()
            assert torch_device_fn.synchronize is original_sync
            check_host_result(valid, inp, index[:1], source[:1], before_valid_inp)
            torch.testing.assert_close(index.cpu(), before_valid_index, rtol=0, atol=0)
            torch.testing.assert_close(source.cpu(), before_valid_source, rtol=0, atol=0)
            report["valid_ABI_values"] = True
            before_invalid_abi = [tensor_abi(value) for value in (inp, index, source)]
            before_invalid_inp = inp.cpu().clone()
            before_invalid_index = index.cpu().clone()
            before_invalid_source = source.cpu().clone()
            report.update(before_invalid_abi=before_invalid_abi,
                          before_invalid_values={"inp": before_invalid_inp.tolist(),
                                                 "index": before_invalid_index.tolist(),
                                                 "source": before_invalid_source.tolist()})
            report["valid_completed"] = True
            report["phase"] = "invalid"
            flag_gems.put(inp, index, source, accumulate=args["accumulate"])
            torch_device_fn.synchronize()
            report["status"] = "NO_ERROR"
        except BaseException as error:
            report.update(
                status="ERROR", error_type=type(error).__name__, message=str(error),
                runtime_error=isinstance(error, RuntimeError),
            )
            if report["phase"] == "invalid" and isinstance(error, IndexError):
                report["host_index_error"] = True
                try:
                    try:
                        torch.put(before_invalid_inp.clone(), before_invalid_index.clone(),
                                  before_invalid_source.clone(), accumulate=args["accumulate"])
                    except IndexError as cpu_error:
                        report["expected_host_message"] = str(cpu_error)
                        assert type(error) is type(cpu_error)
                        assert str(error) == str(cpu_error)
                    else:
                        raise AssertionError("CPU accepted the invalid put index")
                    assert [tensor_abi(value) for value in (inp, index, source)] == before_invalid_abi
                    torch.testing.assert_close(inp.cpu(), before_invalid_inp, rtol=0, atol=0)
                    torch.testing.assert_close(index.cpu(), before_invalid_index, rtol=0, atol=0)
                    torch.testing.assert_close(source.cpu(), before_invalid_source, rtol=0, atol=0)
                    report["input_index_source_readonly"] = True
                    fresh = torch.zeros_like(inp)
                    before = fresh.cpu()
                    recovered = flag_gems.put(fresh, index[:1], source[:1],
                                              accumulate=args["accumulate"])
                    torch_device_fn.synchronize()
                    check_host_result(recovered, fresh, index[:1], source[:1], before)
                    assert index[:1].numel() > 0
                    torch.testing.assert_close(inp.cpu(), before_invalid_inp, rtol=0, atol=0)
                    torch.testing.assert_close(index.cpu(), before_invalid_index, rtol=0, atol=0)
                    torch.testing.assert_close(source.cpu(), before_invalid_source, rtol=0, atol=0)
                    assert [tensor_abi(value) for value in (inp, index, source)] == before_invalid_abi
                    report.update(recovery_ABI_values=True, recovery_nonempty_control=True,
                                  context_failed=False)
                except BaseException as recovery_error:
                    report.update(recovery_ABI_values=False,
                                  recovery_error_type=type(recovery_error).__name__,
                                  recovery_error=str(recovery_error))
            elif report["phase"] == "invalid":
                try:
                    probe = torch.ones(1, device=flag_gems.device) + 1
                    torch_device_fn.synchronize()
                    report["context_failed"] = False
                except RuntimeError as context_error:
                    report.update(context_failed=True, context_error=str(context_error))
            if registry is not None:
                report["calls"] = registry.call_counts()
            Path(args["report"]).write_text(json.dumps(report))
            raise
        else:
            if registry is not None:
                report["calls"] = registry.call_counts()
            Path(args["report"]).write_text(json.dumps(report))
        """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    with tempfile.TemporaryDirectory(prefix="put-bounds-") as directory:
        report_file = Path(directory) / "result.json"
        payload = dict(
            numel=numel,
            bad=bad,
            accumulate=accumulate,
            override=override,
            report=str(report_file),
            ref_cpu=ref_cpu,
            binding=_put_callable_binding(flag_gems.put),
        )
        child = subprocess.Popen(
            [sys.executable, "-c", code, json.dumps(payload)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )

        def interrupt_wait(_signum, _frame):
            raise KeyboardInterrupt("put bounds child wait terminated")

        previous_sigterm = signal.signal(signal.SIGTERM, interrupt_wait)
        try:
            stdout, stderr = child.communicate(timeout=120)
        except BaseException:
            # This process group belongs only to this child and its compiler
            # descendants; never terminate the parent pytest process group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.communicate()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
        proc = subprocess.CompletedProcess(child.args, child.returncode, stdout, stderr)
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
    return proc, report


def _put_device_fault(message):
    # CUDA appends a generic DSA hint to unrelated runtime errors. Inspect only
    # the actual first error line, not that footer or a compiler traceback.
    first_line = message.splitlines()[0].lower() if message else ""
    return first_line in {
        f"{backend} error: {fault}"
        for backend in ("cuda", "hip")
        for fault in (
            "device-side assert triggered",
            "unspecified launch failure",
            "an illegal instruction was encountered",
        )
    }


def _assert_put_bounds_failure(proc, report):
    assert proc.returncode != 0, (
        "an out-of-range index was silently accepted; "
        f"stdout={proc.stdout!r} stderr={proc.stderr[-300:]!r}"
    )
    assert report.get("phase") == "invalid" and report.get("status") == "ERROR", (
        "the put child failed before its out-of-range call: "
        + repr(report)
        + f" returncode={proc.returncode}, stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )
    if report.get("host_index_error"):
        # A genuine host IndexError must match CPU and leave the context healthy.
        assert report.get("valid_completed") and report.get("valid_ABI_values")
        assert report.get("error_type") == "IndexError"
        assert report.get("host_index_error") is True
        assert report.get("actual_oob_verified") and report.get(
            "input_index_source_readonly"
        )
        assert report.get("recovery_nonempty_control")
        assert report.get("message") == report.get("expected_host_message")
        assert report.get("recovery_ABI_values") is True
        assert report.get("context_failed") is False
        if report.get("override") is not None:
            assert report.get("calls", {}).get("flag_gems.put") == 3
        return
    assert (
        report.get("runtime_error") is True
    ), "expected the target-device RuntimeError, got: " + repr(report)
    assert (
        report.get("context_failed") is True
    ), "expected a target-device fault, not a host/compiler exception: " + repr(report)
    assert _put_device_fault(report.get("context_error", ""))
    assert (
        report["message"].splitlines()[0].lower()
        == report["context_error"].splitlines()[0].lower()
    )
    assert _put_device_fault(report.get("message", "")), (
        "expected a device-side failure, got: " + proc.stderr[-300:]
    )
    if report.get("override") is not None:
        assert report.get("calls", {}).get("flag_gems.put") == 2


@pytest.mark.put
@pytest.mark.skipif(
    flag_gems.vendor_name in {"mthreads", "metax", "iluvatar"},
    reason="native put has no target-device trap for invalid indices",
)
@pytest.mark.parametrize("numel", [1, 8, 1000])
@pytest.mark.parametrize("bad_offset", [0, 5])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_index_out_of_range(numel, bad_offset, negative, accumulate, request):
    # --ref cpu changes value comparison, not the target-device trap contract.
    bad = -numel - 1 - bad_offset if negative else numel + bad_offset
    inp = gen_input((numel,), torch.float32, flag_gems.device)
    original = inp.clone()
    index = torch.zeros(1, dtype=torch.int64, device=inp.device)
    source = gen_source(1, inp.dtype, inp.device)
    reference = torch.put(
        _reference(inp),
        utils.to_reference(index, False),
        _reference(source),
        accumulate=accumulate,
    )
    result = _gems_put(inp, index, source, accumulate=accumulate)
    assert result.device == inp.device
    assert result.dtype == inp.dtype
    assert result.shape == inp.shape
    assert result.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    _put_assert_close(result, reference, inp.dtype)
    torch.testing.assert_close(inp, original, rtol=0, atol=0)

    proc, report = _run_put_bounds_child(
        numel, bad, accumulate, _put_child_override(request), utils.TO_CPU
    )
    request.node.user_properties.append(
        ("put_bounds_child", json.dumps(report, sort_keys=True))
    )
    _assert_put_bounds_failure(proc, report)


@pytest.mark.put
@pytest.mark.parametrize("shape", [(64, 64), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put_non_contiguous(shape, dtype):
    # A transposed (non-contiguous) tensor exercises the multi-dim offset path.
    inp = gen_input(shape, dtype, flag_gems.device).transpose(0, 1)
    ref_inp = _reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate=False)
    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = _gems_put(inp, index, source)

    _put_assert_close(res_out, ref_out, dtype)
    assert res_out.stride() == (
        inp.stride()
        if flag_gems.vendor_name == "kunlunxin" and utils.TO_CPU
        else ref_out.stride()
    )


@pytest.mark.put
@pytest.mark.parametrize(
    "shape, perm",
    [
        ((2, 3, 2, 3, 2), (4, 0, 3, 1, 2)),
        ((2, 3, 2, 3, 2, 3), (5, 0, 2, 1, 4, 3)),
        ((2, 2, 2, 3, 2, 2, 2), (6, 0, 5, 1, 4, 2, 3)),
    ],
)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_non_contiguous_high_rank(shape, perm, accumulate):
    # `put` places no limit on rank, so a permuted 6-D or 7-D `self` must work
    # rather than trip a rank assertion.
    dtype = torch.float32
    inp = gen_input(shape, dtype, flag_gems.device).permute(perm)
    ref_inp = _reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = _gems_put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)
    assert res_out.stride() == (
        inp.stride()
        if flag_gems.vendor_name == "kunlunxin" and utils.TO_CPU
        else ref_out.stride()
    )


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_storage_offset(dtype, accumulate):
    # A view into the middle of a larger buffer. A kernel that assumed a zero
    # storage offset would corrupt the elements on either side of the view, so the
    # untouched remainder of the base tensor is checked too.
    base = gen_input((64,), dtype, flag_gems.device)
    base_snapshot = base.clone()
    inp = base[7:31].reshape(4, 6).t()
    ref_inp = _reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = _gems_put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)
    # `put` is out-of-place: neither the view nor its neighbours may change.
    assert _put_same_values(base, base_snapshot)


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_non_contiguous_index_and_source(dtype, accumulate):
    # `index` and `source` are consumed in row-major flatten order, so a
    # transposed index/source pair must be read through its strides.
    inp = gen_input((64,), dtype, flag_gems.device)
    ref_inp = _reference(inp)

    if accumulate:
        index = (
            torch.randint(0, 64, (36,), dtype=torch.int64, device=flag_gems.device)
            .reshape(6, 6)
            .t()
        )
    else:
        index = (
            _put_randperm(64, device=flag_gems.device)[:36]
            .to(torch.int64)
            .reshape(6, 6)
            .t()
        )
    source = gen_source(36, dtype, flag_gems.device).reshape(6, 6).t()

    ref_out = torch.put(
        ref_inp,
        utils.to_reference(index, False),
        _reference(source),
        accumulate=accumulate,
    )
    res_out = _gems_put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.complex64])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_empty_index(dtype, accumulate):
    # An empty index/source pair leaves `self` untouched.
    inp = gen_input((16,), dtype, flag_gems.device)
    index = torch.tensor([], dtype=torch.int64, device=flag_gems.device)
    source = torch.tensor([], dtype=dtype, device=flag_gems.device)

    ref_out = torch.put(
        _reference(inp),
        utils.to_reference(index, False),
        _reference(source),
        accumulate=accumulate,
    )
    res_out = _gems_put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_put_zero_sized_self(dtype):
    # An empty index into an empty `self` is a no-op; a non-empty one is an error.
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    empty_index = torch.tensor([], dtype=torch.int64, device=flag_gems.device)
    empty_source = torch.tensor([], dtype=dtype, device=flag_gems.device)

    ref_out = torch.put(
        _reference(inp),
        utils.to_reference(empty_index, False),
        _reference(empty_source),
    )
    res_out = _gems_put(inp, empty_index, empty_source)
    assert res_out.device == inp.device
    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    _put_assert_exact(res_out, ref_out.to(dtype), dtype)

    with pytest.raises(IndexError):
        _gems_put(
            inp,
            torch.tensor([0], dtype=torch.int64, device=flag_gems.device),
            gen_source(1, dtype, flag_gems.device),
        )


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.complex64])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
def test_put_does_not_mutate_self(dtype, accumulate, layout):
    # `put` is out-of-place for both dense inputs and strided views.
    if layout == "contiguous":
        inp = gen_input((64,), dtype, flag_gems.device)
    else:
        inp = gen_input((8, 12), dtype, flag_gems.device).t()
    snapshot = inp.clone()
    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    _gems_put(inp, index, source, accumulate=accumulate)
    assert _put_same_values(inp, snapshot)


@pytest.mark.put
def test_put_device_mismatch():
    # ATen requires self/index/source to share a device and refuses to move them.
    inp = gen_input((16,), torch.float32, flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = gen_source(1, torch.float32, flag_gems.device)

    with pytest.raises(RuntimeError):
        _gems_put(inp, index.cpu(), source)
    with pytest.raises(RuntimeError):
        _gems_put(inp, index, source.cpu())


@pytest.mark.put
def test_put_invalid_dtypes():
    inp = gen_input((16,), torch.float32, flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)

    # index must be int64
    with pytest.raises(RuntimeError):
        _gems_put(
            inp,
            index.to(torch.int32),
            gen_source(1, torch.float32, flag_gems.device),
        )
    # source must share self's dtype
    mismatch_dtype = (
        torch.int32 if flag_gems.vendor_name == "kunlunxin" else torch.float64
    )
    mismatch_source = gen_source(1, mismatch_dtype, flag_gems.device)
    assert mismatch_source.dtype != inp.dtype
    with pytest.raises(RuntimeError):
        _gems_put(inp, index, mismatch_source)
    # index and source must have the same number of elements
    with pytest.raises(IndexError):
        _gems_put(
            inp,
            torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device),
            gen_source(1, torch.float32, flag_gems.device),
        )


@pytest.mark.put
@pytest.mark.parametrize("shape", [(64,), (64, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put_index_source_diff_shapes(shape, dtype):
    # index and source may have different shapes as long as they share numel.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = _reference(inp)

    count = max(1, inp.numel() // 2)
    # Unique indices for a deterministic, order-independent overwrite.
    index = _put_randperm(inp.numel(), device=flag_gems.device)[:count].to(torch.int64)
    # Reshape source to a different shape with the same number of elements.
    source = gen_source(count, dtype, flag_gems.device).reshape(
        (count // 4, 4) if count % 4 == 0 else (count,)
    )

    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = _gems_put(inp, index, source)

    _put_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# put.out
# ---------------------------------------------------------------------------
@pytest.mark.put_out
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_out(shape, dtype, accumulate):
    # `torch.put` has no `out=` overload, so the reference is `torch.ops.aten.put.out`,
    # which is available on CPU as well as CUDA and so works under --ref=cpu too.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = _reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    out = torch.empty_like(inp)
    pointer = out.data_ptr()
    ref_out = torch.empty_like(ref_inp)
    ref_out = torch.ops.aten.put.out(
        ref_inp, ref_index, ref_source, accumulate, out=ref_out
    )
    res_out = _gems_put_out(inp, index, source, accumulate, out=out)

    assert res_out is out
    assert out.data_ptr() == pointer
    assert_put_close(res_out, ref_out, dtype, accumulate)
    assert_put_close(out, ref_out, dtype, accumulate)

    resized_out = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    resized_result = _gems_put_out(inp, index, source, accumulate, out=resized_out)
    assert resized_result is resized_out
    assert resized_out.shape == inp.shape
    assert resized_out.dtype == inp.dtype
    assert_put_close(resized_out, ref_out, dtype, accumulate)


@pytest.mark.put_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_out_non_contiguous_out(dtype, accumulate):
    # ATen writes into `out` through its own strides and leaves them alone.
    inp = gen_input((12, 8), dtype, flag_gems.device)
    ref_inp = _reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index, False)
    ref_source = _reference(source)

    out = torch.empty((8, 12), dtype=dtype, device=flag_gems.device).t()
    pointer = out.data_ptr()
    stride = out.stride()
    ref_out = torch.empty_like(_reference(out))
    ref_out = torch.ops.aten.put.out(
        ref_inp, ref_index, ref_source, accumulate, out=ref_out
    )
    res_out = _gems_put_out(inp, index, source, accumulate, out=out)

    assert_put_close(res_out, ref_out, dtype, accumulate)
    assert res_out is out
    assert out.data_ptr() == pointer
    assert out.stride() == stride


@pytest.mark.put_out
def test_put_out_invalid():
    inp = gen_input((16,), torch.float32, flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = gen_source(1, torch.float32, flag_gems.device)

    # `out` must share self's dtype and device.
    with pytest.raises(RuntimeError):
        _gems_put_out(
            inp,
            index,
            source,
            False,
            out=torch.empty(16, dtype=torch.float64, device=flag_gems.device),
        )
    with pytest.raises(RuntimeError):
        _gems_put_out(inp, index, source, False, out=torch.empty(16))


@pytest.mark.put_out
@pytest.mark.parametrize(
    "bad_arg",
    ["index_dtype", "source_dtype", "numel", "out_dtype"],
)
def test_put_out_error_leaves_out_untouched(bad_arg):
    """A rejected call must not write anything into ``out``.

    Native validates before it materializes anything, so a caller-provided
    buffer keeps its data on the error path. The ``out=`` overload used to
    copy ``self`` into ``out`` first, clobbering it before the dtype/numel
    validation ran.
    """
    shape = (3, 4)
    inp = gen_input(shape, torch.float32, flag_gems.device)
    index = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    source = gen_source(2, torch.float32, flag_gems.device)
    out_dtype = torch.float32

    if bad_arg == "index_dtype":
        index = index.to(torch.int32)
    elif bad_arg == "source_dtype":
        source = source.to(torch.float64)
    elif bad_arg == "numel":
        source = source[:1]
    else:
        out_dtype = torch.float64

    sentinel = 99.0
    ref_out = torch.full(shape, sentinel, dtype=out_dtype, device=flag_gems.device)
    ref_exc = None
    try:
        torch.ops.aten.put.out(inp, index, source, False, out=ref_out)
    except (RuntimeError, IndexError) as e:
        ref_exc = e

    res_out = torch.full(shape, sentinel, dtype=out_dtype, device=flag_gems.device)
    with pytest.raises(type(ref_exc)) as exc_info:
        _gems_put_out(inp, index, source, False, out=res_out)

    assert str(exc_info.value) == str(ref_exc)
    assert torch.equal(
        res_out, torch.full(shape, sentinel, dtype=out_dtype, device=flag_gems.device)
    ), "out was mutated on the error path"
    assert torch.equal(res_out, ref_out)


@pytest.mark.put
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend native reference does not support ComplexHalf clone/put",
)
def test_put_complex32_rejected():
    # Native CUDA `put` has no ComplexHalf kernel. `complex32` used to be routed
    # through the FP16 real-view path, silently producing a result where ATen
    # raises; the two must agree on the exception type and message.
    # These backends store ComplexHalf, but their device fill kernel lacks it.
    factory_device = (
        "cpu"
        if flag_gems.vendor_name in ("ascend", "kunlunxin", "mthreads")
        else flag_gems.device
    )
    inp = torch.zeros(4, dtype=torch.complex32, device=factory_device).to(
        flag_gems.device
    )
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.complex32, device=factory_device).to(
        flag_gems.device
    )

    # Backends/PyTorch versions use RuntimeError or its NotImplementedError
    # subclass. Keep the native rejection family and exact message.
    with pytest.raises(RuntimeError, match="not implemented.*ComplexHalf") as ref_err:
        torch.put(inp.clone(), index, source)
    with pytest.raises(type(ref_err.value)) as res_err:
        _gems_put(inp, index, source)

    assert str(ref_err.value) == str(res_err.value)
