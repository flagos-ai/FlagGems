import inspect
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import accuracy_utils as utils

# Shapes covering 1-D, multi-D and large flatten spans. Mirrors `test_put.py`.
PUT_SHAPES = [
    (),
    (1,),
    (16,),
    (64, 64),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# Accumulating many low-precision values via reordered atomic adds accumulates
# rounding error; use a looser tolerance than the default `1e-4` for these cases.
_ACCUMULATE_ATOL = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-1,
    torch.float32: 1e-4,
}


def _assert_put_equal(actual, expected, dtype):
    # Preserve the target contract before observing exact values on CPU.
    target = torch.device(flag_gems.device)
    if target.type != "cpu" and target.index is None:
        target = torch.device(target.type, torch_device_fn.current_device())
    assert actual.device == target
    assert actual.dtype == dtype
    assert actual.shape == expected.shape

    def observe_cpu(value):
        if (
            flag_gems.vendor_name in ("ascend", "enflame")
            and value.ndim > 8
            and value.device.type != "cpu"
        ):
            # These strided copy kernels reject the original rank-nine views.
            # Copy unchanged physical storage as 1-D and reconstruct its CPU
            # logical view with every original stride and storage offset.
            storage_numel = value.untyped_storage().nbytes() // value.element_size()
            physical = value.as_strided((storage_numel,), (1,), 0).cpu()
            return physical.as_strided(
                value.shape, value.stride(), value.storage_offset()
            )
        return value.cpu()

    utils.gems_assert_equal(observe_cpu(actual), observe_cpu(expected).to(dtype))


def _put_assert_close(actual, expected, dtype, **kwargs):
    target = torch.device(flag_gems.device)
    if target.type != "cpu" and target.index is None:
        target = torch.device(target.type, torch_device_fn.current_device())
    assert actual.device == target
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    if (
        (flag_gems.vendor_name == "ascend" and expected.device.type == "cpu")
        or flag_gems.vendor_name == "enflame"
        or (flag_gems.vendor_name in ("ascend", "kunlunxin") and actual.is_complex())
    ):
        actual, expected = actual.cpu(), expected.cpu()
    utils.gems_assert_close(actual, expected, dtype, **kwargs)


def accumulate_atol(dtype):
    return _ACCUMULATE_ATOL.get(dtype, 1e-4)


def _put_randperm(numel, *, device):
    # Enflame's native int64 randperm aborts before the operator under test.
    # Build the same unique-index fixture on CPU, then preserve its target ABI.
    factory_device = "cpu" if flag_gems.vendor_name == "enflame" else device
    return torch.randperm(numel, device=factory_device).to(device)


def gen_input(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device, requires_grad=False)


def gen_index_and_source(inp, dtype, device, accumulate, count=None):
    import numpy as np

    numel = inp.numel()
    if count is None:
        # Use a subset of positions to exercise partial writes.
        count = max(1, numel // 3)
    # When not accumulating, repeated indices race (PyTorch keeps the first
    # write in order while our kernel is unordered), so only allow repeats when
    # accumulating, where atomic-add makes the result order-independent.
    replace = bool(accumulate)
    index = np.random.choice(np.arange(numel), size=count, replace=replace)
    index = torch.tensor(index, dtype=torch.int64, device=device)
    source = torch.randn(count, dtype=dtype, device=device, requires_grad=False)
    return index, source


# ---------------------------------------------------------------------------
# put_ (in-place)
# ---------------------------------------------------------------------------
@pytest.mark.put_
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_(shape, dtype, accumulate):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source, True)

    ref_inp.put_(ref_index, ref_source, accumulate=accumulate)
    res = flag_gems.put_(inp, index, source, accumulate=accumulate)

    # `put_` is in-place: the return value must be `self` itself, not a copy.
    assert res is inp

    if accumulate:
        _put_assert_close(inp, ref_inp, dtype, atol=accumulate_atol(dtype))
    else:
        _put_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__negative_index(shape, dtype):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

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
    source = torch.randn(count, dtype=dtype, device=flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source, True)

    if flag_gems.vendor_name == "ascend":
        ref_inp, ref_index, ref_source = (
            value.cpu() for value in (ref_inp, ref_index, ref_source)
        )
    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    _put_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("shape", [(64, 64), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__non_contiguous(shape, dtype):
    # A transposed (non-contiguous) tensor exercises the multi-dim offset path.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp.clone(), True)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate=False)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source, True)

    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    _put_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16, torch.bool])
@pytest.mark.parametrize(
    "layout",
    ["transpose", "slice", "narrow_perm", "row_slice"],
)
def test_put__non_contiguous_narrow_accumulate(dtype, layout):
    """Non-contiguous narrow int/bool under ``accumulate=True``.

    These dtypes accumulate through an int32 staging buffer because Triton has
    no atomic for them. ``put_`` indexes ``self`` by its row-major *logical*
    flattening, so both the widen (reading ``self``) and the narrow (writing it
    back) must decode that flat offset against the real strides -- staging a
    strided ``self`` as if it were contiguous corrupts exactly the indexed
    elements.
    """
    utils.init_seed(0)
    base = _gen_typed((16, 24), dtype, flag_gems.device)
    inp = {
        "transpose": lambda t: t.t(),
        "slice": lambda t: t[:, ::2],
        "narrow_perm": lambda t: t[3:11, 2:18].t(),
        "row_slice": lambda t: t[::2],
    }[layout](base)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp.clone(), False)

    numel = inp.numel()
    index = torch.randint(
        0, numel, (numel * 3,), dtype=torch.int64, device=flag_gems.device
    )
    source = _gen_typed((index.numel(),), dtype, flag_gems.device)

    ref_inp.put_(
        utils.to_reference(index), utils.to_reference(source, False), accumulate=True
    )
    res = flag_gems.put_(inp, index, source, accumulate=True)

    assert res is inp
    _assert_put_equal(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("shape", [(64,), (64, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__index_source_diff_shapes(shape, dtype):
    # index and source may have different shapes as long as they share numel.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    count = max(1, inp.numel() // 2)
    # Unique indices for a deterministic, order-independent overwrite.
    index = _put_randperm(inp.numel(), device=flag_gems.device)[:count].to(torch.int64)
    # Reshape source to a different shape with the same number of elements.
    source = torch.randn(
        (count // 4, 4) if count >= 4 else (count,),
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source, True)

    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    _put_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_put__integer(dtype):
    inp = torch.zeros(100, dtype=dtype, device=flag_gems.device)
    # Integer put is exact; avoid a lossy floating-point reference conversion.
    ref_inp = utils.to_reference(inp.clone())

    import numpy as np

    n = inp.numel()
    idx_np = np.random.choice(np.arange(n), size=(50,), replace=False)
    index = torch.tensor(idx_np, dtype=torch.int64, device=flag_gems.device)
    source = torch.randint(0, 1000, (50,), dtype=dtype, device=flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)
    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    _put_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__accumulate_duplicates(dtype):
    # Exercise the accumulate path with heavy index duplication, where the
    # kernel accumulates in the tensor's own (low-precision) dtype.
    utils.init_seed(0)
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    import numpy as np

    n = inp.numel()
    idx_np = np.random.choice(np.arange(n), size=(n * 8,), replace=True)
    index = torch.tensor(idx_np, dtype=torch.int64, device=flag_gems.device)
    source = torch.randn(n * 8, dtype=dtype, device=flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source, True)
    ref_inp.put_(ref_index, ref_source, accumulate=True)
    flag_gems.put_(inp, index, source, accumulate=True)

    _put_assert_close(inp, ref_inp, dtype, atol=accumulate_atol(dtype))


# ---------------------------------------------------------------------------
# dtype coverage: integer, bool and complex, with and without accumulate.
# ---------------------------------------------------------------------------
# `accumulate=True` has no Triton atomic for int8/uint8/int16/bool, so those
# dtypes take a staged int32 path; the parametrisation below covers both sides.
_PUT_ALL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.BOOL_TYPES
    + [torch.complex64, torch.complex128]
)


def _gen_typed(shape, dtype, device):
    """Random tensor of any dtype `put_` accepts."""
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    if dtype.is_complex:
        if flag_gems.vendor_name == "kunlunxin" and dtype == torch.complex128:
            pytest.skip("Kunlunxin target storage does not retain complex128 dtype")
        factory_device = (
            "cpu" if flag_gems.vendor_name in ("enflame", "kunlunxin") else device
        )
        real = torch.randn(shape, device=factory_device)
        imag = torch.randn(shape, device=factory_device)
        return (real + 1j * imag).to(dtype).to(device)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    info = torch.iinfo(dtype)
    lo = max(info.min, -100)
    hi = min(info.max, 100)
    return torch.randint(lo, hi, shape, dtype=dtype, device=device)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", _PUT_ALL_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__dtypes(dtype, accumulate):
    utils.init_seed(0)
    numel = 64
    inp = _gen_typed((numel,), dtype, flag_gems.device)
    upcast = dtype.is_floating_point or dtype.is_complex
    ref_inp = utils.to_reference(inp.clone(), upcast)

    # Duplicate indices only when accumulating: with `accumulate=False` a
    # repeated index is last-writer-wins and ATen does not specify the winner.
    count = numel * 4 if accumulate else numel // 2
    if accumulate:
        index = torch.randint(
            0, numel, (count,), dtype=torch.int64, device=flag_gems.device
        )
    else:
        index = _put_randperm(numel, device=flag_gems.device)[:count].to(torch.int64)
    source = _gen_typed((count,), dtype, flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source, upcast)

    ref_inp.put_(ref_index, ref_source, accumulate=accumulate)
    res = flag_gems.put_(inp, index, source, accumulate=accumulate)
    assert res is inp

    # Integer and bool results must match bit-exactly. Float and complex
    # accumulation goes through atomics, so the summation order varies between
    # runs and the low bits legitimately differ from the reference.
    if accumulate and (dtype.is_floating_point or dtype.is_complex):
        _put_assert_close(inp, ref_inp, dtype, atol=accumulate_atol(dtype))
    else:
        _assert_put_equal(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16])
def test_put__narrow_int_accumulate_wraparound(dtype):
    # The staged-int32 accumulate path must wrap exactly like native's integer
    # add, so start near the dtype's maximum and overflow it.
    info = torch.iinfo(dtype)
    for init in [0, info.max - 3, info.min + 3]:
        inp = torch.full((4,), init, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp.clone(), False)

        n = 20
        index = torch.zeros(n, dtype=torch.int64, device=flag_gems.device)
        source = torch.ones(n, dtype=dtype, device=flag_gems.device)

        ref_inp.put_(
            utils.to_reference(index),
            utils.to_reference(source, False),
            accumulate=True,
        )
        flag_gems.put_(inp, index, source, accumulate=True)

        _assert_put_equal(inp, ref_inp, dtype)


@pytest.mark.put_
def test_put__bool_accumulate_is_logical_or():
    # Native `accumulate=True` on bool behaves as a logical OR: any True source
    # lands as True and an already-True slot stays True.
    for init in [False, True]:
        for k in [1, 2, 3, 257]:
            inp = torch.zeros(2, dtype=torch.bool, device=flag_gems.device)
            inp[0] = init
            ref_inp = utils.to_reference(inp.clone(), False)

            index = torch.zeros(k, dtype=torch.int64, device=flag_gems.device)
            source = torch.ones(k, dtype=torch.bool, device=flag_gems.device)

            ref_inp.put_(
                utils.to_reference(index),
                utils.to_reference(source, False),
                accumulate=True,
            )
            flag_gems.put_(inp, index, source, accumulate=True)

            _assert_put_equal(inp, ref_inp, torch.bool)


# ---------------------------------------------------------------------------
# Arbitrary rank: `put_` addresses the flattened tensor, so rank is unbounded.
# ---------------------------------------------------------------------------
@pytest.mark.put_
@pytest.mark.parametrize("rank", [1, 2, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("contiguous", [True, False])
def test_put__high_rank(rank, contiguous):
    if flag_gems.vendor_name == "ascend" and rank > 8 and not utils.TO_CPU:
        pytest.skip("Ascend native put_ reference accepts at most eight dimensions")
    utils.init_seed(0)
    shape = tuple(range(2, 2 + rank))
    # These native normal factories accept at most eight dimensions. Preserve
    # the original rank-nine target fixture via a CPU-created tensor and copy.
    factory_device = (
        "cpu"
        if flag_gems.vendor_name in ("ascend", "enflame") and rank > 8
        else flag_gems.device
    )
    factory_inp = torch.randn(shape, dtype=torch.float32, device=factory_device)
    inp = factory_inp.to(flag_gems.device)
    if not contiguous:
        # A fully reversed permutation is the worst case for dim collapsing:
        # no two dimensions are memory-adjacent, so the rank survives intact.
        inp = inp.permute(*reversed(range(rank)))
    reference_input = inp
    if flag_gems.vendor_name in ("ascend", "enflame") and rank > 8 and utils.TO_CPU:
        reference_input = factory_inp
        if not contiguous:
            reference_input = reference_input.permute(*reversed(range(rank)))
    reference_input = utils.to_reference(reference_input, True)
    if flag_gems.vendor_name == "enflame" and rank > 8 and not utils.TO_CPU:
        # Native put_ accepts this view; its preparatory strided clone does not.
        storage_numel = (
            reference_input.untyped_storage().nbytes() // reference_input.element_size()
        )
        physical = reference_input.as_strided((storage_numel,), (1,), 0).clone()
        ref_inp = physical.as_strided(
            reference_input.shape,
            reference_input.stride(),
            reference_input.storage_offset(),
        )
    else:
        ref_inp = reference_input.clone()

    numel = inp.numel()
    count = max(1, numel // 3)
    index = _put_randperm(numel, device=flag_gems.device)[:count].to(torch.int64)
    source = torch.randn(count, dtype=torch.float32, device=flag_gems.device)

    ref_inp.put_(utils.to_reference(index), utils.to_reference(source, True))
    res = flag_gems.put_(inp, index, source)
    assert res is inp

    _assert_put_equal(inp, ref_inp, torch.float32)


# ---------------------------------------------------------------------------
# Error semantics: these must match the exception *type* torch raises.
# ---------------------------------------------------------------------------
@pytest.mark.put_
def test_put__empty_self_raises():
    # Non-empty indices into a zero-element tensor: no valid index exists.
    inp = torch.zeros(0, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        torch.zeros(0, dtype=torch.float32).put_(index.cpu(), source.cpu())
    with pytest.raises(IndexError):
        flag_gems.put_(inp, index, source)


@pytest.mark.put_
@pytest.mark.parametrize("self_numel", [0, 4])
def test_put__empty_index_is_noop(self_numel):
    # An empty index is valid even on an empty tensor, and writes nothing.
    inp = torch.randn(self_numel, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    index = torch.zeros(0, dtype=torch.int64, device=flag_gems.device)
    source = torch.zeros(0, dtype=torch.float32, device=flag_gems.device)

    ref_inp.put_(utils.to_reference(index), utils.to_reference(source, True))
    res = flag_gems.put_(inp, index, source)
    assert res is inp

    _assert_put_equal(inp, ref_inp, torch.float32)


@pytest.mark.put_
@pytest.mark.parametrize("bad", [4, 5, 100, -5, -100])
def test_put__index_out_of_range(bad, request):
    if flag_gems.vendor_name in {"mthreads", "hygon", "metax", "iluvatar"}:
        pytest.skip("native put_ does not enforce device-side index bounds")

    # A real valid-boundary control also exercises the currently injected root
    # in this process. The isolated negative call below must use that same root.
    inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([-1 if bad < 0 else 0], device=inp.device)
    source = torch.ones(1, dtype=inp.dtype, device=inp.device)
    expected = torch.zeros(4, dtype=inp.dtype)
    expected.put_(index.cpu(), source.cpu())
    original_ptr = inp.data_ptr()
    result = flag_gems.put_(inp, index, source)
    assert result is inp
    assert result.data_ptr() == original_ptr
    _assert_put_equal(result, expected, torch.float32)

    # ATen's CUDA `put_` enforces `-numel <= index < numel` with a *device-side*
    # assert, and our kernel mirrors it with an unconditional device trap rather
    # than `tl.device_assert` (which is compiled out unless TRITON_DEBUG=1).
    # A tripped device assert poisons the CUDA context for the whole process, so
    # the failure is exercised in an isolated subprocess.
    from flag_gems.cli_override import load_override_config, parse_override_spec

    overrides = {}
    options = request.config.option
    if getattr(options, "override_config", None):
        overrides.update(load_override_config(options.override_config))
    for spec in getattr(options, "override", None) or ():
        name, filename, function = parse_override_spec(spec)
        overrides[name] = (filename, function)
    selected = overrides.get("put_")
    if selected is not None:
        selected = (str(Path(selected[0]).resolve()), selected[1])

    def binding(function):
        function = inspect.unwrap(function)
        code = getattr(function, "__code__", None)
        return {
            "module": getattr(function, "__module__", None),
            "qualname": getattr(function, "__qualname__", None),
            "file": str(Path(code.co_filename).resolve()) if code else None,
        }

    payload = {
        "bad": bad,
        "override": selected,
        "ref_cpu": utils.TO_CPU,
        "binding": binding(flag_gems.put_),
    }
    code = textwrap.dedent("""
        import hashlib
        import inspect
        import json
        import sys
        import traceback
        from pathlib import Path

        import torch
        import flag_gems
        from flag_gems.dynamic_registry import DynamicOpOverride
        from flag_gems.runtime import torch_device_fn
        from tests import conftest as cfg

        payload = json.loads(sys.argv[1])
        cfg.TO_CPU = payload["ref_cpu"]
        from tests import accuracy_utils as child_utils
        child_utils.TO_CPU = payload["ref_cpu"]
        registry = DynamicOpOverride()
        if payload["override"] is not None:
            filename, function = payload["override"]
            assert registry.override_from_file("put_", filename, function)

        function = inspect.unwrap(flag_gems.put_)
        source_code = getattr(function, "__code__", None)
        actual_binding = {
            "module": getattr(function, "__module__", None),
            "qualname": getattr(function, "__qualname__", None),
            "file": (
                str(Path(source_code.co_filename).resolve()) if source_code else None
            ),
        }
        assert actual_binding == payload["binding"]
        report = {
            "bad": payload["bad"], "binding": actual_binding,
            "ref_cpu": child_utils.TO_CPU, "valid_completed": False,
            "valid_call_returned": False,
        }
        def device_fault(error):
            if not isinstance(error, RuntimeError):
                return False
            first = str(error).partition(chr(10))[0].strip().lower()
            return first in {
                prefix + fault
                for prefix in ("cuda error: ", "hip error: ")
                for fault in (
                    "device-side assert triggered",
                    "unspecified launch failure",
                    "an illegal instruction was encountered",
                )
            }
        inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
        index = torch.tensor(
            [payload["bad"]], dtype=torch.int64, device=flag_gems.device
        )
        source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

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
        assert actual_indices == [payload["bad"]]
        assert payload["bad"] < -inp.numel() or payload["bad"] >= inp.numel()
        report.update(invalid_index_values=actual_indices, actual_oob_verified=True,
                      before_valid_abi=before_valid_abi,
                          before_valid_values={"inp": before_valid_inp.tolist(),
                                              "index": before_valid_index.tolist(),
                                              "source": before_valid_source.tolist()})
        # Preserve the existing nonempty success/no-sync control on these backends.
        # This selects control inputs only; actual errors select the failure lane.
        nonempty_valid_control = flag_gems.vendor_name in ("enflame", "kunlunxin")
        valid_index = (torch.tensor([-1 if payload["bad"] < 0 else 0],
                                    dtype=torch.int64, device=inp.device)
                       if nonempty_valid_control else index[:0])
        valid_source = source if nonempty_valid_control else source[:0]
        valid_pointer = inp.data_ptr()
        report["valid_control_numel"] = valid_index.numel()
        # A valid index must not synchronize on the success path.
        synchronize = torch_device_fn.synchronize
        sync_code = getattr(inspect.unwrap(synchronize), "__code__", None)
        sync_file = getattr(sync_code, "co_filename", None)
        report["sync_binding"] = {
            "module": getattr(synchronize, "__module__", None),
            "qualname": getattr(synchronize, "__qualname__", None),
            "file": sync_file,
            "sha256": (hashlib.sha256(Path(sync_file).read_bytes()).hexdigest()
                       if sync_file and Path(sync_file).is_file() else None),
        }
        def unexpected_valid_sync(*args, **kwargs):
            raise AssertionError("put_ synchronized its valid call")
        report["phase"] = "valid"
        try:
            torch_device_fn.synchronize = unexpected_valid_sync
            try:
                valid = flag_gems.put_(inp, valid_index, valid_source)
                report["valid_call_returned"] = True
            finally:
                torch_device_fn.synchronize = synchronize
            assert valid is inp
            assert valid.device == inp.device and valid.dtype == inp.dtype
            assert valid.shape == inp.shape
            # Complete the valid operation outside the candidate, so a delayed
            # valid-call GPU failure cannot masquerade as the bounds failure.
            synchronize()
            assert torch_device_fn.synchronize is synchronize
            assert valid.data_ptr() == valid_pointer
            expected = before_valid_inp.clone().put_(valid_index.cpu(), valid_source.cpu())
            torch.testing.assert_close(inp.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(index.cpu(), before_valid_index, rtol=0, atol=0)
            torch.testing.assert_close(source.cpu(), before_valid_source, rtol=0, atol=0)
            before_invalid_abi = [tensor_abi(value) for value in (inp, index, source)]
            before_invalid_inp = inp.cpu().clone()
            before_invalid_index = index.cpu().clone()
            before_invalid_source = source.cpu().clone()
            report.update(before_invalid_abi=before_invalid_abi,
                          before_invalid_values={"inp": before_invalid_inp.tolist(),
                                                 "index": before_invalid_index.tolist(),
                                                 "source": before_invalid_source.tolist()})
            report["valid_ABI_values"] = True
        except Exception as error:
            report.update(
                error_type=type(error).__name__, error=str(error),
                device_assert=False, candidate_calls=registry.call_counts(),
            )
            print("PUT_INPLACE_BOUNDS_RESULT:" + json.dumps(report), flush=True)
            traceback.print_exception(type(error), error, error.__traceback__)
            sys.exit(2)
        report["valid_completed"] = True
        report["phase"] = "invalid"
        try:
            flag_gems.put_(inp, index, source)
            torch_device_fn.synchronize()
        except Exception as error:
            report.update(error_type=type(error).__name__, error=str(error))
            report["error_first_line"] = str(error).partition(chr(10))[0]
            traceback.print_exception(type(error), error, error.__traceback__)
            report["context_poisoned"] = False
            if isinstance(error, IndexError):
                report["context_poisoned"] = None
                report["host_index_error"] = True
                try:
                    try:
                        before_invalid_inp.clone().put_(before_invalid_index.clone(), before_invalid_source.clone())
                    except IndexError as cpu_error:
                        report["expected_host_message"] = str(cpu_error)
                        assert type(error) is type(cpu_error)
                        assert str(error) == str(cpu_error)
                    else:
                        raise AssertionError("CPU accepted the invalid put_ index")
                    assert [tensor_abi(value) for value in (inp, index, source)] == before_invalid_abi
                    torch.testing.assert_close(inp.cpu(), before_invalid_inp, rtol=0, atol=0)
                    torch.testing.assert_close(index.cpu(), before_invalid_index, rtol=0, atol=0)
                    torch.testing.assert_close(source.cpu(), before_invalid_source, rtol=0, atol=0)
                    report["input_index_source_readonly"] = True
                    fresh = torch.zeros_like(inp)
                    pointer = fresh.data_ptr()
                    recovery_index = torch.tensor([-1 if payload["bad"] < 0 else 0],
                                                  dtype=torch.int64, device=inp.device)
                    assert recovery_index.numel() > 0
                    recovered = flag_gems.put_(fresh, recovery_index, source)
                    synchronize()
                    assert recovered is fresh and recovered.data_ptr() == pointer
                    assert recovered.device == inp.device and recovered.dtype == inp.dtype
                    assert recovered.shape == inp.shape
                    recovery_expected = torch.zeros(4).put_(recovery_index.cpu(), before_invalid_source)
                    torch.testing.assert_close(recovered.cpu(), recovery_expected, rtol=0, atol=0)
                    torch.testing.assert_close(inp.cpu(), before_invalid_inp, rtol=0, atol=0)
                    torch.testing.assert_close(index.cpu(), before_invalid_index, rtol=0, atol=0)
                    torch.testing.assert_close(source.cpu(), before_invalid_source, rtol=0, atol=0)
                    assert [tensor_abi(value) for value in (inp, index, source)] == before_invalid_abi
                    report.update(recovery_ABI_values=True, recovery_nonempty_control=True,
                                  context_poisoned=False)
                except BaseException as recovery_error:
                    report.update(recovery_ABI_values=False,
                                  recovery_error_type=type(recovery_error).__name__,
                                  recovery_error=str(recovery_error))
                report.update(device_assert=False, candidate_calls=registry.call_counts())
                print("PUT_INPLACE_BOUNDS_RESULT:" + json.dumps(report), flush=True)
                sys.exit(87 if report.get("recovery_ABI_values") else 2)
            try:
                probe = torch.ones(1, device=inp.device)
                probe.add_(1)
                torch_device_fn.synchronize()
            except RuntimeError as poison:
                report["context_poisoned"] = device_fault(poison)
                report["context_error"] = str(poison)
                report["context_error_first_line"] = str(poison).partition(chr(10))[0]
            report["device_assert"] = (
                device_fault(error) and report["context_poisoned"]
                and report["error_first_line"].strip().lower()
                == report["context_error_first_line"].strip().lower()
            )
            report["candidate_calls"] = registry.call_counts()
            print("PUT_INPLACE_BOUNDS_RESULT:" + json.dumps(report), flush=True)
            sys.exit(86 if report["device_assert"] else 2)
        else:
            report.update(device_assert=False, context_poisoned=False,
                          candidate_calls=registry.call_counts())
            print("PUT_INPLACE_BOUNDS_RESULT:" + json.dumps(report), flush=True)
            sys.exit(0)
        """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    command = [sys.executable, "-c", code, json.dumps(payload)]
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    ) as child:

        def interrupt_wait(_signum, _frame):
            raise KeyboardInterrupt("put_ bounds child wait terminated")

        previous_sigterm = signal.signal(signal.SIGTERM, interrupt_wait)
        try:
            stdout, stderr = child.communicate(timeout=120)
        except BaseException as error:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = child.communicate()
            if isinstance(error, subprocess.TimeoutExpired):
                pytest.fail("put_ bounds child timed out: " + stderr[-500:])
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
        reports = [
            json.loads(line.partition(":")[2])
            for line in stdout.splitlines()
            if line.startswith("PUT_INPLACE_BOUNDS_RESULT:")
        ]
        assert len(reports) == 1, (
            f"missing bounds report: returncode={child.returncode}, "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )
        report = reports[0]
        assert report["bad"] == bad
        assert report["binding"] == payload["binding"]
        assert report["ref_cpu"] == utils.TO_CPU
        request.node.user_properties.append(
            ("put_inplace_bounds_child", json.dumps(report, sort_keys=True))
        )
        assert report["valid_completed"]
        assert report["phase"] == "invalid"
        if report.get("host_index_error"):
            assert child.returncode == 87
            assert report.get("valid_ABI_values") and report.get("recovery_ABI_values")
            assert report.get("error_type") == "IndexError"
            assert report.get("host_index_error") is True
            assert report.get("actual_oob_verified") and report.get(
                "input_index_source_readonly"
            )
            assert report.get("recovery_nonempty_control")
            assert report.get("error") == report.get("expected_host_message")
            assert (
                report["context_poisoned"] is False and report["device_assert"] is False
            )
            if selected is not None:
                assert report["candidate_calls"].get("flag_gems.put_") == 3
            return
        if selected is not None:
            assert report["candidate_calls"].get("flag_gems.put_") == 2
        assert child.returncode == 86 and report["device_assert"], (
            "expected an actual device-side bounds failure after the valid call; "
            f"returncode={child.returncode}, report={report}, stderr={stderr[-500:]!r}"
        )


@pytest.mark.put_
@pytest.mark.parametrize("boundary", [-4, -1, 0, 3])
def test_put__index_boundary_valid(boundary):
    # The values just inside the valid range must still be accepted.
    inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    index = torch.tensor([boundary], dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    if flag_gems.vendor_name == "ascend" and boundary < 0:
        ref_inp = ref_inp.cpu()
        ref_inp.put_(
            utils.to_reference(index).cpu(), utils.to_reference(source, True).cpu()
        )
    else:
        ref_inp.put_(utils.to_reference(index), utils.to_reference(source, True))
    flag_gems.put_(inp, index, source)

    _assert_put_equal(inp, ref_inp, torch.float32)


@pytest.mark.put_
def test_put__device_mismatch_raises():
    # torch requires self/index/source on one device and refuses to transfer.
    inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
    index_gpu = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source_gpu = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.put_(inp, index_gpu.cpu(), source_gpu)
    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.put_(inp, index_gpu, source_gpu.cpu())


@pytest.mark.put_
def test_put__index_dtype_and_numel_raise():
    inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="long tensor for index"):
        flag_gems.put_(
            inp,
            torch.tensor([0], dtype=torch.int32, device=flag_gems.device),
            source,
        )
    mismatch_dtype = (
        torch.int32 if flag_gems.vendor_name == "kunlunxin" else torch.float64
    )
    mismatch_source = torch.ones(1, dtype=mismatch_dtype, device=flag_gems.device)
    assert mismatch_source.dtype != inp.dtype
    with pytest.raises(RuntimeError, match="same dtype"):
        flag_gems.put_(
            inp,
            torch.tensor([0], dtype=torch.int64, device=flag_gems.device),
            mismatch_source,
        )
    with pytest.raises(IndexError, match="same number of elements"):
        flag_gems.put_(
            inp,
            torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device),
            source,
        )


# ---------------------------------------------------------------------------
# Memory-overlap semantics.
# ---------------------------------------------------------------------------
@pytest.mark.put_
def test_put__expanded_self_raises():
    # An expanded `self` aliases one element from several positions, so the
    # write target is ambiguous and torch refuses it.
    inp = torch.zeros(1, 4, dtype=torch.float32, device=flag_gems.device).expand(3, 4)
    index = torch.arange(4, dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(4, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="more than one element"):
        torch.zeros(1, 4).expand(3, 4).put_(index.cpu(), source.cpu())
    with pytest.raises(RuntimeError, match="more than one element"):
        flag_gems.put_(inp, index, source)


@pytest.mark.put_
def test_put__aliased_index_or_source_raises():
    # torch rejects an index/source that overlaps `self`, since the write would
    # race against the read.
    index = torch.arange(4, dtype=torch.int64, device=flag_gems.device)

    inp = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.put_(inp, index, inp[4:8])

    # `source is self`: the index must cover all 16 elements, since the
    # index/source numel check runs before the overlap check.
    inp = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.put_(
            inp,
            torch.arange(16, dtype=torch.int64, device=flag_gems.device),
            inp,
        )

    idx_self = torch.arange(16, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.put_(
            idx_self,
            idx_self[:4],
            torch.arange(100, 104, dtype=torch.int64, device=flag_gems.device),
        )


@pytest.mark.put_
def test_put__interleaved_views_allowed():
    # torch permits an overlap it classifies as "too hard" to check -- here
    # `self` and `source` interleave in one storage without sharing elements.
    # It must keep working and agree with native.
    base = torch.arange(20, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    index = torch.arange(10, dtype=torch.int64, device=flag_gems.device)
    ref_base[::2].put_(utils.to_reference(index), ref_base[1::2])
    flag_gems.put_(base[::2], index, base[1::2])

    _assert_put_equal(base, ref_base, torch.float32)


@pytest.mark.put_
def test_put__disjoint_views_same_storage():
    # Non-overlapping slices of one storage are plainly legal.
    base = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    index = torch.arange(4, dtype=torch.int64, device=flag_gems.device)
    ref_base[0:4].put_(utils.to_reference(index), ref_base[8:12])
    flag_gems.put_(base[0:4], index, base[8:12])

    _assert_put_equal(base, ref_base, torch.float32)


# ---------------------------------------------------------------------------
# Non-contiguous `index` / `source`.
# ---------------------------------------------------------------------------
# `index.reshape(-1)` is a no-op on an already-1-D strided view and preserves
# the non-unit stride, so the kernel must not assume unit-stride operands.
@pytest.mark.put_
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__strided_source(accumulate):
    inp = torch.zeros(8, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    src_base = torch.arange(1, 17, dtype=torch.float32, device=flag_gems.device)
    source = src_base[::2]  # 1-D, stride 2
    index = torch.arange(8, dtype=torch.int64, device=flag_gems.device)

    ref_inp.put_(
        utils.to_reference(index),
        utils.to_reference(source, True),
        accumulate=accumulate,
    )
    flag_gems.put_(inp, index, source, accumulate=accumulate)

    _assert_put_equal(inp, ref_inp, torch.float32)


@pytest.mark.put_
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__strided_index(accumulate):
    inp = torch.zeros(8, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    # Interleave the real indices with a sentinel so a unit-stride read would
    # pick up the wrong values instead of silently working.
    idx_base = torch.tensor(
        [0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7, 7, 7],
        dtype=torch.int64,
        device=flag_gems.device,
    )
    index = idx_base[::2]  # 1-D, stride 2 -> 0..7
    source = torch.arange(1, 9, dtype=torch.float32, device=flag_gems.device)

    ref_inp.put_(
        utils.to_reference(index),
        utils.to_reference(source, True),
        accumulate=accumulate,
    )
    flag_gems.put_(inp, index, source, accumulate=accumulate)

    _assert_put_equal(inp, ref_inp, torch.float32)


@pytest.mark.put_
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__transposed_index_and_source(accumulate):
    # A transposed 2-D index/source flattens in row-major order, which differs
    # from its memory order.
    inp = torch.zeros(6, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    index = torch.arange(6, dtype=torch.int64, device=flag_gems.device).reshape(2, 3).T
    source = (
        torch.arange(1, 7, dtype=torch.float32, device=flag_gems.device).reshape(2, 3).T
    )

    ref_inp.put_(
        utils.to_reference(index),
        utils.to_reference(source, True),
        accumulate=accumulate,
    )
    flag_gems.put_(inp, index, source, accumulate=accumulate)

    _assert_put_equal(inp, ref_inp, torch.float32)


# ---------------------------------------------------------------------------
# Views of a larger storage: writes must land in the view and leave the rest of
# the base tensor untouched.
# ---------------------------------------------------------------------------
@pytest.mark.put_
@pytest.mark.parametrize(
    "make_view, base_numel, index_vals",
    [
        (lambda b: b[5:15], 20, [0, 9]),
        (lambda b: b[::2], 20, [0, 4, 9]),
        (lambda b: b[3::2], 20, [0, 8]),
        (lambda b: b.view(4, 6)[1:3, 1:5], 24, [0, 3, 7]),
        (lambda b: b.view(4, 5).T, 20, [0, 5, 19]),
        (lambda b: b.flip(0), 20, [0, 19]),
        (lambda b: b.view(2, 3, 4).permute(2, 0, 1), 24, [0, 11, 23]),
    ],
)
def test_put__view_of_larger_storage(make_view, base_numel, index_vals):
    base = torch.arange(base_numel, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    index = torch.tensor(index_vals, dtype=torch.int64, device=flag_gems.device)
    source = -torch.arange(
        1, len(index_vals) + 1, dtype=torch.float32, device=flag_gems.device
    )

    make_view(ref_base).put_(utils.to_reference(index), utils.to_reference(source))
    view = make_view(base)
    res = flag_gems.put_(view, index, source)
    assert res is view

    # Comparing the whole base catches both a wrong offset and a stray write.
    _assert_put_equal(base, ref_base, torch.float32)


@pytest.mark.put_
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend native reference does not support ComplexHalf clone/put",
)
def test_put__complex32_rejected():
    # Native CUDA `put_` has no ComplexHalf kernel; `complex32` used to fall
    # through to the FP16 real-view path, silently succeeding where ATen raises.
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

    # RuntimeError vs. NotImplementedError varies with the backend/PyTorch
    # version; the candidate must retain the native family and exact message.
    with pytest.raises(RuntimeError, match="not implemented.*ComplexHalf") as ref_err:
        inp.clone().put_(index, source)
    with pytest.raises(type(ref_err.value)) as res_err:
        flag_gems.put_(inp, index, source)

    assert str(ref_err.value) == str(res_err.value)
