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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg
from . import test_utils as tu

pytestmark = pytest.mark.sparse_coo_tensor

_COO_DTYPES = tu.REQUIRED_DTYPES + [torch.float64, torch.int16, torch.bool]

_COO_2D_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]]),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((6, 8), [[0, 1, 3, 0], [1, 2, 4, 0]]),
    ((5, 6), [[2, 4, 1], [3, 0, 5]]),
    ((3, 3), [[0, 2, 1, 2], [0, 1, 2, 0]]),
]

_COO_ND_CASES = [
    ((5,), [[0, 2, 4]]),
    ((3, 4, 5), [[0, 1, 2, 1], [1, 3, 0, 2]]),
    ((2, 3, 7), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]]),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2], [1, 3, 0]]),
    ((3, 3, 3), [[0, 2, 1], [1, 0, 2], [2, 1, 0]]),
]

_COO_INFERRED_CASES = [
    ((2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ((4, 5), [[0, 1, 3, 0], [1, 2, 4, 0]], ()),
    ((5, 6), [[2, 4, 1], [3, 0, 5]], ()),
    ((2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

_COO_SIZE_ONLY_CASES = [((5,),), ((2, 3),), ((4, 5, 6),)]

_COO_EMPTY_CASES = [((2, 3), 2), ((4, 5, 6), 2), ((2, 3, 4, 5), 2), ((3, 4, 5), 3)]

_COO_VALUE_CASES = [
    ("indices_size", (2, 3), [[0, 1, 1], [2, 0, 2]], ()),
    ("indices_size", (2, 3, 4, 5), [[0, 1, 1], [2, 0, 2]], (4, 5)),
    ("indices", (2, 3, 4), [[0, 1, 1], [2, 0, 2]], (4,)),
]

_NAN_INF_CASES = [
    ((5,), [[0, 2, 4]]),
    ((2, 3), [[0, 1, 1], [2, 0, 2]]),
    ((4, 5, 6), [[0, 1, 1], [2, 0, 2]]),
]

_VALUE_RANGE_CASES = [
    pytest.param(value_range, case, dtype)
    for case in _COO_VALUE_CASES
    for dtype in _COO_DTYPES
    for value_range in tu.selected_ranges()
]

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _required_devices():
    target = torch.empty(0, dtype=torch.uint8, device=flag_gems.device).device
    return target, torch.device("cpu") if cfg.TO_CPU else target


def _storage_bytes(inp):
    storage = inp.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _copy_tensor(inp, device):
    # Reinterpret on the original device before transport: typed Double copies
    # can narrow on KU, and device FP8 casts/slices are auxiliary gaps.
    raw = _storage_bytes(inp).to(device=device).clone()
    result = raw.view(inp.dtype).as_strided(
        inp.shape, inp.stride(), inp.storage_offset()
    )
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    result.requires_grad_(inp.requires_grad)
    return result


def _check_dense(inp, dtype, shape, device):
    assert isinstance(inp, torch.Tensor)
    assert inp.layout == torch.strided and inp.dtype == dtype
    assert tuple(inp.shape) == tuple(shape) and inp.device == device
    return inp


def _cpu_encode(dtype):
    return flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        *_FP8_DTYPES,
        torch.float64,
    )


def _make_values(nnz, dense_shape, dtype, value_range=None, *, required_device):
    if value_range is None:
        value_range = ["-1", "1"]
    shape = (nnz,) + tuple(dense_shape)
    if _cpu_encode(dtype):
        # Preserve the original TU make_tensor bounds and dtype clamp semantics.
        low = tu.resolve_bound(value_range[0], dtype)
        high = tu.resolve_bound(value_range[1], dtype)
        if low == high:
            host = torch.full(shape, low, dtype=dtype, device="cpu")
        else:
            host = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        values = _copy_tensor(host, required_device)
    else:
        values = tu.make_input(dtype, shape, value_range).to(required_device)
    return _check_dense(values, dtype, shape, required_device)


def _make_special_values(shape, dtype, scenario, *, required_device):
    # Original FP32 five-payload encoding, then requested dtype and byte repeats
    # entirely on CPU; no device special factory, typed repeat or narrowing cast.
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    base = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu").to(dtype)
    numel = math.prod(int(extent) for extent in shape)
    repeats = (numel + base.numel() - 1) // base.numel()
    raw = base.view(torch.uint8).repeat(repeats)[: numel * base.element_size()]
    host = raw.view(dtype).reshape(shape)
    return _check_dense(
        _copy_tensor(host, required_device), dtype, shape, required_device
    )


def _to_reference(inp, *, required_device, reference_device):
    try:
        assert isinstance(inp, torch.Tensor) and inp.device == required_device
        result = _copy_tensor(inp, reference_device)
        _check_dense(result, inp.dtype, inp.shape, reference_device)
        assert result.stride() == inp.stride()
        assert result.storage_offset() == inp.storage_offset()
        assert result.is_conj() == inp.is_conj() and result.is_neg() == inp.is_neg()
        assert result.untyped_storage()._cdata != inp.untyped_storage()._cdata
    except Exception as error:
        raise AssertionError("Invalid COO reference transport") from error
    return result


def _assert_values(actual, expected, *, required_device, reference_device):
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    _check_dense(actual, expected.dtype, expected.shape, required_device)
    _check_dense(expected, actual.dtype, actual.shape, reference_device)
    # Original comparison/tolerances/NaN policy remain in TU, after ABI checks.
    tu.assert_result_equal(_copy_tensor(actual, "cpu"), _copy_tensor(expected, "cpu"))


def _assert_indices(actual, expected, *, required_device, reference_device):
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    _check_dense(actual, expected.dtype, expected.shape, required_device)
    _check_dense(expected, actual.dtype, actual.shape, reference_device)
    utils.gems_assert_equal(_copy_tensor(actual, "cpu"), _copy_tensor(expected, "cpu"))


def _input_state(inp):
    assert isinstance(inp, torch.Tensor) and inp.layout == torch.strided
    storage = inp.untyped_storage()
    signature = (
        type(inp),
        inp.device,
        inp.dtype,
        inp.layout,
        tuple(inp.shape),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.requires_grad,
        inp._version,
        inp.is_conj(),
        inp.is_neg(),
        id(inp._base),
        storage._cdata,
        inp.data_ptr(),
        storage.data_ptr(),
        storage.nbytes(),
    )
    return signature, storage, inp._base


def _invoke_candidate(*args, required_device, **kwargs):
    # Native errors propagate naturally; guard faults cannot satisfy raises.
    # Writable out is held but intentionally excluded from readonly snapshots.
    out = kwargs.get("out")
    inputs = [x for x in args if isinstance(x, torch.Tensor)]
    try:
        candidate = flag_gems.sparse_coo_tensor
        assert callable(candidate), "COO candidate binding is not callable"
        states = [_input_state(x) for x in inputs]
        assert all(x.device == required_device for x in inputs)
        snapshots = [_storage_bytes(x).cpu().clone() for x in inputs]
        out_dtype = out.dtype if out is not None else None
        if out is not None:
            assert isinstance(out, torch.Tensor) and out.device == required_device
            assert out.layout == torch.sparse_coo
        expected_dtype = kwargs.get("dtype", out_dtype)
        explicit_size = kwargs.get("size")
        if explicit_size is None:
            if len(args) == 1:
                explicit_size = args[0]
            elif len(args) == 3:
                explicit_size = args[2]
        expected_shape = tuple(explicit_size) if explicit_size is not None else None
    except Exception as error:
        raise AssertionError("Invalid COO input ABI before candidate") from error
    try:
        result = candidate(*args, **kwargs)
    finally:
        try:
            for inp, state, snapshot in zip(inputs, states, snapshots):
                assert _input_state(inp)[0] == state[0]
                assert torch.equal(_storage_bytes(inp).cpu(), snapshot)
        except Exception as error:
            raise AssertionError("COO constructor mutated an input") from error
    try:
        assert isinstance(result, torch.Tensor)
        assert result.device == required_device and result.layout == torch.sparse_coo
        if expected_dtype is not None:
            assert result.dtype == expected_dtype
        if expected_shape is not None:
            assert tuple(result.shape) == expected_shape
        if out is not None:
            assert result is out
    except Exception as error:
        raise AssertionError("Invalid COO candidate output ABI") from error
    return result


def _make_out_buffer(size, dtype, device, nnz):
    # Non-empty original buffers make size_out clear existing entries.
    if nnz == 0:
        return torch.ops.aten.sparse_coo_tensor(list(size), dtype=dtype, device=device)
    indices = torch.zeros((len(size), nnz), dtype=torch.long, device=device)
    values = _make_values(
        nnz, (), dtype, ["0", "1"], required_device=torch.device(device)
    )
    return torch.ops.aten.sparse_coo_tensor(
        indices, values, list(size), dtype=dtype, device=device
    )


def _assert_coo_structure(
    res_out,
    ref_out,
    size,
    nnz,
    dtype,
    sparse_dim,
    dense_dim,
    is_coalesced=None,
    *,
    required_device,
    reference_device,
):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.layout == ref_out.layout == torch.sparse_coo
    assert tuple(res_out.shape) == tuple(ref_out.shape) == tuple(size)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == required_device and ref_out.device == reference_device
    assert res_out.sparse_dim() == ref_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == ref_out.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out) == nnz
    assert tuple(torch.ops.aten._indices(res_out).shape) == (sparse_dim, nnz)
    assert tuple(torch.ops.aten._values(res_out).shape) == (nnz,) + tuple(
        size[sparse_dim:]
    )
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    if is_coalesced is not None:
        assert res_out.is_coalesced() == is_coalesced
    _assert_indices(
        res_out._indices(),
        ref_out._indices(),
        required_device=required_device,
        reference_device=reference_device,
    )


def _call_reference(indices, values, size, dtype, *, required_device, reference_device):
    ref_indices = _to_reference(
        indices, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    if size is None:
        return torch.ops.aten.sparse_coo_tensor(
            ref_indices, ref_values, dtype=dtype, device=ref_indices.device
        )
    return torch.ops.aten.sparse_coo_tensor(
        ref_indices, ref_values, list(size), dtype=dtype, device=ref_indices.device
    )


def _call_candidate(indices, values, size, dtype, *, required_device, **extra):
    if size is None:
        return _invoke_candidate(
            indices,
            values,
            dtype=dtype,
            device=required_device,
            required_device=required_device,
            **extra,
        )
    return _invoke_candidate(
        indices,
        values,
        list(size),
        dtype=dtype,
        device=required_device,
        required_device=required_device,
        **extra,
    )


def _master_components(nnz, sparse_dim, dense_shape, dtype, device):
    indices = torch.randint(0, 13, (sparse_dim, nnz), device=device)
    values = torch.randn((nnz, *dense_shape), dtype=dtype, device=device)
    return indices, values


def _assert_master_constructor(
    indices,
    values,
    size=None,
    dtype=None,
    *,
    required_device,
    reference_device,
    **options,
):
    ref_indices = _to_reference(
        indices, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    size_args = (list(size),) if size is not None else ()
    reference = torch.sparse_coo_tensor(
        ref_indices, ref_values, *size_args, dtype=dtype, **options
    )
    result = _invoke_candidate(
        indices,
        values,
        *size_args,
        dtype=dtype,
        required_device=required_device,
        **options,
    )
    _assert_master_result(
        result,
        reference,
        required_device=required_device,
        reference_device=reference_device,
    )
    return result


def _assert_master_result(result, reference, *, required_device, reference_device):
    _assert_coo_structure(
        result,
        reference,
        reference.shape,
        reference._nnz(),
        reference.dtype,
        reference.sparse_dim(),
        reference.dense_dim(),
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        result._values(),
        reference._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.parametrize("nnz", [0, 1, 37, 2051])
@pytest.mark.parametrize("sparse_dim,dense_shape", [(1, ()), (2, (3, 2)), (0, (2,))])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_infer_size(nnz, sparse_dim, dense_shape, dtype):
    target, reference = _required_devices()
    indices, values = _master_components(nnz, sparse_dim, dense_shape, dtype, target)
    result = _assert_master_constructor(
        indices, values, required_device=target, reference_device=reference
    )
    if indices.numel():
        assert result._indices().data_ptr() == indices.data_ptr()
    if values.numel():
        assert result._values().data_ptr() == values.data_ptr()


_FP64_CONSTRUCTOR_CAST = pytest.param(
    torch.float64,
    marks=pytest.mark.skipif(
        flag_gems.vendor_name == "iluvatar",
        reason="Iluvatar native FP32-to-FP64 casts zero sparse values and gradients",
    ),
)


@pytest.mark.parametrize("dtype", [None, _FP64_CONSTRUCTOR_CAST, torch.float16])
@pytest.mark.parametrize("explicit", [False, True])
def test_sparse_coo_tensor_strided_long_indices(dtype, explicit):
    target, reference = _required_devices()
    indices, values = _master_components(74, 2, (3,), torch.float32, target)
    _assert_master_constructor(
        indices[:, ::2],
        values[::2, ::2],
        (13, 13, 2) if explicit else None,
        dtype,
        required_device=target,
        reference_device=reference,
    )


@pytest.mark.parametrize("dtype", [None, _FP64_CONSTRUCTOR_CAST])
@pytest.mark.parametrize("explicit", [False, True])
def test_sparse_coo_tensor_scalar_values(dtype, explicit):
    target, reference = _required_devices()
    indices = torch.tensor([[2], [4]], device=target)
    values = torch.tensor(3.0, device=target)
    _assert_master_constructor(
        indices,
        values,
        (3, 5) if explicit else None,
        dtype,
        required_device=target,
        reference_device=reference,
    )


@pytest.mark.parametrize("coalesced", [None, False, True])
def test_sparse_coo_tensor_inferred_coalesced(coalesced):
    target, reference = _required_devices()
    indices = torch.tensor([[0, 1, 2]], device=target)
    values = torch.randn(3, device=target)
    _assert_master_constructor(
        indices,
        values,
        is_coalesced=coalesced,
        required_device=target,
        reference_device=reference,
    )


@pytest.mark.parametrize("dtype", [None, _FP64_CONSTRUCTOR_CAST, torch.float16])
def test_sparse_coo_tensor_autograd(dtype):
    target, _ = _required_devices()
    indices, values = _master_components(37, 2, (), torch.float32, target)
    values.requires_grad_()
    result = _invoke_candidate(indices, values, dtype=dtype, required_device=target)
    result.to_dense().sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_sparse_coo_tensor_cpu_transfer():
    target, reference_device = _required_devices()
    indices = torch.tensor([[1, 2]])
    values = torch.tensor([3.0, 4.0])
    reference = torch.sparse_coo_tensor(indices, values, device=reference_device)
    result = flag_gems.sparse_coo_tensor(indices, values, device=target)
    _assert_master_result(
        result, reference, required_device=target, reference_device=reference_device
    )


def test_sparse_coo_tensor_device_index():
    if flag_gems.device != "cuda" or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    source = torch.device("cuda:1")
    target = torch.device("cuda:0")
    indices, values = _master_components(37, 2, (), torch.float32, source)
    with torch.cuda.device(target):
        reference = torch.sparse_coo_tensor(indices, values, device="cuda")
        result = flag_gems.sparse_coo_tensor(indices, values, device="cuda")
    _assert_master_result(
        result, reference, required_device=target, reference_device=target
    )


def test_sparse_coo_tensor_mismatched_devices():
    target, _ = _required_devices()
    indices, values = _master_components(37, 2, (), torch.float32, target)
    with pytest.raises(RuntimeError):
        flag_gems.sparse_coo_tensor(indices.cpu(), values)


def test_sparse_coo_tensor_rejects_sparse_indices():
    target, reference_device = _required_devices()
    indices = torch.tensor([[1, 2]], device=target)
    values = torch.ones(2, device=target)
    with pytest.raises((RuntimeError, ValueError)):
        torch.ops.aten.sparse_coo_tensor.indices(
            indices.to(reference_device).to_sparse(), values.to(reference_device)
        )
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.sparse_coo_tensor(indices.to_sparse(), values)


def test_sparse_coo_tensor_size_overflow():
    target, reference_device = _required_devices()
    indices = torch.tensor([[torch.iinfo(torch.int64).max]], device=target)
    values = torch.ones(1, device=target)
    with pytest.raises(RuntimeError):
        torch.sparse_coo_tensor(indices.to(reference_device), values.to(reference_device))
    with pytest.raises(RuntimeError, match="overflows|SymInt"):
        flag_gems.sparse_coo_tensor(indices, values)


def test_sparse_coo_tensor_large_reduction():
    target, reference = _required_devices()
    indices, values = _master_components(1024 * 1024 + 1, 2, (), torch.float32, target)
    indices[0, -1] = 25
    _assert_master_constructor(
        indices, values, required_device=target, reference_device=reference
    )
    indices[1, -1] = -1
    with pytest.raises(RuntimeError, match="negative index"):
        _invoke_candidate(indices, values, required_device=target)


def test_sparse_coo_tensor_inferred_invariant_check():
    target, reference = _required_devices()
    indices = torch.tensor([[1, 0]], device=target)
    values = torch.ones(2, device=target)
    with torch.sparse.check_sparse_tensor_invariants():
        with pytest.raises(RuntimeError):
            torch.sparse_coo_tensor(
                indices.to(reference), values.to(reference), (2,), is_coalesced=True
            )
        with pytest.raises(RuntimeError):
            _invoke_candidate(
                indices, values, is_coalesced=True, required_device=target
            )


@pytest.mark.parametrize("nnz", [0, 1, 2051])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dense_shape", [(), (3,)])
def test_sparse_coo_tensor_explicit_size(nnz, dtype, dense_shape):
    target, reference = _required_devices()
    indices, values = _master_components(nnz, 2, dense_shape, dtype, target)
    _assert_master_constructor(
        indices,
        values,
        (13, 13) + dense_shape,
        required_device=target,
        reference_device=reference,
    )


@pytest.mark.parametrize("case", ["shape", "nnz", "layout"])
def test_sparse_coo_tensor_inferred_invalid_inputs(case):
    target, reference = _required_devices()
    indices, values = _master_components(37, 2, (), torch.float32, target)
    options = {}
    if case == "shape":
        indices = indices.flatten()
    elif case == "nnz":
        values = values[:-1]
    else:
        options["layout"] = torch.sparse_csr
    ref_indices = _to_reference(
        indices, required_device=target, reference_device=reference
    )
    ref_values = _to_reference(
        values, required_device=target, reference_device=reference
    )
    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor.indices(
            ref_indices, ref_values, **options
        ),
        lambda: _invoke_candidate(indices, values, required_device=target, **options),
    )


@pytest.mark.parametrize("case", ["out_of_range", "negative_index"])
def test_sparse_coo_tensor_explicit_bounds(case):
    target, reference = _required_devices()
    indices = torch.tensor([[1, 2]], device=target)
    values = torch.ones(2, device=target)
    size = (2,) if case == "out_of_range" else (3,)
    if case == "negative_index":
        indices[0, 0] = -1
    ref_indices = _to_reference(
        indices, required_device=target, reference_device=reference
    )
    ref_values = _to_reference(
        values, required_device=target, reference_device=reference
    )
    with torch.sparse.check_sparse_tensor_invariants():
        with pytest.raises(RuntimeError) as expected:
            torch.sparse_coo_tensor(ref_indices, ref_values, size)
        with pytest.raises(RuntimeError) as actual:
            _invoke_candidate(indices, values, list(size), required_device=target)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("pin_memory", [None, False, True])
@pytest.mark.parametrize("coalesced", [None, False, True])
def test_sparse_coo_tensor_inferred_pin_memory(pin_memory, coalesced):
    target, _ = _required_devices()
    indices = torch.tensor([[0, 1, 2]], device=target)
    values = torch.ones(3, device=target)
    options = {"pin_memory": pin_memory, "is_coalesced": coalesced}
    try:
        expected = torch.ops.aten.sparse_coo_tensor.indices(indices, values, **options)
    except RuntimeError as error:
        assert pin_memory is True
        assert "pin" in str(error).lower()
        with pytest.raises(RuntimeError) as actual:
            _invoke_candidate(indices, values, required_device=target, **options)
        assert str(actual.value) == str(error)
    else:
        actual = _invoke_candidate(indices, values, required_device=target, **options)
        _assert_master_result(
            actual, expected, required_device=target, reference_device=target
        )
        assert actual.is_pinned() == expected.is_pinned()


@pytest.mark.parametrize("pin_memory", [None, False, True])
def test_sparse_coo_tensor_explicit_pin_memory(pin_memory):
    target, _ = _required_devices()
    indices, values = _master_components(37, 2, (), torch.float32, target)
    options = {"pin_memory": bool(pin_memory)}
    if pin_memory:
        with pytest.raises(RuntimeError):
            torch.sparse_coo_tensor(indices, values, (13, 13), **options)
        with pytest.raises(RuntimeError):
            _invoke_candidate(
                indices, values, [13, 13], required_device=target, **options
            )
    else:
        _assert_master_constructor(
            indices,
            values,
            (13, 13),
            required_device=target,
            reference_device=target,
            **options,
        )


def _assert_rejected(ref_call, candidate_call):
    with pytest.raises(RuntimeError):
        ref_call()
    with pytest.raises((NotImplementedError, RuntimeError, TypeError, ValueError)):
        candidate_call()


def _skip_ku_double_allocator(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip(
            "Kunlunxin native COO Float64 size-only/size_out allocation is unavailable"
        )


_COO_INPUT_DTYPE_CASES = [pytest.param(dtype) for dtype in _COO_DTYPES]


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_size(case, dtype):
    required_device, reference_device = _required_devices()
    _skip_ku_double_allocator(dtype)
    (size,) = case
    ref_device = reference_device
    ref_out = torch.ops.aten.sparse_coo_tensor(
        list(size), dtype=dtype, device=ref_device
    )
    res_out = _invoke_candidate(
        list(size), dtype=dtype, device=required_device, required_device=required_device
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        0,
        dtype,
        len(size),
        0,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor_size_out
@pytest.mark.parametrize("case", _COO_SIZE_ONLY_CASES)
@pytest.mark.parametrize("dtype", _COO_INPUT_DTYPE_CASES)
def test_sparse_coo_tensor_size_out(case, dtype):
    required_device, reference_device = _required_devices()
    _skip_ku_double_allocator(dtype)
    (size,) = case
    ref_device = reference_device
    ref_out = _make_out_buffer(size, dtype, ref_device, nnz=2)
    out = _make_out_buffer(size, dtype, required_device, nnz=2)
    ref_ret = torch.ops.aten.sparse_coo_tensor.size_out(list(size), out=ref_out)
    res_ret = _invoke_candidate(list(size), out=out, required_device=required_device)
    assert res_ret is out
    _assert_coo_structure(
        res_ret,
        ref_ret,
        size,
        0,
        dtype,
        len(size),
        0,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_ret._values(),
        ref_ret._values(),
        required_device=required_device,
        reference_device=reference_device,
    )
    assert tuple(out.shape) == tuple(size)
    assert torch.ops.aten._nnz(out) == 0


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_2D_CASES)
@pytest.mark.parametrize("dtype", _COO_INPUT_DTYPE_CASES)
def test_sparse_coo_tensor_indices_size(case, dtype):
    required_device, reference_device = _required_devices()
    size, indices = case
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=required_device)
    values = _make_values(nnz, (), dtype, required_device=required_device)
    ref_out = _call_reference(
        indices_t,
        values,
        size,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        indices_t, values, size, dtype, required_device=required_device
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        nnz,
        dtype,
        2,
        0,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_ND_CASES)
@pytest.mark.parametrize("dtype", _COO_INPUT_DTYPE_CASES)
def test_sparse_coo_tensor_indices_size_nd(case, dtype):
    required_device, reference_device = _required_devices()
    size, indices = case
    sparse_dim = len(indices)
    dense_dim = len(size) - sparse_dim
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=required_device)
    values = _make_values(
        nnz, tuple(size[sparse_dim:]), dtype, required_device=required_device
    )
    ref_out = _call_reference(
        indices_t,
        values,
        size,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        indices_t, values, size, dtype, required_device=required_device
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        nnz,
        dtype,
        sparse_dim,
        dense_dim,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _COO_INPUT_DTYPE_CASES)
def test_sparse_coo_tensor_indices(case, dtype):
    required_device, reference_device = _required_devices()
    size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=required_device)
    values = _make_values(nnz, dense_shape, dtype, required_device=required_device)
    ref_out = _call_reference(
        indices_t,
        values,
        None,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        indices_t, values, None, dtype, required_device=required_device
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        nnz,
        dtype,
        sparse_dim,
        dense_dim,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_indices_size_empty(case, dtype):
    required_device, reference_device = _required_devices()
    size, sparse_dim = case
    dense_shape = tuple(size[sparse_dim:])
    indices_t = torch.empty(sparse_dim, 0, dtype=torch.long, device=required_device)
    values = _make_values(0, dense_shape, dtype, required_device=required_device)
    ref_out = _call_reference(
        indices_t,
        values,
        size,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        indices_t, values, size, dtype, required_device=required_device
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        0,
        dtype,
        sparse_dim,
        len(dense_shape),
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _COO_INPUT_DTYPE_CASES)
def test_sparse_coo_tensor_indices_size_is_coalesced(dtype):
    required_device, reference_device = _required_devices()
    size = (2, 3)
    indices = [[0, 1, 1], [2, 0, 2]]
    nnz = 3
    indices_t = torch.tensor(indices, dtype=torch.long, device=required_device)
    values = _make_values(nnz, (), dtype, required_device=required_device)
    ref_indices = _to_reference(
        indices_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_coo_tensor(
        ref_indices,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_indices.device,
        is_coalesced=True,
    )
    res_out = _call_candidate(
        indices_t,
        values,
        size,
        dtype,
        is_coalesced=True,
        required_device=required_device,
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        nnz,
        dtype,
        2,
        0,
        is_coalesced=True,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("value_range,case,dtype", _VALUE_RANGE_CASES)
def test_sparse_coo_tensor_value_ranges(value_range, case, dtype):
    required_device, reference_device = _required_devices()
    variant, size, indices, dense_shape = case
    sparse_dim = len(indices)
    dense_dim = len(dense_shape)
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=required_device)
    values = _make_values(
        nnz, dense_shape, dtype, value_range, required_device=required_device
    )
    ref_out = _call_reference(
        indices_t,
        values,
        size if variant == "indices_size" else None,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        indices_t,
        values,
        size if variant == "indices_size" else None,
        dtype,
        required_device=required_device,
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        nnz,
        dtype,
        sparse_dim,
        dense_dim,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _NAN_INF_CASES)
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(dtype, scenario)
        for (dtype, scenario) in tu.selected_cases(tu.special_value_cases(_COO_DTYPES))
    ],
)
def test_sparse_coo_tensor_nan_inf(case, dtype, scenario):
    required_device, reference_device = _required_devices()
    size, indices = case
    sparse_dim = len(indices)
    dense_shape = tuple(size[sparse_dim:])
    nnz = len(indices[0])
    indices_t = torch.tensor(indices, dtype=torch.long, device=required_device)
    values = _make_special_values(
        (nnz,) + dense_shape, dtype, scenario, required_device=required_device
    )
    ref_out = _call_reference(
        indices_t,
        values,
        size,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        indices_t, values, size, dtype, required_device=required_device
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        nnz,
        dtype,
        sparse_dim,
        len(dense_shape),
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", _COO_DTYPES)
def test_sparse_coo_tensor_zero_extent(dtype):
    required_device, reference_device = _required_devices()
    size = (0, 4)
    indices_t = torch.empty(2, 0, dtype=torch.long, device=required_device)
    values = _make_values(0, (), dtype, required_device=required_device)
    ref_out = _call_reference(
        indices_t,
        values,
        size,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        indices_t, values, size, dtype, required_device=required_device
    )
    _assert_coo_structure(
        res_out,
        ref_out,
        size,
        0,
        dtype,
        2,
        0,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        res_out._values(),
        ref_out._values(),
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("case", _COO_VALUE_CASES)
def test_sparse_coo_tensor_inputs_not_mutated(case):
    required_device, reference_device = _required_devices()
    variant, size, indices, dense_shape = case
    nnz = len(indices[0])
    dtype = torch.float32
    indices_t = torch.tensor(indices, dtype=torch.long, device=required_device)
    values = _make_values(nnz, dense_shape, dtype, required_device=required_device)
    indices_before = _to_reference(
        indices_t, required_device=required_device, reference_device=reference_device
    )
    values_before = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    out = _call_candidate(
        indices_t,
        values,
        size if variant == "indices_size" else None,
        dtype,
        required_device=required_device,
    )
    assert out is not indices_t
    _assert_indices(
        indices_t,
        indices_before,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_values(
        values,
        values_before,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_ndim():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor([0, 1, 2], dtype=torch.long, device=required_device)
    values = _make_values(3, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            [3],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, [3], torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_size():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor([[0, 1], [2, 0]], dtype=torch.long, device=required_device)
    values = _make_values(2, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            [-2, 3],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, [-2, 3], torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_size_only_size():
    required_device, reference_device = _required_devices()
    ref_device = reference_device
    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            [-2, 3], dtype=torch.float32, device=ref_device
        ),
        lambda: _invoke_candidate(
            [-2, 3],
            dtype=torch.float32,
            device=required_device,
            required_device=required_device,
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_non_integer_size():
    required_device, reference_device = _required_devices()
    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            [2.5, 3], dtype=torch.float32, device=reference_device
        ),
        lambda: _invoke_candidate(
            [2.5, 3],
            dtype=torch.float32,
            device=required_device,
            required_device=required_device,
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_dtype():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.int32, device=required_device
    )
    values = _make_values(2, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            [2, 3],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, [2, 3], torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_float():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor(
        [[0, 1], [2, 0]], dtype=torch.long, device=required_device
    ).float()
    values = _make_values(2, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            [2, 3],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, [2, 3], torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_nnz_mismatch():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor([[0, 1], [2, 0]], dtype=torch.long, device=required_device)
    values = _make_values(3, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            [2, 3],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, [2, 3], torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_values_dense_dims():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor([[0, 1], [2, 0]], dtype=torch.long, device=required_device)
    values = _make_values(2, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            [2, 3, 4],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, [2, 3, 4], torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_indices_size_rank():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor([[0, 1], [2, 0]], dtype=torch.long, device=required_device)
    values = _make_values(2, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            [2],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, [2], torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_inferred_negative_index():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor([[-1, 1]], dtype=torch.long, device=required_device)
    values = _make_values(2, (), torch.float32, required_device=required_device)
    _assert_rejected(
        lambda: _call_reference(
            indices_t,
            values,
            None,
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        ),
        lambda: _call_candidate(
            indices_t, values, None, torch.float32, required_device=required_device
        ),
    )


@pytest.mark.sparse_coo_tensor_negative
def test_sparse_coo_tensor_negative_layout():
    required_device, reference_device = _required_devices()
    indices_t = torch.tensor([[0, 1], [2, 0]], dtype=torch.long, device=required_device)
    values = _make_values(2, (), torch.float32, required_device=required_device)
    ref_indices = _to_reference(
        indices_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor(
            ref_indices,
            ref_values,
            [2, 3],
            dtype=torch.float32,
            device=ref_indices.device,
            layout=torch.sparse_csr,
        ),
        lambda: _call_candidate(
            indices_t,
            values,
            [2, 3],
            torch.float32,
            layout=torch.sparse_csr,
            required_device=required_device,
        ),
    )


@pytest.mark.sparse_coo_tensor_size_out
def test_sparse_coo_tensor_size_out_negative_shape():
    required_device, reference_device = _required_devices()
    ref_out = torch.ops.aten.sparse_coo_tensor(
        [4, 5], dtype=torch.float32, device=reference_device
    )
    out = torch.ops.aten.sparse_coo_tensor(
        [4, 5], dtype=torch.float32, device=required_device
    )
    _assert_rejected(
        lambda: torch.ops.aten.sparse_coo_tensor.size_out([2, 3], out=ref_out),
        lambda: _invoke_candidate([2, 3], out=out, required_device=required_device),
    )
