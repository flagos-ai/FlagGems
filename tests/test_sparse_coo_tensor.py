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

import importlib
from unittest.mock import patch

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _components(nnz=37, sparse_dim=2, dense_shape=(), dtype=torch.float32):
    indices = torch.randint(0, 13, (sparse_dim, nnz), device=flag_gems.device)
    values = torch.randn((nnz, *dense_shape), dtype=dtype, device=flag_gems.device)
    return indices, values


def _assert_components(actual, expected):
    assert actual.layout == expected.layout == torch.sparse_coo
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert actual.is_coalesced() == expected.is_coalesced()
    torch.testing.assert_close(actual._indices(), expected._indices())
    torch.testing.assert_close(actual._values(), expected._values())


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("nnz", [0, 1, 37, 2051])
@pytest.mark.parametrize("sparse_dim,dense_shape", [(1, ()), (2, (3, 2)), (0, (2,))])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_infer_size(nnz, sparse_dim, dense_shape, dtype):
    indices, values = _components(nnz, sparse_dim, dense_shape, dtype)
    ref = torch.sparse_coo_tensor(indices, values)
    res = flag_gems.sparse_coo_tensor(indices, values)
    _assert_components(res, ref)
    if indices.numel():
        assert res._indices().data_ptr() == indices.data_ptr()
    if values.numel():
        assert res._values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize(
    "index_dtype", [torch.int64, torch.int32, torch.float32, torch.complex64]
)
@pytest.mark.parametrize("dtype", [None, torch.float64, torch.float16])
@pytest.mark.parametrize("explicit", [False, True])
def test_sparse_coo_tensor_conversion_and_strides(index_dtype, dtype, explicit):
    indices, values = _components(74, 2, (3,))
    indices = indices.to(index_dtype)[:, ::2]
    values = values[::2, ::2]
    size = ((13, 13, 2),) if explicit else ()
    ref = torch.sparse_coo_tensor(indices, values, *size, dtype=dtype)
    res = flag_gems.sparse_coo_tensor(indices, values, *size, dtype=dtype)
    _assert_components(res, ref)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", [None, torch.float64])
@pytest.mark.parametrize("explicit", [False, True])
def test_sparse_coo_tensor_scalar_values(dtype, explicit):
    indices = torch.tensor([[2], [4]], device=flag_gems.device)
    values = torch.tensor(3.0, device=flag_gems.device)
    size = ((3, 5),) if explicit else ()
    _assert_components(
        flag_gems.sparse_coo_tensor(indices, values, *size, dtype=dtype),
        torch.sparse_coo_tensor(indices, values, *size, dtype=dtype),
    )


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize(
    "case", ["negative", "shape", "nnz", "sparse_indices", "layout"]
)
def test_sparse_coo_tensor_invalid_inputs(case):
    indices, values = _components()
    kwargs = {}
    if case == "negative":
        indices[0, 0] = -1
    elif case == "shape":
        indices = indices.flatten()
    elif case == "nnz":
        values = values[:-1]
    elif case == "sparse_indices":
        indices = indices.to_sparse()
    elif case == "layout":
        kwargs["layout"] = torch.sparse_csr
    with pytest.raises((RuntimeError, ValueError)):
        torch.ops.aten.sparse_coo_tensor.indices(indices, values, **kwargs)
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.sparse_coo_tensor(indices, values, **kwargs)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("coalesced", [None, False, True])
def test_sparse_coo_tensor_coalesced(coalesced):
    indices = torch.tensor([[0, 1, 2]], device=flag_gems.device)
    values = torch.randn(3, device=flag_gems.device)
    kwargs = dict(is_coalesced=coalesced)
    _assert_components(
        flag_gems.sparse_coo_tensor(indices, values, **kwargs),
        torch.ops.aten.sparse_coo_tensor.indices(indices, values, **kwargs),
    )


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_dispatch():
    indices, values = _components()
    ref = torch.sparse_coo_tensor(indices, values)
    module = importlib.import_module("flag_gems.ops.sparse_coo_tensor")
    original_kernel = module._coo_bounds_kernel
    previous = flag_gems.current_work_registrar
    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=["sparse_coo_tensor"])
        with (
            patch.object(module.logger, "debug") as log,
            patch.object(module, "_infer_size", wraps=module._infer_size) as infer,
            patch.object(module, "_coo_bounds_kernel") as bounds_kernel,
        ):
            bounds_kernel.__getitem__.side_effect = original_kernel.__getitem__
            res = torch.ops.aten.sparse_coo_tensor.indices(indices, values)
            assert bounds_kernel.__getitem__.call_count == 1
            assert log.call_count == 1
            assert infer.call_count == 1
            _assert_components(res, ref)
            log.reset_mock()
            infer.reset_mock()
            res = torch.ops.aten.sparse_coo_tensor.indices_size(
                indices, values, list(ref.shape)
            )
            assert log.call_count == 1
            assert infer.call_count == 0
            _assert_components(res, ref)
            log.reset_mock()
            empty = torch.ops.aten.sparse_coo_tensor.size(
                list(ref.shape), device=flag_gems.device
            )
            out = torch.empty(
                ref.shape, layout=torch.sparse_coo, device=flag_gems.device
            )
            result = torch.ops.aten.sparse_coo_tensor.size_out(list(ref.shape), out=out)
            assert log.call_count == 0
            assert empty._nnz() == 0
            assert result is out and out._nnz() == 0
            # Public factories select the sparse backend, not our dense key.
            _assert_components(torch.sparse_coo_tensor(indices, values), ref)
            _assert_components(torch.sparse_coo_tensor(indices, values, ref.shape), ref)
            assert log.call_count == 0
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_registration_scope():
    entries = [
        e[0] for e in flag_gems._FULL_CONFIG if e[1] is flag_gems.sparse_coo_tensor
    ]
    assert entries == ["sparse_coo_tensor.indices", "sparse_coo_tensor.indices_size"]


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_device_index():
    if flag_gems.device != "cuda" or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with torch.cuda.device(1):
        indices, values = _components()
    with torch.cuda.device(0):
        ref = torch.sparse_coo_tensor(indices, values, device="cuda")
        res = flag_gems.sparse_coo_tensor(indices, values, device="cuda")
    _assert_components(res, ref)
    assert res.device == torch.device("cuda:0")


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_mismatched_devices():
    indices, values = _components()
    with pytest.raises(RuntimeError):
        flag_gems.sparse_coo_tensor(indices.cpu(), values)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("dtype", [None, torch.float64, torch.float16])
def test_sparse_coo_tensor_autograd(dtype):
    indices, values = _components()
    values.requires_grad_()
    result = flag_gems.sparse_coo_tensor(indices, values, dtype=dtype)
    result.to_dense().sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("pin_memory", [None, False, True])
@pytest.mark.parametrize("coalesced", [None, False, True])
@pytest.mark.parametrize("dispatched", [False, True])
def test_sparse_coo_tensor_pin_memory(pin_memory, coalesced, dispatched):
    indices = torch.tensor([[0, 1, 2]], device=flag_gems.device)
    values = torch.ones(3, device=flag_gems.device)
    kwargs = dict(pin_memory=pin_memory, is_coalesced=coalesced)
    error = None
    try:
        ref = torch.ops.aten.sparse_coo_tensor.indices(indices, values, **kwargs)
    except RuntimeError as exc:
        assert pin_memory is True
        assert "pin" in str(exc).lower()
        error = str(exc)
    previous = flag_gems.current_work_registrar
    lib = torch.library.Library("aten", "IMPL")
    try:
        if dispatched:
            flag_gems.only_enable(lib=lib, include=["sparse_coo_tensor"])
        op = (
            torch.ops.aten.sparse_coo_tensor.indices
            if dispatched
            else flag_gems.sparse_coo_tensor
        )
        if error is not None:
            with pytest.raises(RuntimeError) as actual_error:
                op(indices, values, **kwargs)
            assert str(actual_error.value) == error
        else:
            res = op(indices, values, **kwargs)
            _assert_components(res, ref)
            assert res.is_pinned() == ref.is_pinned()
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("coalesced", [None, False, True])
def test_sparse_coo_tensor_pin_memory_rejection(coalesced):
    indices, values = _components()
    module = importlib.import_module("flag_gems.ops.sparse_coo_tensor")
    message = "cannot pin device tensors"
    with (
        patch.object(
            torch, "sparse_coo_tensor", side_effect=RuntimeError(message)
        ) as factory,
        patch.object(module, "_infer_size") as infer,
    ):
        with pytest.raises(RuntimeError, match=message):
            flag_gems.sparse_coo_tensor(
                indices, values, pin_memory=True, is_coalesced=coalesced
            )
        factory.assert_called_once_with(
            indices, values, dtype=None, device=values.device, pin_memory=True
        )
        infer.assert_not_called()


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_large_reduction():
    indices, values = _components(1024 * 1024 + 1)
    indices[0, -1] = 25
    _assert_components(
        flag_gems.sparse_coo_tensor(indices, values),
        torch.sparse_coo_tensor(indices, values),
    )
    indices[1, -1] = -1
    with pytest.raises(RuntimeError, match="negative index"):
        flag_gems.sparse_coo_tensor(indices, values)


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_size_overflow():
    indices = torch.tensor([[2**63 - 1]], device=flag_gems.device)
    values = torch.ones(1, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.sparse_coo_tensor(indices, values)
    with pytest.raises(RuntimeError, match="overflows"):
        flag_gems.sparse_coo_tensor(indices, values)


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_cpu_transfer():
    indices = torch.tensor([[1, 2]])
    values = torch.tensor([3.0, 4.0])
    _assert_components(
        flag_gems.sparse_coo_tensor(indices, values, device=flag_gems.device),
        torch.sparse_coo_tensor(indices, values, device=flag_gems.device),
    )


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor_invariant_check():
    indices = torch.tensor([[1, 0]], device=flag_gems.device)
    values = torch.ones(2, device=flag_gems.device)
    with torch.sparse.check_sparse_tensor_invariants():
        with pytest.raises(RuntimeError):
            torch.sparse_coo_tensor(indices, values, (2,), is_coalesced=True)
        with pytest.raises(RuntimeError):
            flag_gems.sparse_coo_tensor(indices, values, is_coalesced=True)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("nnz", [0, 1, 2051])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dense_shape", [(), (3,)])
def test_sparse_coo_tensor_explicit_size(nnz, dtype, dense_shape):
    indices, values = _components(nnz, 2, dense_shape, dtype)
    size = (13, 13) + dense_shape
    actual = flag_gems.sparse_coo_tensor(indices, values, size)
    expected = torch.sparse_coo_tensor(indices, values, size)
    _assert_components(actual, expected)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize(
    "case", ["negative_size", "out_of_range", "negative_index", "dense_shape"]
)
def test_sparse_coo_tensor_explicit_validation(case):
    indices = torch.tensor([[1, 2]], device=flag_gems.device)
    values = torch.ones(2, device=flag_gems.device)
    size = (3,)
    if case == "negative_size":
        size = (-1,)
    elif case == "out_of_range":
        size = (2,)
    elif case == "negative_index":
        indices[0, 0] = -1
    elif case == "dense_shape":
        values = values[:, None]
    with torch.sparse.check_sparse_tensor_invariants():
        with pytest.raises(RuntimeError) as reference_error:
            torch.sparse_coo_tensor(indices, values, size)
        with pytest.raises(RuntimeError) as error:
            flag_gems.sparse_coo_tensor(indices, values, size)
    if case == "negative_size":
        assert "negative dimension" in str(error.value)
    else:
        assert str(error.value) == str(reference_error.value)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("pin_memory", [None, False, True])
def test_sparse_coo_tensor_explicit_pin_memory(pin_memory):
    indices, values = _components()
    kwargs = dict(pin_memory=bool(pin_memory))
    if pin_memory:
        with pytest.raises(RuntimeError):
            torch.sparse_coo_tensor(indices, values, (13, 13), **kwargs)
        with pytest.raises(RuntimeError):
            flag_gems.sparse_coo_tensor(indices, values, (13, 13), **kwargs)
    else:
        _assert_components(
            flag_gems.sparse_coo_tensor(indices, values, (13, 13), **kwargs),
            torch.sparse_coo_tensor(indices, values, (13, 13), **kwargs),
        )
