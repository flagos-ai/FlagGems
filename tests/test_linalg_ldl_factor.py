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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# Native CPU LDL can be absent in a backend's Torch build. This fallback uses
# the existing NumPy wheel's real ILP64 OpenBLAS SYTRF, retaining signed pivots,
# packed lower LD and INFO. It is used only for the exact missing-LAPACK error.
def _numpy_lapack_library():
    import ctypes
    from pathlib import Path

    import numpy as np

    root = Path(np.__file__).resolve().parent.parent
    found = []
    for p in sorted(
        set([*root.glob("numpy.libs/*blas*.so*"), *root.glob("numpy/.libs/*blas*.so*")])
    ):
        lib = ctypes.CDLL(str(p))
        if all(hasattr(lib, n + "_64_") for n in ["ssytrf", "dsytrf"]):
            found.append((p, lib))
    if len(found) != 1:
        raise ValueError(
            "Expected one existing NumPy ILP64 OpenBLAS with both real SYTRF exports"
        )
    return found[0]


def _numpy_lapack_factor_ex(array, *, lower=True, hermitian=False):
    import ctypes

    import numpy as np

    if lower is not True:
        raise ValueError("Torch LDL lower-packed convention is mandatory")
    if type(hermitian) is not bool:
        raise ValueError("Expected boolean hermitian")
    arr = np.asarray(array)
    if arr.dtype not in [np.dtype("float32"), np.dtype("float64")]:
        raise ValueError("Only original real Float32/Float64 scope")
    if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError("SYTRF needs square matrices")
    _path, lib = _numpy_lapack_library()
    LapackInt = ctypes.c_int64
    F = ctypes.c_float if arr.dtype == np.float32 else ctypes.c_double
    fn = getattr(lib, ("ssytrf" if arr.dtype == np.float32 else "dsytrf") + "_64_")
    fn.restype = None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(LapackInt),
        ctypes.POINTER(F),
        ctypes.POINTER(LapackInt),
        ctypes.POINTER(LapackInt),
        ctypes.POINTER(F),
        ctypes.POINTER(LapackInt),
        ctypes.POINTER(LapackInt),
        ctypes.c_size_t,
    ]
    shape = arr.shape
    n = shape[-1]
    LD = np.empty(shape, dtype=arr.dtype).swapaxes(-1, -2)
    pivots = np.empty((*shape[:-2], n), dtype=np.int32)
    infos = np.empty(shape[:-2], dtype=np.int32)
    for index in np.ndindex(shape[:-2]):
        matrix = np.array(arr[index], dtype=arr.dtype, order="F", copy=True)
        ipiv = np.empty(n, dtype=np.int64)
        nn = LapackInt(n)
        lda = LapackInt(max(1, n))
        info = LapackInt(0)
        lwork = LapackInt(-1)
        query = np.empty(1, dtype=arr.dtype)
        uplo = ctypes.create_string_buffer(b"L")
        before = matrix.copy(order="F")
        fn(
            uplo,
            ctypes.byref(nn),
            matrix.ctypes.data_as(ctypes.POINTER(F)),
            ctypes.byref(lda),
            ipiv.ctypes.data_as(ctypes.POINTER(LapackInt)),
            query.ctypes.data_as(ctypes.POINTER(F)),
            ctypes.byref(lwork),
            ctypes.byref(info),
            ctypes.c_size_t(1),
        )
        if info.value != 0 or not np.array_equal(matrix, before, equal_nan=True):
            raise ValueError("Genuine SYTRF workspace query failed or mutated input")
        size = max(1, int(query[0]))
        work = np.empty(size, dtype=arr.dtype)
        lwork = LapackInt(size)
        fn(
            uplo,
            ctypes.byref(nn),
            matrix.ctypes.data_as(ctypes.POINTER(F)),
            ctypes.byref(lda),
            ipiv.ctypes.data_as(ctypes.POINTER(LapackInt)),
            work.ctypes.data_as(ctypes.POINTER(F)),
            ctypes.byref(lwork),
            ctypes.byref(info),
            ctypes.c_size_t(1),
        )
        if info.value < 0:
            raise ValueError(
                "Genuine SYTRF reported illegal argument " + str(-info.value)
            )
        if np.any(ipiv > np.iinfo(np.int32).max) or np.any(
            ipiv < np.iinfo(np.int32).min
        ):
            raise ValueError("LAPACK pivots do not fit originalTorchInt32 ABI")
        # SYTRF leaves the unused upper input triangle untouched; Torch returns
        # canonical lower-packed LD with that unused triangle zero. Keep all actual
        # lower/D block entries, signed pivots and INFO exactly as the native call.
        LD[index] = np.tril(matrix)
        pivots[index] = ipiv.astype(np.int32)
        infos[index] = info.value
    return LD, pivots, infos


def _cpu_ldl_missing_lapack(inp, *, hermitian=False):
    if inp.device.type != "cpu" or inp.requires_grad:
        raise RuntimeError("The CPU LAPACK fallback requires a non-autograd CPU input")
    ld, pivots, info = _numpy_lapack_factor_ex(
        inp.detach().numpy(), hermitian=hermitian
    )
    return torch.from_numpy(ld), torch.from_numpy(pivots), torch.from_numpy(info)


def _reference_ldl_factor(inp, *, hermitian=False):
    try:
        return torch.linalg.ldl_factor(inp, hermitian=hermitian)
    except RuntimeError as exc:
        if (
            inp.device.type != "cpu"
            or "requires compiling PyTorch with LAPACK" not in str(exc)
        ):
            raise
    ld, pivots, info = _cpu_ldl_missing_lapack(inp, hermitian=hermitian)
    if torch.any(info != 0).item():
        raise torch.linalg.LinAlgError(
            "CPU SYTRF reported a singular factorization: " + str(info.tolist())
        )
    return ld, pivots


def _reference_ldl_factor_ex(inp, *, hermitian=False):
    try:
        return torch.linalg.ldl_factor_ex(inp, hermitian=hermitian)
    except RuntimeError as exc:
        if (
            inp.device.type != "cpu"
            or "requires compiling PyTorch with LAPACK" not in str(exc)
        ):
            raise
    return _cpu_ldl_missing_lapack(inp, hermitian=hermitian)


LDL_FACTOR_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]
LDL_FACTOR_DTYPES = [torch.float32, torch.float64]


def _make_symmetric_positive_definite(shape, dtype, device):
    torch.manual_seed(0)
    n = shape[-1]
    # These backends cannot construct a faithful FP64 SPD matrix on device.
    factory_device = (
        "cpu"
        if dtype == torch.float64 and flag_gems.vendor_name in ("ascend", "iluvatar")
        else device
    )
    inp = torch.randn(shape, dtype=dtype, device=factory_device)
    inp = inp @ inp.transpose(-2, -1) + n * torch.eye(
        n, dtype=dtype, device=factory_device
    )
    return inp.to(device)


def _skip_unsupported_case(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin cannot preserve float64 input storage")
    if (
        flag_gems.vendor_name == "iluvatar"
        and dtype == torch.float64
        and not utils.TO_CPU
    ):
        pytest.skip("Iluvatar native ldl_factor does not support float64")


def _assert_factorization_equal(result, reference, dtype, expected_device):
    result_ld, result_pivots = result
    reference_ld, reference_pivots = reference
    assert result_ld.device == expected_device
    assert result_pivots.device == expected_device
    assert result_ld.shape == reference_ld.shape
    assert result_ld.dtype == dtype
    assert result_pivots.shape == reference_pivots.shape
    assert result_pivots.dtype == torch.int32
    if flag_gems.vendor_name == "mthreads":
        result_ld, result_pivots = result_ld.cpu(), result_pivots.cpu()
    utils.gems_assert_close(result_ld, reference_ld, dtype)
    utils.gems_assert_equal(result_pivots, reference_pivots)


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
@pytest.mark.parametrize("shape", LDL_FACTOR_SHAPES)
@pytest.mark.parametrize("dtype", LDL_FACTOR_DTYPES)
@pytest.mark.parametrize("hermitian", [False, True])
def test_linalg_ldl_factor(shape, dtype, hermitian):
    _skip_unsupported_case(dtype)
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Not supported float64")
    inp = _make_symmetric_positive_definite(shape, dtype, flag_gems.device)
    inp_before = inp.clone()
    # An existing FP64 input must not be narrowed by the backend upcast policy.
    reference_input = inp.cpu() if flag_gems.vendor_name == "mthreads" else inp
    ref_inp = utils.to_reference(reference_input, dtype != torch.float64)

    reference = _reference_ldl_factor(ref_inp, hermitian=hermitian)
    result = flag_gems.ldl_factor(inp, hermitian=hermitian)

    _assert_factorization_equal(result, reference, dtype, inp.device)
    assert inp.device == inp_before.device
    if flag_gems.vendor_name == "mthreads":
        assert inp.dtype == inp_before.dtype
        utils.gems_assert_equal(inp.cpu(), inp_before.cpu())
    else:
        utils.gems_assert_equal(inp, utils.to_reference(inp_before, False))


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
@pytest.mark.parametrize("shape", [(2, 4, 4), (3, 8, 8)])
@pytest.mark.parametrize("dtype", LDL_FACTOR_DTYPES)
def test_linalg_ldl_factor_batched(shape, dtype):
    _skip_unsupported_case(dtype)
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Not supported float64")
    inp = _make_symmetric_positive_definite(shape, dtype, flag_gems.device)
    reference_input = inp.cpu() if flag_gems.vendor_name == "mthreads" else inp
    ref_inp = utils.to_reference(reference_input, dtype != torch.float64)

    reference = _reference_ldl_factor(ref_inp)
    result = flag_gems.ldl_factor(inp)

    _assert_factorization_equal(result, reference, dtype, inp.device)


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
@pytest.mark.parametrize(
    "shape,dtype,error_type,message",
    [
        ((8,), torch.float32, ValueError, "at least 2D"),
        ((4, 5), torch.float32, ValueError, "square"),
        ((4, 4), torch.int32, TypeError, "float32 and float64"),
        ((65, 65), torch.float32, ValueError, "exceeds maximum"),
    ],
)
def test_linalg_ldl_factor_rejects_invalid_inputs(shape, dtype, error_type, message):
    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)

    with pytest.raises(error_type, match=message):
        flag_gems.ldl_factor(inp)
