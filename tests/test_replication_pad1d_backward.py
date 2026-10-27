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

# 2D/3D shapes covering small to medium width; pad1d operates on last dim
REPLICATION_PAD1D_BACKWARD_SHAPES = [(2, 3, 8), (4, 16, 64), (1, 5, 16), (32, 256)]
# Asymmetric and symmetric padding combinations, including the negative padding
# that crops rather than replicates. Any combination is accepted as long as the
# resulting width stays positive: ATen's forward rejects a padding only when
# W_out <= 0, so (-2, -2) is valid for the widths used here (W >= 8 gives
# W_out >= 4). The error message quoted elsewhere ("input (W: 4) is too small.
# Calculated output W: 0") comes from W=4, which these shapes never use.
REPLICATION_PAD1D_BACKWARD_PADDING = [
    (1, 1),
    (0, 2),
    (2, 1),
    (1, 2),
    (-2, 1),
    (1, -2),
    (-1, 0),
    (0, -1),
    (-1, -1),
    (-2, -2),
    (-3, 2),
]

# ATen supports fp64 and complex64/complex128 natively on CUDA for this operator.
# utils.COMPLEX_DTYPES is not used as-is because it includes complex32, which
# ATen's own forward rejects ('"replication_pad1d" not implemented for
# ComplexHalf'), so there would be no reference to compare against.
REPLICATION_PAD1D_BACKWARD_DTYPES = utils.ALL_FLOAT_DTYPES + [
    torch.complex64,
    torch.complex128,
]


def _to_input_device(value):
    if flag_gems.vendor_name == "kunlunxin" and value.dtype in (
        torch.float64,
        torch.complex128,
    ):
        return (
            value.contiguous()
            .view(torch.uint8)
            .to(flag_gems.device)
            .view(value.dtype)
            .reshape(value.shape)
        )
    return value.to(flag_gems.device)


def _make_input(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float64,
        torch.complex128,
    ):
        return _to_input_device(torch.randn(shape, dtype=dtype, device="cpu"))
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _reference_input(value):
    if (
        utils.TO_CPU
        and flag_gems.vendor_name == "kunlunxin"
        and value.dtype in (torch.float64, torch.complex128)
    ):
        return (
            value.contiguous()
            .view(torch.uint8)
            .cpu()
            .view(value.dtype)
            .reshape(value.shape)
        )
    return utils.to_reference(value)


def _reference_backward(grad, inp, padding):
    if (
        flag_gems.vendor_name == "iluvatar"
        and inp.device.type != "cpu"
        and inp.dtype == torch.complex128
    ):
        pytest.skip(
            "Iluvatar native complex128 replication_pad1d_backward computes incorrect values; CPU reference remains covered"
        )
    try:
        return torch.ops.aten.replication_pad1d_backward(grad, inp, padding)
    except ValueError as error:
        if (
            flag_gems.vendor_name == "kunlunxin"
            and inp.device.type != "cpu"
            and (
                (
                    min(padding) < 0
                    and str(error).startswith("[INVALID PARAMETER]: error code= 1(")
                )
                or (
                    inp.dtype in (torch.complex64, torch.complex128)
                    and str(error).startswith("[NOT IMPLEMENTED]: error code= 4(")
                )
            )
        ):
            pytest.skip("Kunlunxin native backward does not support this padding/dtype")
        raise


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize("shape", REPLICATION_PAD1D_BACKWARD_SHAPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD1D_BACKWARD_PADDING)
@pytest.mark.parametrize("dtype", REPLICATION_PAD1D_BACKWARD_DTYPES)
def test_replication_pad1d_backward(shape, padding, dtype):
    if (
        flag_gems.vendor_name == "ascend"
        and not utils.TO_CPU
        and dtype == torch.bfloat16
        and any(p < 0 for p in padding)
    ):
        pytest.skip("Ascend native BF16 negative-padding backward fails in PadV3Grad")
    if dtype.is_complex and flag_gems.vendor_name == "ascend":
        pytest.skip("Ascend randn does not support complex input construction")
    if dtype.is_complex and flag_gems.vendor_name == "mthreads":
        pytest.skip("MUSA replication_pad1d input preparation does not support complex")
    inp = _make_input(shape, dtype)
    assert inp.dtype == dtype
    ref_inp = _reference_input(inp)

    grad_shape = (*shape[:-1], shape[-1] + sum(padding))
    grad_output = _to_input_device(
        torch.ones(grad_shape, dtype=inp.dtype, device="cpu")
    )
    ref_grad = _reference_input(grad_output)

    ref_out = _reference_backward(ref_grad, ref_inp, padding)
    res_out = flag_gems.replication_pad1d_backward(grad_output, inp, padding)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.complex128:
        assert res_out.device == inp.device
        assert res_out.dtype == dtype
        assert res_out.shape == inp.shape
        result_cpu = (
            res_out.contiguous()
            .view(torch.uint8)
            .cpu()
            .view(dtype)
            .reshape(res_out.shape)
        )
        flag_gems.testing.assert_close(result_cpu, ref_out, dtype)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward_grad_output_dtype_mismatch():
    """A gradient whose dtype differs from the input is rejected, as in ATen.

    ATen raises "expected scalar type Double but found Float" rather than
    promoting, and its output takes the *input's* options, so silently following
    grad_output would hand back a differently-typed gradient.
    """
    inp = _make_input((2, 3, 8), torch.float64)
    assert inp.dtype == torch.float64
    bad_grad = torch.ones(2, 3, 10, dtype=torch.float32, device=inp.device)
    with pytest.raises(RuntimeError):
        flag_gems.replication_pad1d_backward(bad_grad, inp, (1, 1))


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replication_pad1d_backward_grad_input_takes_input_options(dtype):
    """grad_input has the input's dtype and device, matching at::empty_like(self)."""
    inp = _make_input((2, 3, 8), dtype)
    assert inp.dtype == dtype
    grad = _to_input_device(torch.ones(2, 3, 11, dtype=inp.dtype, device="cpu"))
    res = flag_gems.replication_pad1d_backward(grad, inp, (2, 1))
    assert res.dtype == inp.dtype
    assert res.device == inp.device
    assert tuple(res.shape) == tuple(inp.shape)
