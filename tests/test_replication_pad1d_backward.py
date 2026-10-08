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


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize("shape", REPLICATION_PAD1D_BACKWARD_SHAPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD1D_BACKWARD_PADDING)
@pytest.mark.parametrize("dtype", REPLICATION_PAD1D_BACKWARD_DTYPES)
def test_replication_pad1d_backward(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    padded_out = torch.ops.aten.replication_pad1d(inp, padding)
    grad_output = torch.ones_like(padded_out)
    ref_grad = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.replication_pad1d_backward(ref_grad, ref_inp, padding)
    res_out = flag_gems.replication_pad1d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward_grad_output_dtype_mismatch():
    """A gradient whose dtype differs from the input is rejected, as in ATen.

    ATen raises "expected scalar type Double but found Float" rather than
    promoting, and its output takes the *input's* options, so silently following
    grad_output would hand back a differently-typed gradient.
    """
    inp = torch.randn(2, 3, 8, dtype=torch.float64, device=flag_gems.device)
    padded = torch.ops.aten.replication_pad1d(inp, (1, 1))
    bad_grad = torch.ones_like(padded, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="expected scalar type"):
        flag_gems.replication_pad1d_backward(bad_grad, inp, (1, 1))


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replication_pad1d_backward_grad_input_takes_input_options(dtype):
    """grad_input has the input's dtype and device, matching at::empty_like(self)."""
    inp = torch.randn(2, 3, 8, dtype=dtype, device=flag_gems.device)
    padded = torch.ops.aten.replication_pad1d(inp, (2, 1))
    grad = torch.ones_like(padded)
    res = flag_gems.replication_pad1d_backward(grad, inp, (2, 1))
    assert res.dtype == inp.dtype
    assert res.device == inp.device
    assert tuple(res.shape) == tuple(inp.shape)
