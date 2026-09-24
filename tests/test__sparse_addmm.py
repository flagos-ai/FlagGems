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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import test_utils as tu

# pytest.mark cannot build a marker for an underscore-prefixed name through
# attribute access, so register it explicitly.
setattr(
    pytest.mark,
    "_sparse_addmm",
    MarkDecorator(Mark("_sparse_addmm", (), {}, _ispytest=True), _ispytest=True),
)

# Dtypes the native kernel accepts, probed with valid calls on the active NVIDIA
# backend: float32, float64, complex64 (complex128 is accepted as well, but it is
# not one of the spec dtypes). The spec dtypes it rejects stay covered by the
# negative test below:
#   float16 -> 'addmm_sparse_cuda' not implemented for 'Half'
#   bfloat16 -> the same message for 'BFloat16'
#   int8 'Char' / uint8 'Byte' / int32 'Int' / int64 'Long' / bool 'Bool' ->
#       the same 'addmm_sparse_cuda' message
#   float8_e4m3fn / float8_e5m2 -> 'coalesce_sparse_cuda' not implemented for
#       'Float8_e4m3fn' / 'Float8_e5m2' (a COO assembled from explicit indices
#       and ones values is constructed fine; the operator coalesces the operand)
# float64 is gated by a static capability flag, so collecting this file probes
# nothing and allocates nothing.
_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)
_DTYPES.append(torch.complex64)

_REAL_DTYPES = [dtype for dtype in _DTYPES if not dtype.is_complex]

# _sparse_addmm(self, mat1, mat2) is inherently 2-D: the native kernel requires
# mat1.sparse_dim() == 2 and mat2.dim() == 2, and self is expanded to
# (mat1.size(0), mat2.size(1)). The spec's 0-D/1-D/3-D/4-D/5-D shapes cannot
# describe a 2-D sparse operand, so each spec shape level is mapped to an
# (M, N, K) triple of the same size scale. The last three rows are the zero-extent
# boundaries: M == 0, K == 0 and N == 0 (probe: each returns the correctly shaped
# empty or sparse-only result instead of raising).
_MNK_SHAPES = [
    (1, 1, 1),
    (256, 256, 256),
    (1024, 1024, 1024),
    (20, 320, 15),
    (16, 128, 64),
    (16, 7, 57),
    (64, 1, 32),
    (0, 4, 3),
    (4, 3, 0),
    (4, 0, 3),
]

# Quick mode keeps a single triple, mirroring the spec's (2, 19, 7) quick shape.
_MNK_ROWS = tu.selected_cases(_MNK_SHAPES, quick=[(2, 19, 7)])


def _sparse_mat1(dense):
    """2-D COO twin of ``dense`` - the sparse operand of the operator."""
    return dense.to_sparse()


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("mnk", _MNK_ROWS)
def test__sparse_addmm_value_grid(mnk, value_range, dtype):
    # beta/alpha are omitted on purpose: their schema defaults (1, 1) are part of
    # the public call contract, and the sweeps further down pass them explicitly.
    m, n, k = mnk
    self_t = tu.make_input(dtype, (m, n), value_range)
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), value_range))
    mat2 = tu.make_input(dtype, (k, n), value_range)

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2)
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)

    tu.assert_result_close(res_out, ref_out)


# Dense operands may arrive as views; the kernel reads them as plain dense
# tensors, so a non-contiguous stride or a non-zero storage offset must not
# change the result.
_STRIDED_MNK = (20, 320, 15)
_STRIDED_ROWS = tu.selected_cases(
    ["transposed_self", "transposed_mat2", "offset_mat2"],
    quick=[],
)


def _strided_operands(label, dtype):
    """Dense self/mat2 carrying a non-trivial stride or storage offset.

    transposed_self - self is a transposed view, stride (1, M)
    transposed_mat2 - mat2 is a transposed view, stride (1, K)
    offset_mat2     - mat2 is a row slice of a larger buffer, storage offset 3 * N
    """
    m, n, k = _STRIDED_MNK
    value_range = ["-1", "1"]
    self_t = tu.make_input(dtype, (m, n), value_range)
    mat2 = tu.make_input(dtype, (k, n), value_range)
    if label == "transposed_self":
        self_t = tu.make_input(dtype, (n, m), value_range).t()
    elif label == "transposed_mat2":
        mat2 = tu.make_input(dtype, (n, k), value_range).t()
    else:
        mat2 = tu.make_input(dtype, (k + 3, n), value_range)[3:]
    return self_t, mat2


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("label", _STRIDED_ROWS)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test__sparse_addmm_strided_dense_operands(label, dtype):
    m, n, k = _STRIDED_MNK
    self_t, mat2 = _strided_operands(label, dtype)
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), ["-1", "1"]))

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2)
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)

    tu.assert_result_close(res_out, ref_out)


# Sparse operands whose state differs from a fresh dense-to-COO conversion: the
# stored values of duplicate coordinates, explicitly stored zeros, an empty
# index list, and compressed-row storage.
_STATE_MNK = (16, 12, 8)
_SPARSE_STATE_ROWS = tu.selected_cases(
    ["uncoalesced_duplicates", "stored_zeros", "zero_nnz", "csr"],
    quick=[],
)


def _sparse_state_mat1(label, shape, dtype):
    """Sparse operand in a state a fresh ``to_sparse()`` never produces."""
    m, k = shape
    if label == "uncoalesced_duplicates":
        # Duplicate (row, col) entries left uncoalesced: the values stored at one
        # coordinate have to be summed.
        indices = torch.tensor(
            [[0, 0, 1, 3, 3, 3], [0, 0, 2, 1, 1, 5]],
            dtype=torch.int64,
            device=flag_gems.device,
        )
        values = torch.tensor(
            [0.5, -1.5, 2.0, 0.25, 0.75, -1.0],
            dtype=dtype,
            device=flag_gems.device,
        )
        return torch.sparse_coo_tensor(
            indices, values, (m, k), dtype=dtype, device=flag_gems.device
        )
    if label == "stored_zeros":
        # Explicitly stored zeros, including a stored -0.0.
        indices = torch.tensor(
            [[0, 1, 2, 4, 5, 7], [1, 0, 3, 2, 7, 4]],
            dtype=torch.int64,
            device=flag_gems.device,
        )
        values = torch.tensor(
            [0.0, 1.5, -0.0, 2.5, 0.0, -3.5], dtype=dtype, device=flag_gems.device
        )
        return torch.sparse_coo_tensor(
            indices, values, (m, k), dtype=dtype, device=flag_gems.device
        ).coalesce()
    if label == "zero_nnz":
        # No stored values at all: only the dense addend contributes.
        indices = torch.empty((2, 0), dtype=torch.int64, device=flag_gems.device)
        values = torch.empty(0, dtype=dtype, device=flag_gems.device)
        return torch.sparse_coo_tensor(
            indices, values, (m, k), dtype=dtype, device=flag_gems.device
        )
    dense = tu.make_input(dtype, (m, k), ["-1", "1"]).to_sparse()
    return dense.coalesce().to_sparse_csr()


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("label", _SPARSE_STATE_ROWS)
def test__sparse_addmm_sparse_operand_state(label):
    dtype = torch.float32
    m, n, k = _STATE_MNK
    self_t = tu.make_input(dtype, (m, n), ["-1", "1"])
    mat1 = _sparse_state_mat1(label, (m, k), dtype)
    mat2 = tu.make_input(dtype, (k, n), ["-1", "1"])

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2)
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)

    tu.assert_result_close(res_out, ref_out)


# self is expanded to (mat1.size(0), mat2.size(1)), so the broadcast patterns are
# expressed through self's shape: a 1-D self (which gains a leading 1), a self
# with a 1-sized row and a 1-sized column, and the scalar form. The scalar form is
# a 0-dim tensor: the schema types self as Tensor, so a Python float is rejected
# by the dispatcher ('Expected a value of type Tensor for argument self') while a
# 0-dim tensor broadcasts to the full result (probe). Sizes come from the spec's
# large shapes. dtype float32: none of the spec's broadcast dtype candidates
# (bf16/fp16/int32/int64/int8/uint8/fp8_e4m3) has a native sparse addmm kernel on
# this backend (see _DTYPES).
_BROADCAST_ROWS = tu.selected_cases(
    [
        ((1024, 1024, 1024), (1024,)),
        ((20, 320, 15), (1, 320)),
        ((16, 128, 64), (16, 1)),
        ((16, 7, 57), ()),
    ],
    quick=[],
)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("mnk,self_shape", _BROADCAST_ROWS)
def test__sparse_addmm_self_broadcast(mnk, self_shape):
    m, n, k = mnk
    dtype = torch.float32
    self_t = tu.make_input(dtype, self_shape, ["-1", "1"])
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), ["-1", "1"]))
    mat2 = tu.make_input(dtype, (k, n), ["-1", "1"])

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2)
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)

    tu.assert_result_close(res_out, ref_out)


# Parameter coverage: the value grid omits beta/alpha (schema defaults), these
# sweeps pass explicit scalars. beta scales the dense addend elementwise, so its
# nan/inf boundary is order-independent; alpha multiplies the reduced result and
# its +/-inf boundary is asserted separately on a single-sign operand pair.
_PARAM_MNK = (20, 320, 15)
_BETA_VALUES = (0, 1, -1, 2, 0.5, -2.5, 1e20, float("nan"), float("inf"), float("-inf"))
_ALPHA_VALUES = (0, 1, -1, 2, 0.5, -2.5, 1e20, float("nan"))

_BETA_ROWS = tu.selected_cases(
    [(dtype, beta) for dtype in _REAL_DTYPES for beta in _BETA_VALUES], quick=[]
)
_ALPHA_ROWS = tu.selected_cases(
    [(dtype, alpha) for dtype in _REAL_DTYPES for alpha in _ALPHA_VALUES], quick=[]
)
_ALPHA_INF_ROWS = tu.selected_cases(
    [
        (dtype, alpha)
        for dtype in _REAL_DTYPES
        for alpha in (float("inf"), float("-inf"))
    ],
    quick=[],
)


def _param_operands(dtype, value_range):
    m, n, k = _PARAM_MNK
    self_t = tu.make_input(dtype, (m, n), value_range)
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), value_range))
    mat2 = tu.make_input(dtype, (k, n), value_range)
    return self_t, mat1, mat2


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype,beta", _BETA_ROWS)
def test__sparse_addmm_beta_values(dtype, beta):
    self_t, mat1, mat2 = _param_operands(dtype, ["-1", "1"])

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t),
        tu.to_reference(mat1),
        tu.to_reference(mat2),
        beta=beta,
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2, beta=beta)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype,alpha", _ALPHA_ROWS)
def test__sparse_addmm_alpha_values(dtype, alpha):
    self_t, mat1, mat2 = _param_operands(dtype, ["-1", "1"])

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t),
        tu.to_reference(mat1),
        tu.to_reference(mat2),
        alpha=alpha,
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype,alpha", _ALPHA_INF_ROWS)
def test__sparse_addmm_alpha_inf_boundary(dtype, alpha):
    # alpha = +/-Inf is only order-independent when the reduced operand has a
    # single sign (probe: alpha=inf over signed input produced 51 NaN and 21 Inf
    # in a 72-element output, because +Inf and -Inf terms cancel in summation
    # order), so this boundary is asserted on a non-negative operand pair.
    self_t, mat1, mat2 = _param_operands(dtype, ["0", "1"])

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t),
        tu.to_reference(mat1),
        tu.to_reference(mat2),
        alpha=alpha,
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2, alpha=alpha)

    tu.assert_result_close(res_out, ref_out)


# Special values, following the shared generator's contract. Each payload is
# placed so that exactly one special value reaches an output entry, which keeps
# +Inf/-Inf from cancelling inside an accumulation of unspecified order.
_SPECIAL_MNK = (20, 320, 15)
_SPECIAL_DTYPES = list(_DTYPES)

# tu.special_value_cases enumerates floating dtypes only, and complex64 is a
# supported positive dtype of this operator, so its scenarios are appended with
# the same payload contract that make_special_input uses (the float32 payload cast
# to the target dtype) rather than dropping the dtype.
_SPECIAL_ROWS = tu.special_value_cases(_SPECIAL_DTYPES)
_SPECIAL_ROWS += tu.selected_cases(
    [
        (dtype, scenario)
        for dtype in _SPECIAL_DTYPES
        if not dtype.is_floating_point
        for scenario in ("nan", "inf", "mixed")
    ],
    quick=[],
)
_SPECIAL_ROWS = tu.selected_cases(_SPECIAL_ROWS, quick=[])


def _with_leading_special_values(tensor, scenario):
    """Write the shared NaN/Inf payload into the leading elements."""
    payload = tu.make_special_input(tensor.dtype, scenario)
    tensor.reshape(-1)[: payload.numel()] = payload
    return tensor


def _diagonal_special_coo(payload, shape, dtype):
    """COO holding ``payload`` on its diagonal, zero elsewhere."""
    m, k = shape
    size = min(m, k, payload.numel())
    idx = torch.arange(size, dtype=torch.int64, device=flag_gems.device)
    values = torch.zeros(size, dtype=dtype, device=flag_gems.device)
    values[:size] = payload[:size]
    return torch.sparse_coo_tensor(
        torch.stack([idx, idx]), values, (m, k), dtype=dtype, device=flag_gems.device
    ).coalesce()


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test__sparse_addmm_special_values_in_self(dtype, scenario):
    m, n, k = _SPECIAL_MNK
    # self is added elementwise, so every scenario propagates unchanged.
    self_t = _with_leading_special_values(
        tu.make_input(dtype, (m, n), ["-1", "1"]), scenario
    )
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), ["-1", "1"]))
    mat2 = tu.make_input(dtype, (k, n), ["-1", "1"])

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2)
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test__sparse_addmm_special_values_in_sparse_operand(dtype, scenario):
    m, n, k = _SPECIAL_MNK
    self_t = tu.make_input(dtype, (m, n), ["-1", "1"])
    # The payload sits on the diagonal and mat2 is a ones matrix, so each output
    # entry accumulates exactly one special stored value: +Inf and -Inf stay
    # separated instead of cancelling inside one sum.
    payload = tu.make_special_input(dtype, scenario)
    mat1 = _diagonal_special_coo(payload, (m, k), dtype)
    mat2 = torch.ones((k, n), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2)
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test__sparse_addmm_special_values_in_mat2(dtype, scenario):
    m, n, k = _SPECIAL_MNK
    self_t = tu.make_input(dtype, (m, n), ["-1", "1"])
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), ["-1", "1"]))
    # The payload occupies mat2's first row (one entry per output column), so
    # every output entry mixes exactly one special value with finite products and
    # its NaN-ness does not depend on the summation order.
    mat2 = _with_leading_special_values(
        tu.make_input(dtype, (k, n), ["-1", "1"]), scenario
    )

    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2)
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test__sparse_addmm_beta_zero_drops_special_self(dtype, scenario):
    m, n, k = _SPECIAL_MNK
    self_t = _with_leading_special_values(
        tu.make_input(dtype, (m, n), ["-1", "1"]), scenario
    )
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), ["-1", "1"]))
    mat2 = tu.make_input(dtype, (k, n), ["-1", "1"])

    # beta == 0 removes self from the product, so the native result is finite even
    # though self holds NaN/Inf; the shared comparison rejects any unexpected
    # NaN/Inf the candidate would leave behind.
    ref_out = torch.ops.aten._sparse_addmm(
        tu.to_reference(self_t), tu.to_reference(mat1), tu.to_reference(mat2), beta=0
    )
    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2, beta=0)

    tu.assert_result_close(res_out, ref_out)


# Gradient coverage: dL/dself is the upstream gradient, dL/dmat2 = mat1^T @ up and
# dL/dmat1 = up @ mat2^T is the sparse operand's own gradient. The native
# gradient of a sparse operand is a coalesced COO tensor (probe: layout
# torch.sparse_coo, shape (M, K), nnz equal to the stored nnz of mat1 for
# float32 / float64 / complex64), which the shared comparison handles directly.
_BACKWARD_ROWS = tu.selected_cases(
    [(mnk, dtype) for mnk in ((20, 320, 15), (16, 128, 64)) for dtype in _DTYPES],
    quick=[],
)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("mnk,dtype", _BACKWARD_ROWS)
def test__sparse_addmm_backward(mnk, dtype):
    m, n, k = mnk
    value_range = ["-1", "1"]
    # Candidate operands are ordinary leaves on the candidate device; the
    # reference operands below are independent copies of the same values, so the
    # two autograd graphs share no storage.
    self_t = tu.make_input(dtype, (m, n), value_range).requires_grad_(True)
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), value_range)).requires_grad_(True)
    mat2 = tu.make_input(dtype, (k, n), value_range).requires_grad_(True)
    upstream = tu.make_input(dtype, (m, n), value_range)

    ref_self = tu.to_reference(self_t).detach().requires_grad_(True)
    ref_mat1 = tu.to_reference(mat1).detach().requires_grad_(True)
    ref_mat2 = tu.to_reference(mat2).detach().requires_grad_(True)

    res_out = flag_gems._sparse_addmm(self_t, mat1, mat2)
    ref_out = torch.ops.aten._sparse_addmm(
        ref_self, ref_mat1, ref_mat2, beta=1, alpha=1
    )

    # Comparing the forward results first also proves both operand sets hold the
    # same values, so the gradients below describe the same workload.
    tu.assert_result_close(res_out, ref_out)

    res_dself, res_dmat1, res_dmat2 = torch.autograd.grad(
        res_out, [self_t, mat1, mat2], grad_outputs=upstream
    )
    ref_dself, ref_dmat1, ref_dmat2 = torch.autograd.grad(
        ref_out,
        [ref_self, ref_mat1, ref_mat2],
        grad_outputs=tu.to_reference(upstream),
    )

    tu.assert_result_close(res_dself, ref_dself)
    tu.assert_result_close(res_dmat2, ref_dmat2)
    tu.assert_result_close(res_dmat1, ref_dmat1)


_OUT_ROWS = tu.selected_cases(_DTYPES, quick=[])


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype", _OUT_ROWS)
def test__sparse_addmm_out_overload(dtype):
    # aten::_sparse_addmm.out is a real native overload on this backend (probe:
    # it returns the caller's buffer), so both paths call it directly instead of
    # simulating it with default + copy_.
    m, n, k = (8, 9, 6)
    self_t = tu.make_input(dtype, (m, n), ["-1", "1"])
    mat1 = _sparse_mat1(tu.make_input(dtype, (m, k), ["-1", "1"]))
    mat2 = tu.make_input(dtype, (k, n), ["-1", "1"])

    ref_self = tu.to_reference(self_t)
    ref_out = torch.full((m, n), 7, dtype=ref_self.dtype, device=ref_self.device)
    torch.ops.aten._sparse_addmm.out(
        ref_self, tu.to_reference(mat1), tu.to_reference(mat2), out=ref_out
    )

    out = torch.full((m, n), 7, dtype=self_t.dtype, device=self_t.device)
    res_ret = flag_gems._sparse_addmm(self_t, mat1, mat2, out=out)

    assert res_ret is out
    tu.assert_result_close(out, ref_out)


# The spec dtypes the native operator rejects (see _DTYPES) stay covered here: the
# candidate must reject them too. Two static conditions apply to the list, so no
# runtime probe or skip is involved: the exact rejected set is a property of the
# measured vendor, and a dtype is only listed when this build can construct its
# operands at all (flag_gems.runtime.device support flags). The COO is built from
# explicit indices so the construction itself needs no per-dtype CUDA kernel.
_NATIVE_REJECTED_DTYPES = [
    (torch.bool, None),
    (torch.int8, None),
    (torch.uint8, None),
    (torch.int32, None),
    (torch.float16, None),
    (torch.int64, flag_gems.runtime.device.support_int64),
    (torch.bfloat16, flag_gems.runtime.device.support_bf16),
    (torch.float8_e4m3fn, flag_gems.runtime.device.support_fp8),
    (torch.float8_e5m2, flag_gems.runtime.device.support_fp8),
]

_VENDOR_REJECTED_DTYPES = (
    [dtype for dtype, gated in _NATIVE_REJECTED_DTYPES if gated is None or gated]
    if flag_gems.runtime.device.vendor_name == "nvidia"
    else []
)


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("dtype", _VENDOR_REJECTED_DTYPES)
def test__sparse_addmm_rejects_unsupported_dtype(dtype):
    m, n, k = (8, 9, 6)
    indices = torch.tensor(
        [[0, 1, 2], [0, 3, 5]], dtype=torch.int64, device=flag_gems.device
    )
    values = torch.ones(3, dtype=dtype, device=flag_gems.device)
    self_t = torch.ones((m, n), dtype=dtype, device=flag_gems.device)
    mat1 = torch.sparse_coo_tensor(
        indices, values, (m, k), dtype=dtype, device=flag_gems.device
    )
    mat2 = torch.ones((k, n), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_addmm(self_t, mat1, mat2)


# mat1 must be a 2-D sparse operand, mat2 must be 2-D and self must broadcast to
# (mat1.size(0), mat2.size(1)). Every row is rejected by the native dispatcher or
# kernel with a RuntimeError (probe: 'The expanded size ... must match', 'addmm:
# Argument #3 (dense): Expected dim 0 size', 'addmm: 2D tensor expected', ...), so
# only RuntimeError is expected here; a missing candidate raises AttributeError,
# which is not accepted.
_INVALID_OPERAND_SHAPES = [
    ((9, 9), (8, 6), (6, 9)),
    ((8, 9), (8, 6), (7, 9)),
    ((8, 9), (8, 6), (6, 9, 1)),
    ((2, 8, 9), (8, 6), (6, 9)),
    ((8, 9), (2, 8, 6), (6, 9)),
]


@pytest.mark._sparse_addmm
@pytest.mark.parametrize("self_shape,mat1_shape,mat2_shape", _INVALID_OPERAND_SHAPES)
def test__sparse_addmm_rejects_invalid_operand_shapes(
    self_shape, mat1_shape, mat2_shape
):
    dtype = torch.float32
    self_t = torch.ones(self_shape, dtype=dtype, device=flag_gems.device)
    mat2 = torch.ones(mat2_shape, dtype=dtype, device=flag_gems.device)
    indices = torch.zeros(
        (len(mat1_shape), 1), dtype=torch.int64, device=flag_gems.device
    )
    values = torch.ones(1, dtype=dtype, device=flag_gems.device)
    mat1 = torch.sparse_coo_tensor(
        indices, values, mat1_shape, dtype=dtype, device=flag_gems.device
    )

    with pytest.raises(RuntimeError):
        flag_gems._sparse_addmm(self_t, mat1, mat2)
