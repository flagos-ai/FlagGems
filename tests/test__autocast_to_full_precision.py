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


"""Correctness tests for ``aten::_autocast_to_full_precision``.

``aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool
cpu_enabled) -> Tensor(a)`` promotes an fp16/bf16 input to fp32 when the flag
of the input's *own* device is set and otherwise returns the input itself (same
tensor object, same storage). It has a single ``default`` overload -- there is
no ``.out`` variant -- and it is value preserving, because the widening cast is
exact; results are therefore compared with the shared exact assertion, which
also pins the returned dtype to the dtype the native operator returned.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Static backend capability flags from the shared harness; they are derived
# from the device properties at import time and applied while the case lists
# are built, so no dtype is probed and nothing is skipped at run time.
_GATED_DTYPES = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}


def _gated(dtypes):
    """Keep the dtypes whose static capability flag is on."""
    return [dtype for dtype in dtypes if _GATED_DTYPES.get(dtype, True)]


# The spec's required dtypes plus fp64: bf16, fp64, int64 and both fp8 types are
# kept or dropped by the static flags above, the remaining types always pass.
DTYPES = _gated(
    [
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]
)


def _assert_alias_matches(inp, ref_inp, res_out, ref_out):
    """Check the candidate's alias and geometry relation against the native one.

    The native result defines the relation for this dtype/flag/device: it
    returns the input object when nothing is promoted and a new tensor when it
    promotes, and a promoted layout is backend defined, so the geometry is
    compared against the reference and only for a matching device kind. Empty
    tensors still pin the object identity; only their storage pointer is
    skipped, since an empty tensor has no usable one.
    """
    assert (res_out is inp) == (ref_out is ref_inp)
    if res_out is inp or ref_inp.device.type != inp.device.type:
        return
    if inp.numel():
        assert (res_out.data_ptr() == inp.data_ptr()) == (
            ref_out.data_ptr() == ref_inp.data_ptr()
        )
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test__autocast_to_full_precision(shape, value_range, dtype):
    # Both flags enabled is the accelerator-promoting configuration, and every
    # other dtype/device combination passes through; the expected dtype and
    # layout therefore come from the native result, not from a predicted branch.
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    cuda_enabled, cpu_enabled = True, True

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled, cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(inp, cuda_enabled, cpu_enabled)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches(inp, ref_inp, res_out, ref_out)


# Parameter-coverage geometry: the spec's large shapes; this supplement, like
# the flag sweep and the layout sweep below, is default-only, since the main
# grid already supplies the quick smoke subset.
PARAM_SHAPES = tu.selected_cases(
    [(1024, 1024), (20, 320, 15)],
    quick=[],
)

# The flag pair is the only argument that selects a branch, and only the two
# promotable dtypes ever read it: every other dtype short-circuits to the
# identity before the flags are consulted. The sweep therefore keeps both
# promotable dtypes plus fp32 as a non-promotable control, and leaves
# (True, True) to the main grid, which already covers that branch over the whole
# dtype/range/shape grid.
#
# Both sides receive the original flag pair. Native behaviour, measured: an
# accelerator fp16 input follows cuda_enabled, a CPU one follows cpu_enabled,
# and neither flag promotes elsewhere. The mixed pairs are therefore comparable
# only while the reference runs on the candidate's device, as the default test
# run does. Under ``--ref cpu`` they would compare an accelerator branch against
# a CPU branch, and that row cannot be compared as equal; it stays an
# unresolved local cross-device gap (recorded in the run's protocol-gap notes),
# not something to hide by rewriting flags, skipping cases or dropping
# same-device workloads.
FLAG_PAIRS = tu.selected_cases(
    [(True, False), (False, True), (False, False)],
    quick=[],
)
FLAG_DTYPES = _gated([torch.float16, torch.bfloat16, torch.float32])


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("dtype", FLAG_DTYPES)
@pytest.mark.parametrize("cuda_enabled,cpu_enabled", FLAG_PAIRS)
@pytest.mark.parametrize("shape", PARAM_SHAPES)
def test__autocast_to_full_precision_flags(shape, dtype, cuda_enabled, cpu_enabled):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled, cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(inp, cuda_enabled, cpu_enabled)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches(inp, ref_inp, res_out, ref_out)


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("dtype", FLAG_DTYPES)
@pytest.mark.parametrize("shape", PARAM_SHAPES)
def test__autocast_to_full_precision_keyword_flags(shape, dtype):
    # The schema names both flags and gives neither a default, so binding them
    # by keyword is a distinct call form from the positional sweep above and
    # must reach the same branch on both sides.
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    cuda_enabled, cpu_enabled = True, False

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled=cuda_enabled, cpu_enabled=cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(
        inp, cuda_enabled=cuda_enabled, cpu_enabled=cpu_enabled
    )

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches(inp, ref_inp, res_out, ref_out)


# Zero-element inputs keep the promotion decision -- a promoted empty tensor is
# a new object, a pass-through is the input object -- so they cover the aliasing
# rule at the size boundary as well. One such case stays in the quick subset.
EMPTY_CASES = tu.selected_cases(
    [(shape, dtype) for shape in [(0,), (2, 0, 3)] for dtype in DTYPES],
    quick=[((0,), torch.float16)],
)


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("shape,dtype", EMPTY_CASES)
def test__autocast_to_full_precision_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    cuda_enabled, cpu_enabled = True, True

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled, cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(inp, cuda_enabled, cpu_enabled)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches(inp, ref_inp, res_out, ref_out)


def _views(dtype):
    """Views with non-contiguous strides, a nonzero storage offset and an
    expanded stride, all valid inputs for the operator."""
    base = tu.make_input(dtype, (8, 12), ["-1", "1"])
    return {
        "contiguous": base,
        "transposed": base.transpose(0, 1),
        "column_slice": base[:, 1::2],  # storage offset 1, stride (12, 2)
        "offset_view": base[1:5, 2:9],  # storage offset 14
        "expanded": base[0:1].expand(4, 12),  # stride (0, 1)
    }


_LAYOUTS = ["contiguous", "transposed", "column_slice", "offset_view", "expanded"]


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize(
    "dtype", _gated([torch.float16, torch.bfloat16, torch.float32])
)
@pytest.mark.parametrize("layout", tu.selected_cases(_LAYOUTS, quick=[]))
def test__autocast_to_full_precision_layout(layout, dtype):
    # A pass-through must return the input object, so the view's strides and
    # offset survive untouched, and a promotion must land on the geometry the
    # native operator produced for that same view; a fresh contiguous input
    # cannot establish either. Both relations are read from the reference by
    # _assert_alias_matches after the values match.
    inp = _views(dtype)[layout]
    ref_inp = tu.to_reference(inp)
    cuda_enabled, cpu_enabled = True, True

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled, cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(inp, cuda_enabled, cpu_enabled)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches(inp, ref_inp, res_out, ref_out)


# complex64 has no promotable counterpart, so it always takes the identity path
# and the lazy conjugate bit has to survive it.
CONJ_CASES = tu.selected_cases([(torch.complex64, (4, 8))], quick=[])


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("dtype,shape", CONJ_CASES)
def test__autocast_to_full_precision_conj_view(dtype, shape):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).conj()
    ref_inp = tu.to_reference(inp)
    cuda_enabled, cpu_enabled = True, True

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled, cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(inp, cuda_enabled, cpu_enabled)

    assert res_out.is_conj()
    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches(inp, ref_inp, res_out, ref_out)


# The shared generator already keeps only the scenarios each dtype can represent
# (e4m3fn has no infinity) and covers the promotable dtypes, so every applicable
# nan / inf / mixed combination is collected.
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(DTYPES), quick=[])


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test__autocast_to_full_precision_special_values(dtype, scenario):
    # The widening cast preserves nan and infinity exactly, so the shared exact
    # assertion (equal_nan=True) is the right comparison.
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)
    cuda_enabled, cpu_enabled = True, True

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled, cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(inp, cuda_enabled, cpu_enabled)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches(inp, ref_inp, res_out, ref_out)


# One promoting pair and two flag-disabled pairs: the disabled pairs keep the
# input as the graph leaf, which is the path a detach/ignore-flag defect breaks.
BACKWARD_FLAG_CASES = tu.selected_cases(
    [(True, True), (False, True), (False, False)], quick=[]
)
BACKWARD_DTYPES = _gated([torch.float16, torch.bfloat16, torch.float32, torch.float64])


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("cuda_enabled,cpu_enabled", BACKWARD_FLAG_CASES)
def test__autocast_to_full_precision_backward(dtype, cuda_enabled, cpu_enabled):
    # The cast has a deterministic identity gradient (the upstream gradient
    # rounded to the input dtype), so the shared exact assertion applies. No
    # loss product is needed: the leaf input supports autograd.grad directly,
    # including on the pass-through path. The upstream gradient is built on the
    # reference's device for the reference and on the candidate's device for the
    # candidate.
    inp = tu.make_input(dtype, (16, 32), ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_full_precision(
        ref_inp, cuda_enabled, cpu_enabled
    )
    res_out = flag_gems._autocast_to_full_precision(inp, cuda_enabled, cpu_enabled)
    tu.assert_result_equal(res_out, ref_out)

    upstream = tu.make_input(res_out.dtype, (16, 32), ["-1", "1"])
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )[0]
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    tu.assert_result_equal(res_grad, ref_grad)


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("flags", [(), (True,)], ids=["no-flags", "one-flag"])
def test__autocast_to_full_precision_rejects_missing_flags(flags):
    # Both booleans are required by the schema and have no default.
    inp = tu.make_input(torch.float16, (4, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._autocast_to_full_precision(inp, *flags)


@pytest.mark.autocast_to_full_precision
def test__autocast_to_full_precision_rejects_extra_positional():
    inp = tu.make_input(torch.float16, (4, 4), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._autocast_to_full_precision(inp, True, True, True)


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("bad_self", [1.0, "tensor"], ids=["float", "str"])
def test__autocast_to_full_precision_rejects_non_tensor(bad_self):
    # ``self`` is declared as a Tensor, so a non-tensor argument is rejected
    # rather than reaching an attribute lookup. (A literal None is accepted by
    # the native dispatcher and is therefore not a case.)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._autocast_to_full_precision(bad_self, True, True)


# Both flags are declared as bool, so a string and a multi-element tensor are
# rejected instead of having their truth value coerced. (Plain ints are coerced
# natively, so they are not negative cases.) The tensor is built inside the test
# body, so case collection performs no tensor operation.
_BAD_FLAG_FACTORIES = {
    "string": lambda: "yes",
    "tensor": lambda: torch.ones(3, device=flag_gems.device),
}


@pytest.mark.autocast_to_full_precision
@pytest.mark.parametrize("bad_flag_kind", ["string", "tensor"])
def test__autocast_to_full_precision_rejects_non_bool_flags(bad_flag_kind):
    inp = tu.make_input(torch.float16, (4, 4), ["-1", "1"])
    bad_flag = _BAD_FLAG_FACTORIES[bad_flag_kind]()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._autocast_to_full_precision(inp, bad_flag, True)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._autocast_to_full_precision(inp, True, bad_flag)
