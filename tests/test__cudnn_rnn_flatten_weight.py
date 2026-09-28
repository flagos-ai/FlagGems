# Copyright 2026, The FlagGems Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness tests for ``torch.ops.aten._cudnn_rnn_flatten_weight``.

The native operator packs an RNN weight list into the single flat buffer cuDNN
reads and rebinds every list entry to a contiguous view of that buffer. It is
cuDNN only: the native call raises ``NotImplementedError`` on the CPU reference
device, so ``--ref cpu`` cannot act as the oracle here and the reference device
must be the accelerator one. There is no silent device fallback.

The operand is a ``Tensor[]`` of rank-2 matrices and rank-1 biases, so
``tu.selected_shapes()`` (the rank of one tensor) cannot describe the input;
``_CONFIG_ROWS`` is this operator's shape dimension instead.

Per-layer geometry (``torch.nn.modules.rnn.RNNBase``):
``w_ih = (gates * hidden, input_size or real_hidden * directions)``,
``w_hh = (gates * hidden, real_hidden)`` with
``real_hidden = proj_size if proj_size > 0 else hidden_size``, biases are
``(gates * hidden,)``, and a projected LSTM adds ``w_hr = (proj_size, hidden)``.
``weight_stride0`` is 4 biased / 2 biasless, plus one for a projection.

Both calls mutate the list they are given, so every positive test builds an
independent reference list, records the caller's entries before the call, and
compares the two post-call lists entry by entry (``_assert_rebound``): values,
shape, stride, storage offset, buffer aliasing, ``requires_grad`` and Python
object identity. ``.default`` rebinds the entries into the returned buffer while
``.out`` returns its own buffer, so the alias relation is compared instead of
assumed.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# cuDNN mode -> number of gates per hidden unit.
_GATES = {0: 1, 1: 1, 2: 4, 3: 3}

# config = (mode, input_size, hidden_size, num_layers, proj_size, bidirectional)
_CONFIG_ROWS = [
    (0, 4, 6, 1, 0, False),
    (1, 5, 7, 1, 0, False),
    (2, 6, 8, 1, 0, False),
    (3, 7, 9, 1, 0, False),
    (2, 8, 12, 2, 0, True),
    (3, 9, 10, 2, 0, False),
    (2, 10, 16, 3, 4, True),
]

_QUICK_CONFIG = (2, 4, 3, 1, 0, False)
_SMALL_CONFIG = _QUICK_CONFIG

_EXTRA_ROWS = [
    (0, 3, 5, 1, 0, True),
    (1, 4, 6, 2, 0, False),
    (2, 5, 8, 2, 3, False),
    (3, 6, 9, 3, 0, True),
    (2, 7, 12, 4, 0, True),
]

# Large arrangements; the cuDNN buffers measured for these geometries hold
# 1,444,864 and 4,202,496 elements. Default-only: they are throughput-sized and
# add nothing to the quick smoke subset.
_LARGE_ROWS = tu.selected_cases(
    [
        (0, 897, 512, 1, 0, True),
        (2, 512, 512, 1, 0, True),
    ],
    quick=[],
)

_VALUE_DTYPES = [torch.float32, torch.float16]
if flag_gems.runtime.device.support_fp64:
    _VALUE_DTYPES.append(torch.float64)

# int8/uint8/int32/bfloat16 fail with CUDNN_STATUS_NOT_SUPPORTED, while
# int64/bool/complex64/fp8 fail with "getCudnnDataTypeFromScalarType() not
# supported"; both are RuntimeError. Types whose tensors need a capability this
# backend does not advertise are only added when the static flag allows it.
_REJECTED_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.bool, torch.complex64]
if flag_gems.runtime.device.support_int64:
    _REJECTED_DTYPES.append(torch.int64)
if flag_gems.runtime.device.support_bf16:
    _REJECTED_DTYPES.append(torch.bfloat16)
if flag_gems.runtime.device.support_fp8:
    _REJECTED_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])

# stride 3 is valid for a biasless projected list (see ``_weight_stride0``), so
# this negative is scoped to the biased, non-projected list built below.
_MISMATCH_STRIDES = [1, 3, 5, 8]

# (stride, storage offset) applied to every rank-1 entry: parent[offset::stride].
# Default-only; the entry value range stays [-1,1].
_BIAS_LAYOUT_ROWS = tu.selected_cases([(2, 1), (1, 1)], quick=[])

# Rank-2 entry layouts, both dense in memory order so the native flat copy path
# accepts them:
#   "sliced" -> parent[:, ::2] of a double-width parent, stride (2n, 2), offset 0
#   "offset" -> three elements into a flat parent, then .view(shape)
# A transposed entry is not dense and is rejected (see the negative test).
# Default-only; the entry value range stays [-1,1].
_MATRIX_LAYOUT_ROWS = tu.selected_cases(["sliced", "offset"], quick=[])

_SIDE_EFFECT_ROWS = tu.selected_cases([_CONFIG_ROWS[2], _CONFIG_ROWS[6]], quick=[])

_OUT_CASES = tu.selected_cases(
    [
        (config, variant)
        for config in (_CONFIG_ROWS[2], _CONFIG_ROWS[6])
        for variant in ("empty", "prefilled")
    ],
    quick=[(_QUICK_CONFIG, "empty")],
)

# proj_size must stay below hidden_size; both violating values stay in quick.
_OVERSIZED_PROJ_ROWS = tu.selected_cases(
    [(2, 6, 8, 1, 8, False), (2, 6, 8, 1, 9, False)],
    quick=[(2, 6, 8, 1, 8, False), (2, 6, 8, 1, 9, False)],
)

_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_VALUE_DTYPES), quick=[])

_SENTINEL = -777.0


def _weight_stride0(config, bias):
    """List stride: 4/2 without projection, 5/3 with one."""
    projected = config[4] > 0
    if bias:
        return 5 if projected else 4
    return 3 if projected else 2


def _component_shapes(config, weight_stride0):
    """Weight shapes of one list, in cuDNN's (layer, direction) order."""
    mode, input_size, hidden_size, num_layers, proj_size, bidirectional = config
    gates = _GATES[mode]
    num_directions = 2 if bidirectional else 1
    real_hidden = proj_size if proj_size > 0 else hidden_size
    has_bias = weight_stride0 >= 4
    has_projection = weight_stride0 in (3, 5)
    shapes = []
    for layer in range(num_layers):
        layer_input = input_size if layer == 0 else real_hidden * num_directions
        for _ in range(num_directions):
            shapes.append((gates * hidden_size, layer_input))
            shapes.append((gates * hidden_size, real_hidden))
            if has_bias:
                shapes.append((gates * hidden_size,))
                shapes.append((gates * hidden_size,))
            if has_projection:
                shapes.append((proj_size, hidden_size))
    return shapes


def _config_args(config, batch_first=False):
    """Scalar arguments of the ATen schema, in schema order."""
    mode, input_size, hidden_size, num_layers, proj_size, bidirectional = config
    return (
        input_size,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        batch_first,
        bidirectional,
    )


def _make_weights(dtype, config, weight_stride0, value_range):
    return [
        tu.make_input(dtype, shape, value_range)
        for shape in _component_shapes(config, weight_stride0)
    ]


def _cast_weights(dtype, config, weight_stride0):
    """Rejected-dtype weights cast from float32, so only the dtype under test
    can cause the failure."""
    return [
        tu.make_input(torch.float32, shape, ["-1", "1"]).to(dtype)
        for shape in _component_shapes(config, weight_stride0)
    ]


def _strided_parent(dtype, size, stride, offset):
    """Storage a strided rank-1 entry is taken from."""
    return tu.make_input(dtype, (size * stride + offset,), ["-1", "1"])


def _strided_view(parent, stride, offset):
    return parent[offset::stride]


def _matrix_parent_shape(shape, layout):
    if layout == "sliced":
        return (shape[0], shape[1] * 2)
    return (shape[0] * shape[1] + 3,)


def _matrix_view(parent, shape, layout):
    """The non-default rank-2 view of ``parent`` used as a list entry."""
    if layout == "sliced":
        return parent[:, ::2]
    return parent[3:].view(shape)


def _transposed_matrix(dtype, shape):
    """A rank-2 component whose stride cannot be viewed flat."""
    return tu.make_input(dtype, (shape[1], shape[0]), ["-1", "1"]).t()


def _special_weights(dtype, config, weight_stride0, scenario):
    """Weights whose first matrix carries the requested non-finite payload."""
    weights = [
        torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        for shape in _component_shapes(config, weight_stride0)
    ]
    payload = tu.make_special_input(dtype, scenario)
    weights[0].reshape(-1)[: payload.numel()] = payload
    return weights


def _reference_list(weights):
    """Independent oracle list, remembering the caller's entries."""
    return [tu.to_reference(w) for w in weights]


def _object_ids(weights):
    return [id(weight) for weight in weights]


def _rebinding_signature(weights, ids, result):
    """Native-observed relations of one weight list after a flatten call.

    The candidate and the reference live in different allocations, so only
    relations are comparable: per-entry shape, stride, storage offset, whether
    the entry aliases the returned buffer, ``requires_grad``, and whether the
    entry is still the caller's own Python object instead of a replacement.
    """
    result_ptr = result.untyped_storage().data_ptr()
    return [
        (
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.storage_offset(),
            weight.untyped_storage().data_ptr() == result_ptr,
            weight.requires_grad,
            id(weight) == original_id,
        )
        for weight, original_id in zip(weights, ids)
    ]


def _assert_rebound(candidate, reference):
    """Compare two post-call weight lists.

    Each argument is ``(weights, ids_before_the_call, result)``. The reference
    list went through the native call, so its relations are the oracle; the
    candidate must reproduce them and keep the same values in every entry. A
    candidate that only returns a correct flat buffer out of cloned weights
    fails on the relations and on the per-entry values.
    """
    weights, ids, result = candidate
    ref_weights, ref_ids, ref_result = reference
    assert len(weights) == len(ref_weights)
    assert _rebinding_signature(weights, ids, result) == _rebinding_signature(
        ref_weights, ref_ids, ref_result
    )
    for weight, ref_weight in zip(weights, ref_weights):
        tu.assert_result_equal(weight, ref_weight)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize(
    "config", tu.selected_cases(_CONFIG_ROWS + _EXTRA_ROWS, quick=[_QUICK_CONFIG])
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__cudnn_rnn_flatten_weight_value(config, value_range, bias, dtype):
    weight_stride0 = _weight_stride0(config, bias)
    weights = _make_weights(dtype, config, weight_stride0, value_range)
    ref_weights = _reference_list(weights)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
        ref_weights, weight_stride0, *args
    )
    res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("batch_first", tu.selected_cases([False, True], quick=[]))
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__cudnn_rnn_flatten_weight_batch_first(batch_first, dtype):
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(dtype, config, weight_stride0, ["-1", "1"])
    ref_weights = _reference_list(weights)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config, batch_first)

    ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
        ref_weights, weight_stride0, *args
    )
    res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("config", _LARGE_ROWS)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__cudnn_rnn_flatten_weight_size(config, dtype):
    """Sizes large enough to leave cuDNN's padded staging path, checked with the
    same rebinding contract as the small geometries."""
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(dtype, config, weight_stride0, ["-1", "1"])
    ref_weights = _reference_list(weights)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
        ref_weights, weight_stride0, *args
    )
    res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("config,variant", _OUT_CASES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__cudnn_rnn_flatten_weight_out(config, variant, dtype):
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(dtype, config, weight_stride0, ["-1", "1"])
    ref_weights = _reference_list(weights)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    # An empty ``out`` makes the native operator size its own output buffer, so
    # the reference result yields the exact buffer size without a third run.
    ref_out = torch.empty(0, dtype=dtype, device=ref_weights[0].device)
    ref_res = torch.ops.aten._cudnn_rnn_flatten_weight.out(
        ref_weights, weight_stride0, *args, out=ref_out
    )

    if variant == "empty":
        # The candidate has to size its own empty buffer, like the native call.
        out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    else:
        # A prefilled buffer pins the overwrite contract: every element must be
        # replaced, not just the ones the native result happens to disagree on.
        out = torch.full(ref_res.shape, _SENTINEL, dtype=dtype, device=flag_gems.device)

    res = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args, out=out)

    assert res is out
    tu.assert_result_equal(res, ref_res)
    _assert_rebound((weights, ids, res), (ref_weights, ref_ids, ref_res))


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("layout", _BIAS_LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__cudnn_rnn_flatten_weight_bias_layout(layout, dtype):
    stride, offset = layout
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    shapes = _component_shapes(config, weight_stride0)

    parents = {}
    weights = []
    for index, shape in enumerate(shapes):
        if len(shape) == 1:
            parents[index] = _strided_parent(dtype, shape[0], stride, offset)
            weights.append(_strided_view(parents[index], stride, offset))
        else:
            weights.append(tu.make_input(dtype, shape, ["-1", "1"]))

    ref_weights = _reference_list(weights)
    ref_parents = {index: tu.to_reference(parent) for index, parent in parents.items()}
    for index, ref_parent in ref_parents.items():
        ref_weights[index] = _strided_view(ref_parent, stride, offset)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
        ref_weights, weight_stride0, *args
    )
    res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))
    # Rebinding moves the entries out of these storages; the untouched parent
    # (padding included) must survive both calls unchanged.
    for index, parent in parents.items():
        tu.assert_result_equal(parent, ref_parents[index])


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("layout", _MATRIX_LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__cudnn_rnn_flatten_weight_matrix_layout(layout, dtype):
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    shapes = _component_shapes(config, weight_stride0)
    weights = _make_weights(dtype, config, weight_stride0, ["-1", "1"])

    parent = tu.make_input(dtype, _matrix_parent_shape(shapes[0], layout), ["-1", "1"])
    weights[0] = _matrix_view(parent, shapes[0], layout)
    ref_parent = tu.to_reference(parent)
    ref_weights = _reference_list(weights)
    ref_weights[0] = _matrix_view(ref_parent, shapes[0], layout)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
        ref_weights, weight_stride0, *args
    )
    res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))
    tu.assert_result_equal(parent, ref_parent)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("config", _SIDE_EFFECT_ROWS)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__cudnn_rnn_flatten_weight_list_side_effects(config, dtype):
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(dtype, config, weight_stride0, ["-1", "1"])
    ref_weights = _reference_list(weights)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
        ref_weights, weight_stride0, *args
    )
    res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__cudnn_rnn_flatten_weight_special_values(dtype, scenario):
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    weights = _special_weights(dtype, config, weight_stride0, scenario)
    ref_weights = _reference_list(weights)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
        ref_weights, weight_stride0, *args
    )
    res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", tu.selected_cases(_VALUE_DTYPES, quick=[]))
def test__cudnn_rnn_flatten_weight_no_autograd(dtype):
    """Autograd exemption: the operator has no autograd formula.

    ``torch.nn.modules.rnn.RNNBase.flatten_parameters`` calls the flattener
    inside ``torch.no_grad()`` while the weights still require grad, so the
    checked contract is exactly that call: it succeeds, the returned buffer is
    detached, and the entries are rebound as in the native post-call list.
    With grad enabled the native result instead reports ``requires_grad=True``
    and a ``NotImplemented`` ``grad_fn`` rather than a working backward, so the
    tested contract is that ``no_grad`` call and only its forward metadata is
    compared here; no backward is attempted.
    """
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    weights = [
        tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
        for shape in _component_shapes(config, weight_stride0)
    ]
    ref_weights = _reference_list(weights)
    ids, ref_ids = _object_ids(weights), _object_ids(ref_weights)
    args = _config_args(config)

    with torch.no_grad():
        ref_out = torch.ops.aten._cudnn_rnn_flatten_weight(
            ref_weights, weight_stride0, *args
        )
        res_out = flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.requires_grad is False
    assert res_out.grad_fn is None
    _assert_rebound((weights, ids, res_out), (ref_weights, ref_ids, ref_out))


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test__cudnn_rnn_flatten_weight_rejects_dtype(dtype):
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    weights = _cast_weights(dtype, config, weight_stride0)
    args = _config_args(config)
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("weight_stride0", _MISMATCH_STRIDES)
def test__cudnn_rnn_flatten_weight_rejects_weight_stride0(weight_stride0):
    config = _SMALL_CONFIG
    weights = _make_weights(
        torch.float32, config, _weight_stride0(config, True), ["-1", "1"]
    )
    args = _config_args(config)
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("mode", [4, -1, 99])
def test__cudnn_rnn_flatten_weight_rejects_mode(mode):
    config = (0, 6, 8, 1, 0, False)
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    args = list(_config_args(config))
    args[1] = mode
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("num_layers", [0, -1])
def test__cudnn_rnn_flatten_weight_rejects_num_layers(num_layers):
    config = (0, 6, 8, 1, 0, False)
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    args = list(_config_args(config))
    args[4] = num_layers
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("config", _OVERSIZED_PROJ_ROWS)
def test__cudnn_rnn_flatten_weight_rejects_oversized_proj_size(config):
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    args = _config_args(config)
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
def test__cudnn_rnn_flatten_weight_rejects_empty_weight_list():
    # The dispatcher rejects an empty Tensor[] before any kernel is reached.
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight([], 4, 6, 0, 8, 0, 1, False, False)


@pytest.mark.cudnn_rnn_flatten_weight
def test__cudnn_rnn_flatten_weight_rejects_non_tensor_element():
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    weights[1] = 0.5
    args = _config_args(config)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
def test__cudnn_rnn_flatten_weight_rejects_mismatch():
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    weights.append(tu.make_input(torch.float32, (4, 4), ["-1", "1"]))
    args = _config_args(config)
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
def test__cudnn_rnn_flatten_weight_rejects_projection_for_gru():
    # Projection exists for LSTM only, so a GRU list carrying a projection
    # weight is refused by the native geometry check.
    config = (3, 6, 8, 1, 2, False)
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    args = _config_args(config)
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
def test__cudnn_rnn_flatten_weight_rejects_rank0_weight():
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    weights[0] = torch.zeros((), dtype=torch.float32, device=flag_gems.device)
    args = _config_args(config)
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)


@pytest.mark.cudnn_rnn_flatten_weight
def test__cudnn_rnn_flatten_weight_rejects_transposed_matrix():
    config = _SMALL_CONFIG
    weight_stride0 = _weight_stride0(config, True)
    shapes = _component_shapes(config, weight_stride0)
    weights = _make_weights(torch.float32, config, weight_stride0, ["-1", "1"])
    weights[0] = _transposed_matrix(torch.float32, shapes[0])
    args = _config_args(config)
    with pytest.raises(RuntimeError):
        flag_gems._cudnn_rnn_flatten_weight(weights, weight_stride0, *args)
