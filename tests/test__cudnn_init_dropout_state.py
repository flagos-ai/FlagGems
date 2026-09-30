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

import warnings

import pytest
import torch

import flag_gems

from . import test_utils as tu

# aten::_cudnn_init_dropout_state is a cuDNN factory: it takes no tensor input
# and returns the cuDNN dropout RNG state. The byte encoding is cuDNN-private
# and the extent comes from cudnnDropoutGetStatesSize for the active device, so
# neither is hardcoded here.
#
# Measured native contract: a valid call needs dropout > 0, train=True,
# dtype=torch.uint8 and a CUDA device (no CPU or Sparse dispatch exists), and
# the buffer is a pure function of dropout_seed - repeated calls, including from
# fresh processes, are bit-identical, different seeds differ, and the requested
# dropout changes no byte. Contents are therefore checked through the
# operator's only native consumer (a dropout cuDNN RNN) rather than by
# metadata: the consumer reproduces its reference output from a
# native-initialized state and diverges by O(1e2) for a state from another seed
# or for an all-zero buffer, so an arbitrary byte tensor is rejected. The
# private encoding itself is not asserted.
#
# There is no tensor operand, so the spec's value-range / shape / broadcast /
# backward / special-value grids have nothing to act on; the operator's real
# parameter space, its buffer contract and the invalid dtype / device / layout /
# pin_memory / .out forms take their place. The reference device is the
# flag_gems device because native has no CPU or Sparse dispatch.

_DROPOUTS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, float("inf"))
_SEEDS = (0, 42, 123456789, -1, 2**31 - 1, -(2**63))

# train is the only bool parameter and True is its only accepted value: native
# maps train=False to dropout_p = 0 and then asserts dropout > 0, so the False
# branch lives in the negative cases below.
_POSITIVE_CASES = [(dropout, True, seed) for dropout in _DROPOUTS for seed in _SEEDS]
POSITIVE_CASES = _POSITIVE_CASES

# Rows used by the .out call forms, which need a buffer as well as a call.
_OUT_CALL_CASES = tu.selected_cases(
    [(0.5, True, 42), (0.1, True, -1), (0.9, True, 2**31 - 1)],
    quick=[(0.5, True, 42), (0.1, True, -1), (0.9, True, 2**31 - 1)],
)

# Negative rows are collected unchanged in both levels.
_UNUSABLE_DROPOUTS = [0.0, -0.0, -0.5, float("nan"), float("-inf")]
_TRAIN_DISABLED_CASES = [(0.05, False, 7), (0.5, False, 42), (0.99, False, -1)]
# The descriptor asserts options.dtype() == kByte, so every non-byte dtype is
# rejected, dtype=None included (its schema default means float32); no valid
# call omits the dtype argument.
_REJECTED_DTYPES = [
    torch.float32,
    torch.float16,
    torch.float64,
    torch.int8,
    torch.int64,
    torch.bool,
    None,
]
_REJECTS = (RuntimeError, TypeError, ValueError, NotImplementedError)

# Consumer of the state: a 2-layer cuDNN RNN. cuDNN applies RNN dropout only
# between layers, so a single-layer RNN ignores the state and could not tell a
# real state from a zero-filled one.
_RNN_LAYERS = 2
_RNN_HIDDEN = 8
_RNN_INPUT = 4
_RNN_STEPS = 3
_RNN_BATCH = 2
_RNN_DROPOUT = 0.5
_RNN_GRAPH = None


def _cudnn_rnn_output(state):
    """Observe a dropout state through its native consumer, cuDNN RNN.

    Weights and inputs are fixed once per process, so two states are equivalent
    exactly when this RNN produces the same output from them.
    """
    global _RNN_GRAPH
    if _RNN_GRAPH is None:
        torch.manual_seed(1234)
        weights = []
        for layer in range(_RNN_LAYERS):
            in_size = _RNN_INPUT if layer == 0 else _RNN_HIDDEN
            weights += [
                torch.randn(_RNN_HIDDEN, in_size, device=flag_gems.device),
                torch.randn(_RNN_HIDDEN, _RNN_HIDDEN, device=flag_gems.device),
                torch.randn(_RNN_HIDDEN, device=flag_gems.device),
                torch.randn(_RNN_HIDDEN, device=flag_gems.device),
            ]
        weight_buf = torch.ops.aten._cudnn_rnn_flatten_weight(
            weights, 4, _RNN_INPUT, 0, _RNN_HIDDEN, 0, _RNN_LAYERS, True, False
        )
        inp = torch.randn(_RNN_STEPS, _RNN_BATCH, _RNN_INPUT, device=flag_gems.device)
        hx = torch.randn(_RNN_LAYERS, _RNN_BATCH, _RNN_HIDDEN, device=flag_gems.device)
        _RNN_GRAPH = (weights, weight_buf, inp, hx)
    weights, weight_buf, inp, hx = _RNN_GRAPH
    return torch.ops.aten._cudnn_rnn(
        inp,
        weights,
        4,
        weight_buf,
        hx,
        None,
        0,
        _RNN_HIDDEN,
        0,
        _RNN_LAYERS,
        False,
        _RNN_DROPOUT,
        True,
        False,
        [],
        state,
    )[0]


def _native_state(dropout, train, seed):
    return torch.ops.aten._cudnn_init_dropout_state(
        dropout, train, seed, dtype=torch.uint8, device=flag_gems.device
    )


def _assert_state(res, ref):
    # The extent is the cuDNN state size for this device, taken from the native
    # result instead of a hardcoded constant.
    assert res.dtype == ref.dtype == torch.uint8
    assert res.device.type == torch.device(flag_gems.device).type
    assert res.shape == ref.shape


def _assert_out_state(res, ref, buffer):
    # The .out overload returns the caller's buffer, resized to the state size.
    assert res is buffer
    _assert_state(res, ref)


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dropout,train,seed", POSITIVE_CASES)
def test__cudnn_init_dropout_state(dropout, train, seed):
    ref_out = _native_state(dropout, train, seed)

    res_out = flag_gems._cudnn_init_dropout_state(
        dropout, train, seed, dtype=torch.uint8, device=flag_gems.device
    )

    _assert_state(res_out, ref_out)
    # The factory allocates; it never returns a view or an alias.
    assert not res_out._is_view()


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dropout,train,seed", POSITIVE_CASES)
def test__cudnn_init_dropout_state_consumable_by_cudnn_rnn(dropout, train, seed):
    # Contents oracle: the candidate's state must drive the native consumer like
    # the native state for the same seed.
    ref_out = _native_state(dropout, train, seed)

    res_out = flag_gems._cudnn_init_dropout_state(
        dropout, train, seed, dtype=torch.uint8, device=flag_gems.device
    )

    tu.assert_result_close(_cudnn_rnn_output(res_out), _cudnn_rnn_output(ref_out))


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dropout,train,seed", POSITIVE_CASES)
def test__cudnn_init_dropout_state_out_resizes_empty_buffer(dropout, train, seed):
    ref_buf = torch.empty(0, dtype=torch.uint8, device=flag_gems.device)
    act_buf = torch.empty(0, dtype=torch.uint8, device=flag_gems.device)

    ref_out = torch.ops.aten._cudnn_init_dropout_state.out(
        dropout, train, seed, out=ref_buf
    )

    res_out = flag_gems._cudnn_init_dropout_state(dropout, train, seed, out=act_buf)

    _assert_out_state(res_out, ref_out, act_buf)


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dropout,train,seed", _OUT_CALL_CASES)
def test__cudnn_init_dropout_state_out_reuses_sized_buffer(dropout, train, seed):
    # A buffer that already has the state size is not resized; its old contents
    # are never read.
    ref_buf = _native_state(dropout, train, seed)
    act_buf = ref_buf.clone()
    size_before = ref_buf.shape

    ref_out = torch.ops.aten._cudnn_init_dropout_state.out(
        dropout, train, seed, out=ref_buf
    )

    res_out = flag_gems._cudnn_init_dropout_state(dropout, train, seed, out=act_buf)

    _assert_out_state(res_out, ref_out, act_buf)
    assert act_buf.shape == size_before


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dropout,train,seed", _OUT_CALL_CASES)
def test__cudnn_init_dropout_state_out_resizes_populated_buffer(dropout, train, seed):
    # Native also replaces a non-empty buffer and only warns that the resize is
    # deprecated; the warning is muted and the uninitialized contents are never
    # read.
    ref_buf = torch.empty(7, dtype=torch.uint8, device=flag_gems.device)
    act_buf = torch.empty(7, dtype=torch.uint8, device=flag_gems.device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_out = torch.ops.aten._cudnn_init_dropout_state.out(
            dropout, train, seed, out=ref_buf
        )
        res_out = flag_gems._cudnn_init_dropout_state(dropout, train, seed, out=act_buf)

    _assert_out_state(res_out, ref_out, act_buf)


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dropout", _UNUSABLE_DROPOUTS)
def test__cudnn_init_dropout_state_rejects_unusable_dropout(dropout):
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(
            dropout, True, 42, dtype=torch.uint8, device=flag_gems.device
        )


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dropout,train,seed", _TRAIN_DISABLED_CASES)
def test__cudnn_init_dropout_state_rejects_train_disabled(dropout, train, seed):
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(
            dropout, train, seed, dtype=torch.uint8, device=flag_gems.device
        )


@pytest.mark.cudnn_init_dropout_state
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test__cudnn_init_dropout_state_rejects_non_byte_dtype(dtype):
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(
            0.5, True, 42, dtype=dtype, device=flag_gems.device
        )


@pytest.mark.cudnn_init_dropout_state
def test__cudnn_init_dropout_state_rejects_cpu_device():
    # There is no CPU dispatch, so a CPU request must fail rather than hand back
    # a tensor cuDNN could never consume.
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(
            0.5, True, 42, dtype=torch.uint8, device="cpu"
        )


@pytest.mark.cudnn_init_dropout_state
def test__cudnn_init_dropout_state_rejects_sparse_layout():
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(
            0.5,
            True,
            42,
            dtype=torch.uint8,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )


@pytest.mark.cudnn_init_dropout_state
def test__cudnn_init_dropout_state_rejects_pinned_memory():
    # pin_memory applies to CPU tensors only, while this factory always builds a
    # CUDA buffer, so native rejects the request.
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(
            0.5,
            True,
            42,
            dtype=torch.uint8,
            device=flag_gems.device,
            pin_memory=True,
        )


@pytest.mark.cudnn_init_dropout_state
def test__cudnn_init_dropout_state_out_rejects_non_byte_buffer():
    buf = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(0.5, True, 42, out=buf)


@pytest.mark.cudnn_init_dropout_state
def test__cudnn_init_dropout_state_out_rejects_cpu_buffer():
    buf = torch.empty(0, dtype=torch.uint8, device="cpu")
    with pytest.raises(_REJECTS):
        flag_gems._cudnn_init_dropout_state(0.5, True, 42, out=buf)
