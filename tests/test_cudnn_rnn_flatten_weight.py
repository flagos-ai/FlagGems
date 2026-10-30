# Copyright 2026, The FlagOS Contributors.
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
from .conftest import TO_CPU

# (input_size, hidden_size, num_layers, bidirectional, bias)
RECURRENT_CONFIGS = [
    (4, 3, 1, False, True),
    (4, 3, 2, False, True),
    (4, 3, 3, True, True),
    (4, 3, 2, True, True),
    (8, 5, 2, False, False),
    (8, 5, 2, True, False),
    (16, 8, 2, True, True),
]

# (module class, cudnn mode, proj_size, nonlinearity)
RNN_MODES = [
    (torch.nn.RNN, 0, 0, "relu"),
    (torch.nn.RNN, 1, 0, "tanh"),
    (torch.nn.LSTM, 2, 0, None),
    (torch.nn.GRU, 3, 0, None),
    (torch.nn.LSTM, 2, 2, None),
]


def _make_module(dtype, cfg, mode_info):
    """Build a seeded nn.RNN/LSTM/GRU module on the test device."""
    cls, _, proj_size, nonlinearity = mode_info
    input_size, hidden_size, num_layers, bidirectional, bias = cfg
    kwargs = {
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "bias": bias,
    }
    if proj_size:
        kwargs["proj_size"] = proj_size
    if nonlinearity:
        kwargs["nonlinearity"] = nonlinearity
    module = cls(input_size, hidden_size, **kwargs).to(
        device=flag_gems.device, dtype=dtype
    )
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.copy_(torch.randn_like(param))
    return module


def _make_weights(dtype, cfg, mode_info):
    """Build the flat weight list of an RNN module, matching cuDNN's layout."""
    module = _make_module(torch.float32, cfg, mode_info)
    weights = [
        w.detach().clone().to(device=flag_gems.device, dtype=dtype)
        for w in module._flat_weights
    ]
    stride0 = (4 if cfg[4] else 2) + (1 if mode_info[2] else 0)
    return weights, stride0


def _op_args(cfg, mode_info):
    input_size, hidden_size, num_layers, bidirectional, _ = cfg
    _, mode, proj_size, _ = mode_info
    _, stride0 = _make_weights(torch.float32, cfg, mode_info)
    return (
        stride0,
        input_size,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        True,
        bidirectional,
    )


def _alias_layout(tensors):
    """Describe how ``tensors`` share storage, independent of base address.

    The operator rebinds every weight onto one flat buffer; the reference does
    the same, but at its own addresses, so compare the *relative* layout
    (which tensors share a storage and at what offsets) rather than pointers.
    """
    for t in tensors:
        if len(t.untyped_storage()) == 0:
            return None
    first = tensors[0].untyped_storage().data_ptr()
    return [
        (t.untyped_storage().data_ptr() - first, t.storage_offset(), tuple(t.shape))
        for t in tensors
    ]


def _reference(weights, args, dtype):
    """cuDNN reference; fall back to an fp32 pass when the dtype is unsupported.

    Returns the flat buffer together with the weight list the operator rebound
    in place, so callers can compare the rebinding layout as well. The flat
    weight buffer is a pure copy, so casting the fp32 result back to `dtype` is
    exact.
    """
    try:
        ref_weights = [utils.to_reference(w.clone()) for w in weights]
        return (
            torch.ops.aten._cudnn_rnn_flatten_weight(ref_weights, *args),
            ref_weights,
        )
    except RuntimeError:
        if dtype == torch.float32:
            raise
        ref_weights = [
            utils.to_reference(w.float().clone().contiguous()) for w in weights
        ]
        ref = torch.ops.aten._cudnn_rnn_flatten_weight(ref_weights, *args)
        return ref.to(dtype), ref_weights


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("cfg", RECURRENT_CONFIGS)
@pytest.mark.parametrize("mode_info", RNN_MODES, ids=lambda m: m[0].__name__)
def test_cudnn_rnn_flatten_weight(dtype, cfg, mode_info):
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    weights, _ = _make_weights(dtype, cfg, mode_info)
    args = _op_args(cfg, mode_info)

    ref_out, ref_weights = _reference(weights, args, dtype)

    res_weights = [w.clone() for w in weights]
    res_out = flag_gems.cudnn_rnn_flatten_weight(res_weights, *args)

    utils.gems_assert_equal(res_out, ref_out)
    # The operator is in-place: `nn.RNNBase.flatten_parameters()` discards the
    # return value and relies only on the weights being rebound onto the flat
    # buffer, so the rebinding layout has to match the reference exactly.
    assert _alias_layout(res_weights) == _alias_layout(ref_weights)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("cfg", RECURRENT_CONFIGS)
@pytest.mark.parametrize("mode_info", RNN_MODES, ids=lambda m: m[0].__name__)
def test_cudnn_rnn_flatten_weight_flatten_parameters(dtype, cfg, mode_info):
    """The production caller path: rebinding must actually take effect."""
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    module = _make_module(dtype, cfg, mode_info)
    # `Module.__init__` already calls `flatten_parameters()`, so seed the module
    # with one separately-allocated copy per parameter to reproduce the state a
    # fresh / reloaded module is in before its first flatten.
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.set_(param.detach().clone())
    params = list(module.parameters())
    before = _alias_layout(params)
    values_before = [p.detach().clone() for p in params]
    assert len({t.untyped_storage().data_ptr() for t in params}) == len(params)

    flat = flag_gems.cudnn_rnn_flatten_weight(
        module._flat_weights, *_op_args(cfg, mode_info)
    )

    # Every parameter must now be a view into the flat buffer, which is what
    # makes the module's later `_cudnn_rnn` call see one contiguous chunk.
    after = _alias_layout(params)
    assert after != before, "weights were not rebound onto the flat buffer"
    base = flat.untyped_storage().data_ptr()
    for w in params:
        assert w.untyped_storage().data_ptr() == base
    # Rebinding must keep every weight's own shape and values.
    for w, old in zip(params, values_before):
        assert w.shape == old.shape
        utils.gems_assert_equal(w, old)
    # ... and flattening again is idempotent for the module's weights.
    again = flag_gems.cudnn_rnn_flatten_weight(
        module._flat_weights, *_op_args(cfg, mode_info)
    )
    utils.gems_assert_equal(again, flat)


@pytest.mark.cudnn_rnn_flatten_weight_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("cfg", RECURRENT_CONFIGS)
@pytest.mark.parametrize("mode_info", RNN_MODES, ids=lambda m: m[0].__name__)
def test_cudnn_rnn_flatten_weight_out(dtype, cfg, mode_info):
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    weights, _ = _make_weights(dtype, cfg, mode_info)
    args = _op_args(cfg, mode_info)

    ref_out, ref_weights = _reference(weights, args, dtype)

    res_weights = [w.clone() for w in weights]
    out = torch.empty_like(ref_out)
    res_out = flag_gems.cudnn_rnn_flatten_weight_out(res_weights, *args, out=out)

    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)
    # `.out` still rebinds (the reference does the same), and the weights alias
    # the operator's own buffer rather than the caller's `out`.
    assert _alias_layout(res_weights) == _alias_layout(ref_weights)
    assert all(
        w.untyped_storage().data_ptr() != out.untyped_storage().data_ptr()
        for w in res_weights
    )


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_cudnn_rnn_flatten_weight_other_dtypes(dtype):
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    """The buffer is a bitwise copy, so any fixed-width dtype must work."""
    cfg = (4, 3, 2, True, True)
    mode_info = (torch.nn.LSTM, 2, 0, None)
    weights, _ = _make_weights(torch.float32, cfg, mode_info)
    weights = [w.to(dtype) for w in weights]
    args = _op_args(cfg, mode_info)

    res_weights = [w.clone() for w in weights]
    res_out = flag_gems.cudnn_rnn_flatten_weight(res_weights, *args)

    # cuDNN cannot handle fp64/int64 weights, so derive the copy order from an
    # fp32 probe of the same shapes filled with 0..N-1: the flattened probe is
    # then the index permutation into `cat(weights)`, which is dtype
    # independent because the flat buffer is a pure copy.
    numel = 0
    probe = []
    for w in weights:
        piece = torch.arange(w.numel(), dtype=torch.float32, device=w.device) + numel
        probe.append(piece.reshape(w.shape).contiguous())
        numel += w.numel()
    order = torch.ops.aten._cudnn_rnn_flatten_weight([w.clone() for w in probe], *args)
    ref_out = torch.cat([w.reshape(-1) for w in weights])[order.long()]

    assert res_out.dtype == dtype
    utils.gems_assert_equal(res_out, ref_out.to(dtype))


@pytest.mark.cudnn_rnn_flatten_weight
def test_cudnn_rnn_flatten_weight_errors():
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    weights, stride0 = _make_weights(
        torch.float32, (4, 3, 1, False, True), (torch.nn.LSTM, 2, 0, None)
    )
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_rnn_flatten_weight([], 4, 4, 2, 3, 0, 1, True, False)
    # Truncated weight list.
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_rnn_flatten_weight(
            weights[:2], stride0, 4, 2, 3, 0, 1, True, False
        )
    # Mixed dtypes are rejected by the reference too ("parameter types
    # mismatch"), because the flat buffer is a single-dtype copy.
    mixed = [w.clone() for w in weights]
    mixed[1] = mixed[1].double()
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_rnn_flatten_weight(mixed, stride0, 4, 2, 3, 0, 1, True, False)


@pytest.mark.cudnn_rnn_flatten_weight
def test_cudnn_rnn_flatten_weight_non_current_device():
    """Weights on a GPU that is not the current one must still flatten.

    ``nn.RNNBase.flatten_parameters()`` wraps the call in
    ``torch.cuda.device_of``, so the operator is reachable for a module that
    lives on a non-current device; the descriptor cache must also keep the two
    devices apart.
    """
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    if torch.cuda.device_count() < 2:
        pytest.skip("needs at least two GPUs")
    args = (4, 8, 2, 6, 0, 1, True, False)
    other = torch.device("cuda:1")
    if torch.cuda.current_device() == 1:
        other = torch.device("cuda:0")
    with torch.cuda.device(torch.cuda.current_device()):
        module = torch.nn.LSTM(8, 6, num_layers=1).to(other)
        weights = [w.detach().clone() for w in module._flat_weights]
        ref_out, ref_weights = _reference(weights, args, torch.float32)
        res_weights = [w.clone() for w in weights]
        res_out = flag_gems.cudnn_rnn_flatten_weight(res_weights, *args)
    utils.gems_assert_equal(res_out, ref_out)
    assert _alias_layout(res_weights) == _alias_layout(ref_weights)
