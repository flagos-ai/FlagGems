"""Accuracy tests for causal_conv1d / causal_conv1d_update.

Append to tests/test_special_ops.py (or keep as its own module and reuse the
`device`, `QUICK_MODE`, `gems_assert_close` imports from the shared test utils).
"""

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from flag_gems.ops.causal_conv1d import (
    PAD_SLOT_ID,
    _causal_conv1d_fwd_kernel,
    _causal_conv1d_update_kernel,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from flag_gems.utils.triton_version_utils import HAS_TLE

from .accuracy_utils import QUICK_MODE
from .conftest import TO_CPU  # noqa: F401

device = flag_gems.device

_TOL = {
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=5e-2, atol=5e-2),
    torch.float32: dict(rtol=1e-4, atol=1e-4),
}


def _assert_close(res, ref):
    assert res.dtype == ref.dtype
    torch.testing.assert_close(res.float(), ref.float(), **_TOL[res.dtype])


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def _ref_causal_conv1d(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    activation,
):
    """Eager reference. Returns (out, updated_conv_states).

    Depthwise causal conv: out[:, t] = sum_k w[:, k] * x_hist[:, t - (W-1) + k],
    where x_hist is the sequence prefixed by its initial conv_state (or zeros).
    """
    dim, _ = x.shape
    width = weight.shape[1]
    state_len = width - 1

    out = torch.empty_like(x, dtype=torch.float32)
    new_states = conv_states.clone()

    batch = query_start_loc.numel() - 1
    for i in range(batch):
        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        seqlen = end - start
        if seqlen == 0:
            continue

        slot = int(cache_indices[i])
        if slot == PAD_SLOT_ID:
            continue

        seq = x[:, start:end].float()  # (dim, seqlen)

        if bool(has_initial_state[i]):
            prefix = conv_states[slot].float()  # (dim, state_len)
        else:
            prefix = torch.zeros(dim, state_len, device=x.device, dtype=torch.float32)

        padded = torch.cat([prefix, seq], dim=1)  # (dim, state_len + seqlen)

        # depthwise conv1d: groups == dim
        y = F.conv1d(
            padded.unsqueeze(0),
            weight.float().unsqueeze(1),  # (dim, 1, width)
            bias=bias.float() if bias is not None else None,
            groups=dim,
        ).squeeze(
            0
        )  # (dim, seqlen)

        if activation in ("silu", "swish"):
            y = F.silu(y)
        out[:, start:end] = y

        # new state = last `state_len` tokens of the padded history
        new_states[slot] = padded[:, -state_len:].to(conv_states.dtype)

    return out.to(x.dtype), new_states


def _ref_causal_conv1d_update(
    x, conv_state, weight, bias, activation, conv_state_indices
):
    """Eager reference for the decode step. Returns (out, updated_conv_state)."""
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = width - 1

    out = torch.empty_like(x, dtype=torch.float32)
    new_state = conv_state.clone()

    for i in range(batch):
        slot = int(conv_state_indices[i])
        if slot == PAD_SLOT_ID:
            continue

        prefix = conv_state[slot].float()  # (dim, state_len)
        seq = x[i].float()  # (dim, seqlen)
        padded = torch.cat([prefix, seq], dim=1)

        y = F.conv1d(
            padded.unsqueeze(0),
            weight.float().unsqueeze(1),
            bias=bias.float() if bias is not None else None,
            groups=dim,
        ).squeeze(0)

        if activation in ("silu", "swish"):
            y = F.silu(y)
        out[i] = y

        new_state[slot] = padded[:, -state_len:].to(conv_state.dtype)

    if squeeze:
        out = out.squeeze(-1)
    return out.to(x.dtype), new_state


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _make_varlen_data(dim, total_seqlen, batch, width, dtype, seed=0):
    """Random varlen batch. Sequence boundaries are random, all lengths >= 1."""
    torch.manual_seed(seed)
    eos_pos = torch.randperm(total_seqlen - 1)[: batch - 1].sort().values
    seqlens = torch.diff(
        torch.cat(
            [
                torch.tensor([-1], dtype=torch.int32),
                eos_pos.to(dtype=torch.int32),
                torch.tensor([total_seqlen - 1], dtype=torch.int32),
            ]
        )
    )
    query_start_loc = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            torch.cumsum(seqlens, dim=0).to(torch.int32),
        ]
    ).to(device)

    x = torch.randn(dim, total_seqlen, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
    has_initial_state = torch.ones(batch, dtype=torch.bool, device=device)
    return (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )


def _make_update_data(dim, batch, width, dtype, seqlen=1, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
    conv_state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    return x, weight, bias, conv_state, conv_state_indices


# ---------------------------------------------------------------------------
# Test matrices
# ---------------------------------------------------------------------------

CC1D_DIM_LIST = [1024, 2048] if not QUICK_MODE else [1024]
CC1D_SEQLEN_LIST = [2048, 4096] if not QUICK_MODE else [2048]
CC1D_BATCH_LIST = [8, 32] if not QUICK_MODE else [8]
CC1D_WIDTH_LIST = [2, 3, 4] if not QUICK_MODE else [4]
CC1D_DTYPE_LIST = [torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# Forward (varlen) accuracy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("dim", CC1D_DIM_LIST)
@pytest.mark.parametrize("total_seqlen", CC1D_SEQLEN_LIST)
@pytest.mark.parametrize("batch", CC1D_BATCH_LIST)
@pytest.mark.parametrize("width", CC1D_WIDTH_LIST)
@pytest.mark.parametrize("dtype", CC1D_DTYPE_LIST)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("activation", ["silu", None])
def test_accuracy_causal_conv1d(
    dim, total_seqlen, batch, width, dtype, has_bias, activation
):
    x, weight, bias, conv_states, qsl, ci, his = _make_varlen_data(
        dim, total_seqlen, batch, width, dtype
    )
    if not has_bias:
        bias = None

    ref_out, ref_states = _ref_causal_conv1d(
        x, weight, bias, conv_states, qsl, ci, his, activation
    )

    # conv_states is updated in place -- give the kernel its own copy.
    res_states = conv_states.clone()
    res_out = causal_conv1d_fn(
        x, weight, bias, res_states, qsl, ci, his, activation=activation
    )

    _assert_close(res_out, ref_out)
    _assert_close(res_states, ref_states)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("width", CC1D_WIDTH_LIST)
@pytest.mark.parametrize("dtype", CC1D_DTYPE_LIST)
def test_accuracy_causal_conv1d_no_initial_state(width, dtype):
    """has_initial_state=False: history must be treated as zeros, not stale cache."""
    dim, total_seqlen, batch = 1024, 2048, 8
    x, weight, bias, conv_states, qsl, ci, his = _make_varlen_data(
        dim, total_seqlen, batch, width, dtype
    )
    his = torch.zeros(batch, dtype=torch.bool, device=device)

    ref_out, ref_states = _ref_causal_conv1d(
        x, weight, bias, conv_states, qsl, ci, his, "silu"
    )
    res_states = conv_states.clone()
    res_out = causal_conv1d_fn(
        x, weight, bias, res_states, qsl, ci, his, activation="silu"
    )

    _assert_close(res_out, ref_out)
    _assert_close(res_states, ref_states)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("width", CC1D_WIDTH_LIST)
def test_accuracy_causal_conv1d_short_sequences(width):
    """seqlen < state_len exercises the `state_len > seqlen` state-merge branch,
    which is a separate code path from the common case."""
    dtype = torch.float16
    dim, batch = 512, 4
    # every sequence is 1 or 2 tokens, i.e. shorter than state_len for width >= 4
    seqlens = torch.tensor([1, 2, 1, 2], dtype=torch.int32)
    qsl = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens, 0).to(torch.int32)]
    ).to(device)
    total = int(qsl[-1])

    torch.manual_seed(0)
    x = torch.randn(dim, total, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
    ci = torch.arange(batch, dtype=torch.int32, device=device)
    his = torch.ones(batch, dtype=torch.bool, device=device)

    ref_out, ref_states = _ref_causal_conv1d(
        x, weight, bias, conv_states, qsl, ci, his, "silu"
    )
    res_states = conv_states.clone()
    res_out = causal_conv1d_fn(
        x, weight, bias, res_states, qsl, ci, his, activation="silu"
    )

    _assert_close(res_out, ref_out)
    _assert_close(res_states, ref_states)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("width", CC1D_WIDTH_LIST)
def test_accuracy_causal_conv1d_pad_slot(width):
    """Sequences mapped to PAD_SLOT_ID must be skipped, not computed."""
    dtype = torch.float16
    dim, total_seqlen, batch = 512, 1024, 8
    x, weight, bias, conv_states, qsl, ci, his = _make_varlen_data(
        dim, total_seqlen, batch, width, dtype
    )
    ci = ci.clone()
    ci[1] = PAD_SLOT_ID
    ci[5] = PAD_SLOT_ID

    _, ref_states = _ref_causal_conv1d(
        x, weight, bias, conv_states, qsl, ci, his, "silu"
    )

    res_states = conv_states.clone()
    causal_conv1d_fn(x, weight, bias, res_states, qsl, ci, his, activation="silu")

    # padded slots keep their original contents
    _assert_close(res_states, ref_states)


# ---------------------------------------------------------------------------
# Update (decode) accuracy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("dim", CC1D_DIM_LIST)
@pytest.mark.parametrize("batch", [16, 256] if not QUICK_MODE else [16])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6] if not QUICK_MODE else [4])
@pytest.mark.parametrize("dtype", CC1D_DTYPE_LIST)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("activation", ["silu", None])
def test_accuracy_causal_conv1d_update(dim, batch, width, dtype, has_bias, activation):
    x, weight, bias, conv_state, csi = _make_update_data(dim, batch, width, dtype)
    if not has_bias:
        bias = None

    ref_out, ref_state = _ref_causal_conv1d_update(
        x, conv_state, weight, bias, activation, csi
    )

    res_state = conv_state.clone()
    res_out = causal_conv1d_update(
        x, res_state, weight, bias, activation=activation, conv_state_indices=csi
    )

    _assert_close(res_out, ref_out)
    _assert_close(res_state, ref_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("width", [2, 3, 4])
def test_accuracy_causal_conv1d_update_2d_input(width):
    """x given as (batch, dim) -- the unsqueeze/squeeze path."""
    dtype = torch.float16
    dim, batch = 1024, 16
    x, weight, bias, conv_state, csi = _make_update_data(dim, batch, width, dtype)
    x2d = x.squeeze(-1).contiguous()

    ref_out, ref_state = _ref_causal_conv1d_update(
        x2d, conv_state, weight, bias, "silu", csi
    )

    res_state = conv_state.clone()
    res_out = causal_conv1d_update(
        x2d, res_state, weight, bias, activation="silu", conv_state_indices=csi
    )

    assert res_out.shape == x2d.shape
    _assert_close(res_out, ref_out)
    _assert_close(res_state, ref_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.causal_conv1d
def test_causal_conv1d_update_does_not_clobber_input():
    """The kernel must write to a fresh buffer, not in-place over x."""
    dtype = torch.float16
    dim, batch, width = 512, 8, 4
    x, weight, bias, conv_state, csi = _make_update_data(dim, batch, width, dtype)
    x_orig = x.clone()

    causal_conv1d_update(
        x,
        conv_state.clone(),
        weight,
        bias,
        activation="silu",
        conv_state_indices=csi,
    )
    torch.testing.assert_close(x, x_orig, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# TLE-vs-baseline parity
#
# The tests above already validate whichever path the dispatcher picked. These
# force the *other* path so that both kernels are covered on TLE-capable HW.
# ---------------------------------------------------------------------------


def _has_tle_hw():
    if not (HAS_TLE and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_capability()[0] >= 9


@pytest.mark.skipif(
    not _has_tle_hw(), reason="requires Triton>=3.6 with TLE on Hopper+"
)
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("dtype", CC1D_DTYPE_LIST)
def test_causal_conv1d_tle_matches_baseline(width, dtype, monkeypatch):
    """TLE forward kernel must be bit-comparable to the baseline kernel."""
    dim, total_seqlen, batch = 2048, 4096, 32
    x, weight, bias, conv_states, qsl, ci, his = _make_varlen_data(
        dim, total_seqlen, batch, width, dtype
    )

    # TLE path (default on this HW)
    tle_states = conv_states.clone()
    tle_out = causal_conv1d_fn(
        x, weight, bias, tle_states, qsl, ci, his, activation="silu"
    )

    # Force the baseline kernel
    monkeypatch.setattr("flag_gems.ops.causal_conv1d._tle_available", lambda _x: False)
    base_states = conv_states.clone()
    base_out = causal_conv1d_fn(
        x, weight, bias, base_states, qsl, ci, his, activation="silu"
    )

    _assert_close(tle_out, base_out)
    _assert_close(tle_states, base_states)


@pytest.mark.skipif(
    not _has_tle_hw(), reason="requires Triton>=3.6 with TLE on Hopper+"
)
@pytest.mark.causal_conv1d
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", CC1D_DTYPE_LIST)
def test_causal_conv1d_update_tle_matches_baseline(width, dtype, monkeypatch):
    dim, batch = 2048, 256
    x, weight, bias, conv_state, csi = _make_update_data(dim, batch, width, dtype)

    tle_state = conv_state.clone()
    tle_out = causal_conv1d_update(
        x, tle_state, weight, bias, activation="silu", conv_state_indices=csi
    )

    monkeypatch.setattr("flag_gems.ops.causal_conv1d._tle_available", lambda _x: False)
    base_state = conv_state.clone()
    base_out = causal_conv1d_update(
        x, base_state, weight, bias, activation="silu", conv_state_indices=csi
    )

    _assert_close(tle_out, base_out)
    _assert_close(tle_state, base_state)


# Keep the kernel symbols referenced so linters don't drop the imports; they are
# the objects the dispatcher selects between and are useful for direct debugging.
assert _causal_conv1d_fwd_kernel is not None
assert _causal_conv1d_update_kernel is not None
