import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import flag_gems
from flag_gems.utils.triton_version_utils import HAS_TLE

from .conftest import QUICK_MODE

if flag_gems.vendor_name == "ascend":
    from flag_gems.runtime.backend._ascend.fused import (
        causal_conv1d_fn,
        causal_conv1d_update_npu,
    )
else:
    causal_conv1d_fn = None
    causal_conv1d_update_npu = None

PAD_SLOT_ID = -1

_DECODE_CASES = [
    (3, False, 4096, 3, 1, False, True, torch.bfloat16),
    (64, False, 4096, 3, 1, False, True, torch.bfloat16),
]
_PREFILL_CASES = [
    (4, True, 64, 8, 4, True, True, torch.bfloat16),
    (4, False, 64, 249, 4, True, True, torch.bfloat16),
    (10, True, 4096, 249, 4, True, True, torch.bfloat16),
]

if QUICK_MODE:
    DECODE_CASES = _DECODE_CASES[:1]
    PREFILL_CASES = _PREFILL_CASES[:1]
else:
    DECODE_CASES = _DECODE_CASES
    PREFILL_CASES = _PREFILL_CASES


def _atol(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 5e-2
    if dtype == torch.float16:
        return 5e-3
    return 1e-3


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 3e-4, 1e-3
    if dtype == torch.bfloat16:
        return 1e-2, 5e-2
    return 3e-3, 5e-3


def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def _make_query_start_loc(total_tokens: int, padded_batch: int) -> list[int]:
    """Generate sequence lengths that sum to total_tokens for padded_batch sequences.

    Each sequence is guaranteed to have at least 1 token.
    """
    if padded_batch == 1:
        return [total_tokens]
    if padded_batch > total_tokens:
        raise ValueError(
            f"padded_batch ({padded_batch}) cannot exceed total_tokens ({total_tokens})"
        )
    nsplits = padded_batch - 1
    eos_pos = torch.randperm(total_tokens - 1)[:nsplits].sort().values
    boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int64),
            eos_pos.to(torch.int64),
            torch.tensor([total_tokens - 1], dtype=torch.int64),
        ]
    )
    seqlens = torch.diff(boundaries).tolist()
    assert len(seqlens) == padded_batch
    assert sum(seqlens) == total_tokens
    assert all(s > 0 for s in seqlens)
    return seqlens


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0)
        copy_idx = copy_idx + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | None,
    pad_slot_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_ref = conv_states.clone()
    seqlens = torch.diff(query_start_loc.to(torch.int64)).tolist()
    outputs = []
    start = 0
    x_3d = x.unsqueeze(0)
    for idx, seqlen in enumerate(seqlens):
        state_idx = int(cache_indices[idx].item())
        x_chunk = x_3d[:, :, start : start + seqlen]
        start += seqlen
        if state_idx == pad_slot_id:
            continue
        initial_states = (
            state_ref[state_idx].unsqueeze(0)
            if bool(has_initial_state[idx].item())
            else None
        )
        out, _ = causal_conv1d_ref(
            x_chunk,
            weight,
            bias,
            initial_states=initial_states,
            return_final_states=True,
            final_states_out=state_ref[state_idx].unsqueeze(0),
            activation=activation,
        )
        outputs.append(out.squeeze(0))
    if outputs:
        return torch.cat(outputs, dim=-1), state_ref
    return x.new_empty((x.shape[0], 0)), state_ref


def _torch_prefill_ref(
    x_ref: torch.Tensor,
    seqlens: list[list[int]],
    padded_state_indices: torch.Tensor,
    weight_ref: torch.Tensor,
    bias_ref: torch.Tensor | None,
    activation: str | None,
    final_states_ref: torch.Tensor,
    has_initial_states: torch.Tensor,
) -> torch.Tensor:
    out_ref = []
    out_ref_b = []
    splits = [torch.split(var, seqlens[0], dim=-1) for var in (x_ref)]
    for i in range(len(seqlens[0])):
        x_s = [v[i].unsqueeze(0) for v in splits][0]
        if padded_state_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight_ref,
                bias_ref,
                activation=activation,
                return_final_states=True,
                final_states_out=final_states_ref[padded_state_indices[i]].unsqueeze(0),
                initial_states=final_states_ref[padded_state_indices[i]].unsqueeze(0)
                if has_initial_states[i]
                else None,
            )
        )
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=2))
    return torch.cat(out_ref, dim=0)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend-only fused causal conv1d decode test",
)
@pytest.mark.skipif(not HAS_TLE, reason="Triton TLE support is unavailable")
@pytest.mark.causal_conv1d_update_npu
@pytest.mark.parametrize(
    "batch_size, with_padding, dim, width, seqlen, has_bias, silu_activation, dtype",
    DECODE_CASES,
)
def test_causal_conv1d_update_npu(
    batch_size,
    with_padding,
    dim,
    width,
    seqlen,
    has_bias,
    silu_activation,
    dtype,
):
    device = flag_gems.device
    rtol, atol = _get_tolerances(dtype)
    set_random_seed(0)
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size
    activation = "silu" if silu_activation else None

    x = torch.randn(
        (padded_batch_size, seqlen, dim), device=device, dtype=dtype
    ).transpose(1, 2)
    conv_state_indices = torch.randperm(total_entries, device=device)[:batch_size].to(
        torch.int32
    )
    padded_state_indices = torch.cat(
        [
            conv_state_indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ]
    )
    conv_state = torch.randn(
        (total_entries, width - 1, dim), device=device, dtype=dtype
    ).transpose(1, 2)
    weight = torch.randn((dim, width), device=device, dtype=dtype)
    bias = torch.randn((dim,), device=device, dtype=dtype) if has_bias else None

    x_ref = x.clone()
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()

    out = causal_conv1d_update_npu(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )
    ref_out = causal_conv1d_update_ref(
        x_ref[:batch_size],
        conv_state_ref,
        weight,
        bias,
        activation=activation,
    )

    assert torch.allclose(out[:batch_size].cpu(), ref_out.cpu(), rtol=rtol, atol=atol)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend-only fused causal conv1d prefill test",
)
@pytest.mark.skipif(not HAS_TLE, reason="Triton TLE support is unavailable")
@pytest.mark.causal_conv1d_fn
@pytest.mark.parametrize(
    "batch, with_padding, dim, seqlen, width, has_bias, silu_activation, dtype",
    PREFILL_CASES,
)
def test_causal_conv1d_fn(
    batch,
    with_padding,
    dim,
    seqlen,
    width,
    has_bias,
    silu_activation,
    dtype,
):
    device = flag_gems.device
    rtol, atol = _get_tolerances(dtype)
    set_random_seed(0)
    activation = "silu" if silu_activation else None
    padding = 3 if with_padding else 0
    padded_batch_size = batch + padding
    nsplits = padded_batch_size - 1
    seqlens = []
    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seqlens.append(
        torch.diff(
            torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
        ).tolist()
    )
    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    total_entries = batch * 10
    query_start_loc = torch.cumsum(torch.tensor(seqlens[0]), dim=0).to(torch.int32)
    query_start_loc = torch.concat(
        [torch.tensor([0], dtype=torch.int32), query_start_loc], dim=0
    ).to(device)
    x = rearrange(
        torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=dtype),
        "b s d -> b d s",
    )[:, 4096 : 4096 + dim, :]
    weight = torch.randn((dim, width), device=device, dtype=dtype)
    bias = torch.randn((dim,), device=device, dtype=dtype) if has_bias else None
    x_ref = x.clone()
    weight_ref = weight.clone()
    bias_ref = bias.clone() if bias is not None else None
    conv_states = torch.randn(
        (total_entries, width - 1, dim), device=device, dtype=dtype
    ).transpose(1, 2)
    final_states_ref = conv_states.clone()
    has_initial_state = torch.randint(
        0, 2, (query_start_loc.numel() - 1,), dtype=torch.bool, device=device
    )
    state_indices = torch.randperm(total_entries, device=device)[:batch].to(torch.int32)
    padded_state_indices = torch.cat(
        [
            state_indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    out = causal_conv1d_fn(
        x.squeeze(0),
        weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=padded_state_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
    )
    ref_out = _torch_prefill_ref(
        x_ref,
        seqlens,
        padded_state_indices,
        weight_ref,
        bias_ref,
        activation,
        final_states_ref,
        has_initial_state,
    ).squeeze(0)

    assert torch.allclose(
        conv_states[state_indices],
        final_states_ref[state_indices],
        rtol=rtol,
        atol=atol,
    )

    unpadded_out = out[:, : ref_out.shape[-1]]
    assert torch.allclose(unpadded_out, ref_out, rtol=rtol, atol=atol)
