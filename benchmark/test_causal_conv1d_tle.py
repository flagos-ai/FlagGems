import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import flag_gems
from flag_gems.utils.triton_version_utils import HAS_TLE

from . import base

if flag_gems.vendor_name == "ascend":
    from flag_gems.runtime.backend._ascend.fused import (
        causal_conv1d_fn,
        causal_conv1d_update_npu,
    )
else:
    causal_conv1d_fn = None
    causal_conv1d_update_npu = None

PAD_SLOT_ID = -1

DECODE_CASES = [
    (3, False, 4096, 3, 1, False, True),
    (64, False, 4096, 3, 1, False, True),
]
PREFILL_CASES = [
    (4, True, 64, 8, 4, True, True),
    (4, False, 64, 249, 4, True, True),
    (10, True, 4096, 249, 4, True, True),
]


def _make_query_start_loc(total_tokens: int, padded_batch: int) -> list[int]:
    if padded_batch == 1:
        return [total_tokens]
    eos_pos = torch.randperm(total_tokens - 1)[: padded_batch - 1].sort().values
    boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int64),
            eos_pos.to(torch.int64),
            torch.tensor([total_tokens - 1], dtype=torch.int64),
        ]
    )
    return torch.diff(boundaries).tolist()


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


def causal_conv1d_update_ref_wrapper(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    **_: object,
) -> torch.Tensor:
    if conv_state_indices is None:
        return causal_conv1d_update_ref(x, conv_state, weight, bias, activation)
    valid_mask = conv_state_indices != pad_slot_id
    valid_indices = conv_state_indices[valid_mask].to(torch.long)
    selected_state = conv_state.index_select(0, valid_indices).clone()
    out = causal_conv1d_update_ref(
        x[valid_mask],
        selected_state,
        weight,
        bias,
        activation,
    )
    conv_state.index_copy_(0, valid_indices, selected_state)
    return out


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


def causal_conv1d_fn_ref_wrapper(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    conv_states: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    **_: object,
) -> torch.Tensor:
    assert conv_states is not None
    assert query_start_loc is not None
    assert cache_indices is not None
    assert has_initial_state is not None
    state_ref = conv_states.clone()
    seqlens = torch.diff(query_start_loc.to(torch.int64)).tolist()
    outputs = []
    start = 0
    x_3d = x.unsqueeze(0)
    for idx, seqlen in enumerate(seqlens):
        end = start + seqlen
        state_idx = int(cache_indices[idx].item())
        x_chunk = x_3d[:, :, start:end]
        start = end
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
        return torch.cat(outputs, dim=-1)
    return x.new_empty((x.shape[0], 0))


def _make_decode_input(case, dtype):
    (
        batch_size,
        with_padding,
        dim,
        width,
        seqlen,
        has_bias,
        silu_activation,
    ) = case
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size
    activation = "silu" if silu_activation else None

    x = torch.randn(
        (padded_batch_size, seqlen, dim), device=flag_gems.device, dtype=dtype
    ).transpose(1, 2)
    conv_state = torch.randn(
        (total_entries, width - 1, dim), device=flag_gems.device, dtype=dtype
    ).transpose(1, 2)
    weight = torch.randn((dim, width), device=flag_gems.device, dtype=dtype)
    bias = (
        torch.randn((dim,), device=flag_gems.device, dtype=dtype) if has_bias else None
    )
    conv_state_indices = torch.randperm(total_entries, device=flag_gems.device)[
        :batch_size
    ].to(torch.int32)
    padded_state_indices = torch.cat(
        [
            conv_state_indices,
            torch.full(
                (padding,), PAD_SLOT_ID, dtype=torch.int32, device=flag_gems.device
            ),
        ]
    )
    return (
        x,
        conv_state,
        weight,
        {
            "bias": bias,
            "activation": activation,
            "conv_state_indices": padded_state_indices,
            "pad_slot_id": PAD_SLOT_ID,
        },
    )


def _make_prefill_input(case, dtype):
    (
        batch,
        with_padding,
        dim,
        seqlen,
        width,
        has_bias,
        silu_activation,
    ) = case
    padding = 3 if with_padding else 0
    padded_batch = batch + padding
    total_entries = batch * 10
    activation = "silu" if silu_activation else None
    seqlens = _make_query_start_loc(seqlen, padded_batch)
    query_start_loc = torch.tensor(
        [0]
        + list(torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0).tolist()),
        dtype=torch.int32,
        device=flag_gems.device,
    )
    x = rearrange(
        torch.randn(1, seqlen, 4096 + dim + 64, device=flag_gems.device, dtype=dtype),
        "b s d -> b d s",
    )[:, 4096 : 4096 + dim, :].squeeze(0)
    weight = torch.randn((dim, width), device=flag_gems.device, dtype=dtype)
    bias = (
        torch.randn((dim,), device=flag_gems.device, dtype=dtype) if has_bias else None
    )
    conv_states = torch.randn(
        (total_entries, width - 1, dim), device=flag_gems.device, dtype=dtype
    ).transpose(1, 2)
    has_initial_state = torch.randint(
        0,
        2,
        (query_start_loc.numel() - 1,),
        dtype=torch.bool,
        device=flag_gems.device,
    )
    state_indices = torch.randperm(total_entries, device=flag_gems.device)[:batch].to(
        torch.int32
    )
    padded_state_indices = torch.cat(
        [
            state_indices,
            torch.full(
                (padding,), PAD_SLOT_ID, dtype=torch.int32, device=flag_gems.device
            ),
        ],
        dim=0,
    )
    return (
        x,
        weight,
        {
            "bias": bias,
            "conv_states": conv_states,
            "query_start_loc": query_start_loc,
            "cache_indices": padded_state_indices,
            "has_initial_state": has_initial_state,
            "activation": activation,
            "pad_slot_id": PAD_SLOT_ID,
        },
    )


class CausalConv1dDecodeBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "causal_conv1d_update_npu",
            causal_conv1d_update_ref_wrapper,
            [torch.bfloat16],
        )
        self.set_gems(causal_conv1d_update_npu)

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = DECODE_CASES
        self.shape_desc = "BATCH, WITH_PADDING, DIM, WIDTH, SEQLEN, HAS_BIAS, SILU"

    def get_input_iter(self, dtype):
        for idx, case in enumerate(DECODE_CASES):
            torch.manual_seed(3000 + idx)
            yield _make_decode_input(case, dtype)


class CausalConv1dPrefillBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "causal_conv1d_fn",
            causal_conv1d_fn_ref_wrapper,
            [torch.bfloat16],
        )
        self.set_gems(causal_conv1d_fn)

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = PREFILL_CASES
        self.shape_desc = "BATCH, WITH_PADDING, DIM, SEQLEN, WIDTH, HAS_BIAS, SILU"

    def get_input_iter(self, dtype):
        for idx, case in enumerate(PREFILL_CASES):
            torch.manual_seed(4000 + idx)
            yield _make_prefill_input(case, dtype)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend-only fused causal conv1d decode benchmark",
)
@pytest.mark.skipif(not HAS_TLE, reason="Triton TLE support is unavailable")
@pytest.mark.causal_conv1d_update_npu
def test_causal_conv1d_update_npu_benchmark():
    bench = CausalConv1dDecodeBenchmark()
    bench.run()


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend-only fused causal conv1d prefill benchmark",
)
@pytest.mark.skipif(not HAS_TLE, reason="Triton TLE support is unavailable")
@pytest.mark.causal_conv1d_fn
def test_causal_conv1d_fn_benchmark():
    bench = CausalConv1dPrefillBenchmark()
    bench.run()
