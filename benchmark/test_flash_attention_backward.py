import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import base

_SAFE_HEAD_SIZES = (64, 128)


def _is_unsafe_decode_causal(seq_len: int, is_causal: bool) -> bool:
    return seq_len == 1 and is_causal


def _device_index(device) -> int:
    if isinstance(device, int):
        return device
    d = torch.device(device)
    if d.index is not None:
        return d.index
    return torch_device_fn.current_device()


def flash_attn_forward_native(
    Q,
    K,
    V,
    is_causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    window_size_left=-1,
    window_size_right=-1,
):
    Q_fa = Q.transpose(1, 2).contiguous()
    K_fa = K.transpose(1, 2).contiguous()
    V_fa = V.transpose(1, 2).contiguous()

    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    kwargs = dict(scale=scale)
    if window_size_left >= 0:
        kwargs["window_size_left"] = window_size_left
    if window_size_right >= 0:
        kwargs["window_size_right"] = window_size_right

    output, lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        Q_fa,
        K_fa,
        V_fa,
        None,
        None,
        Q.shape[2],
        K.shape[2],
        dropout_p,
        is_causal,
        False,
        **kwargs,
    )
    philox_seed = rng_state[0]
    philox_offset = rng_state[1]

    out = output.transpose(1, 2).contiguous()
    return out, lse.float(), philox_seed, philox_offset


def sdp_flash_attn_forward_native(
    Q,
    K,
    V,
    is_causal=False,
    dropout_p=0.0,
    softmax_scale=None,
):
    B, H, S_q, D = Q.shape
    _, _, S_k, _ = K.shape

    Q_fa = Q.transpose(1, 2).contiguous()
    K_fa = K.transpose(1, 2).contiguous()
    V_fa = V.transpose(1, 2).contiguous()

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(D))

    output, lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        Q_fa,
        K_fa,
        V_fa,
        None,
        None,
        S_q,
        S_k,
        dropout_p,
        is_causal,
        False,
        scale=scale,
    )
    rng_state_i64 = rng_state.view(torch.int64)
    philox_seed = rng_state_i64[0]
    philox_offset = rng_state_i64[1]

    out = output.transpose(1, 2).contiguous()
    return out, lse.float(), philox_seed, philox_offset


def cudnn_attn_forward_native(
    Q,
    K,
    V,
    attn_bias=None,
    is_causal=False,
    dropout_p=0.0,
    softmax_scale=None,
):
    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    results = torch.ops.aten._scaled_dot_product_cudnn_attention(
        Q,
        K,
        V,
        attn_bias,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=False,
        scale=scale,
    )
    out = results[0]
    lse_4d = results[1]
    philox_seed = results[2]
    philox_offset = results[3]

    lse = lse_4d.squeeze(-1).float()
    return out, lse, philox_seed, philox_offset


def efficient_attn_forward_native(
    Q,
    K,
    V,
    bias=None,
    is_causal=False,
    dropout_p=0.0,
    softmax_scale=None,
):
    Q_fa = Q.permute(0, 2, 1, 3).contiguous()
    K_fa = K.permute(0, 2, 1, 3).contiguous()
    V_fa = V.permute(0, 2, 1, 3).contiguous()

    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    results = torch.ops.aten._efficient_attention_forward(
        Q_fa,
        K_fa,
        V_fa,
        bias,
        None,
        None,
        Q.shape[2],
        K.shape[2],
        dropout_p=dropout_p,
        custom_mask_type=1 if is_causal else 0,
        compute_log_sumexp=True,
        scale=scale,
    )
    out_fa = results[0]
    lse_aligned = results[1]
    philox_seed = results[2]
    philox_offset = results[3]

    out = out_fa.permute(0, 2, 1, 3).contiguous()
    lse = lse_aligned[:, :, : Q.shape[2]].float()
    return out, lse_aligned.float(), lse, philox_seed, philox_offset


def efficient_attn_sdp_forward_native(
    Q,
    K,
    V,
    attn_bias=None,
    is_causal=False,
    dropout_p=0.0,
    softmax_scale=None,
):
    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    results = torch.ops.aten._scaled_dot_product_efficient_attention(
        Q,
        K,
        V,
        attn_bias,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    out = results[0]
    lse = results[1]
    philox_seed = results[2]
    philox_offset = results[3]

    actual_seq_len = Q.shape[2]
    lse_clipped = lse[:, :, :actual_seq_len].float().contiguous()
    return out, lse_clipped, philox_seed, philox_offset


def _make_philox(device="cuda"):
    return (
        torch.tensor(0, dtype=torch.int64, device=device),
        torch.tensor(0, dtype=torch.int64, device=device),
    )


class FlashAttentionBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = []

        for head_size in _SAFE_HEAD_SIZES:
            for is_causal in (False, True):
                self.shapes.append((4, 16, 1024, 1024, head_size, is_causal))
                self.shapes.append((2, 16, 2048, 2048, head_size, is_causal))

        for seq_len in (1024, 2048, 4096, 8192):
            self.shapes.append((2, 16, seq_len, seq_len, 128, False))
            self.shapes.append((2, 16, seq_len, seq_len, 128, True))

        for num_heads in (8, 16, 32, 64):
            self.shapes.append((2, num_heads, 2048, 2048, 128, False))
            self.shapes.append((2, num_heads, 2048, 2048, 128, True))

        for batch in (1, 2, 4, 8):
            self.shapes.append((batch, 16, 2048, 2048, 128, False))

        self.shapes.append((1, 8, 4096, 4096, 128, False))
        self.shapes.append((1, 8, 4096, 4096, 128, True))
        self.shapes.append((2, 16, 4096, 4096, 128, False))
        self.shapes.append((2, 16, 4096, 4096, 128, True))
        self.shapes.append((4, 16, 4096, 4096, 128, False))
        self.shapes.append((4, 16, 4096, 4096, 128, True))
        self.shapes.append((1, 32, 4096, 4096, 128, False))
        self.shapes.append((2, 32, 2048, 2048, 64, False))

    def set_more_shapes(self):
        return None


def flash_attention_backward_input_fn(config, dtype, device):
    batch, num_heads, q_seq_len, kv_seq_len, head_size, is_causal = config
    scale = 1.0 / math.sqrt(head_size)

    dev_idx = _device_index(device)
    set_philox_state(1234567890, 0, dev_idx)

    Q = torch.empty(
        batch, num_heads, q_seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    K = torch.empty(
        batch, num_heads, kv_seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    V = torch.empty(
        batch, num_heads, kv_seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    dOut = torch.randn(
        batch, num_heads, q_seq_len, head_size, dtype=dtype, device=device
    )

    out, lse, philox_seed, philox_offset = flash_attn_forward_native(
        Q,
        K,
        V,
        is_causal=is_causal,
        softmax_scale=scale,
    )
    rng_state = torch.stack([philox_seed, philox_offset])

    Q_bshd = Q.transpose(1, 2).contiguous()
    K_bshd = K.transpose(1, 2).contiguous()
    V_bshd = V.transpose(1, 2).contiguous()
    out_bshd = out.transpose(1, 2).contiguous()
    dOut_bshd = dOut.transpose(1, 2).contiguous()

    yield (
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        out_bshd,
        lse,
        None,
        None,
        q_seq_len,
        kv_seq_len,
        0.0,
        is_causal,
        rng_state,
        {"scale": scale},
    )


def _flash_attn_bwd_aten(
    dOut_bshd,
    Q_bshd,
    K_bshd,
    V_bshd,
    out_bshd,
    lse,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    scale=None,
):
    philox_seed = rng_state[0]
    philox_offset = rng_state[1]
    return torch.ops.aten._flash_attention_backward(
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        out_bshd,
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )


def _flash_attn_bwd_gems(
    dOut_bshd,
    Q_bshd,
    K_bshd,
    V_bshd,
    out_bshd,
    lse,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    scale=None,
):
    return flag_gems.ops.flash_attention_backward(
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        out_bshd,
        lse,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        rng_state=rng_state,
        unused=None,
        scale=scale,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.flash_attention_backward
def test_perf_flash_attention_backward():
    bench = FlashAttentionBackwardBenchmark(
        op_name="flash_attention_backward",
        input_fn=flash_attention_backward_input_fn,
        torch_op=_flash_attn_bwd_aten,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(_flash_attn_bwd_gems)
    bench.run()


class SdpFlashAttentionBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = []
        for head_size in _SAFE_HEAD_SIZES:
            for is_causal in (False, True):
                self.shapes.append((4, 8, 1024, head_size, is_causal))
        for batch, num_heads, seq_len in (
            (1, 1, 128),
            (2, 4, 256),
            (4, 8, 512),
            (2, 16, 512),
            (1, 32, 1024),
            (8, 8, 256),
            (1, 8, 2048),
        ):
            self.shapes.append((batch, num_heads, seq_len, 128, False))
            self.shapes.append((batch, num_heads, seq_len, 128, True))
        for num_heads in (8, 16, 32):
            self.shapes.append((1, num_heads, 1, 128, False))
            self.shapes.append((4, num_heads, 1, 128, False))
        for seq_len in (17, 65, 129, 513, 1025):
            self.shapes.append((2, 8, seq_len, 64, False))

    def set_more_shapes(self):
        return None


def sdp_flash_attn_bwd_input_fn(config, dtype, device):
    batch, num_heads, seq_len, head_size, is_causal = config
    if _is_unsafe_decode_causal(seq_len, is_causal):
        return
    scale = 1.0 / math.sqrt(head_size)

    dev_idx = _device_index(device)
    set_philox_state(1234567890, 0, dev_idx)

    Q = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    K = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    V = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    dOut = torch.randn(batch, num_heads, seq_len, head_size, dtype=dtype, device=device)

    out, lse, philox_seed, philox_offset = sdp_flash_attn_forward_native(
        Q,
        K,
        V,
        is_causal=is_causal,
        softmax_scale=scale,
    )

    Q_bshd = Q.transpose(1, 2).contiguous()
    K_bshd = K.transpose(1, 2).contiguous()
    V_bshd = V.transpose(1, 2).contiguous()
    out_bshd = out.transpose(1, 2).contiguous()
    dOut_bshd = dOut.transpose(1, 2).contiguous()

    yield (
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        out_bshd,
        lse,
        None,
        None,
        seq_len,
        seq_len,
        0.0,
        is_causal,
        philox_seed,
        philox_offset,
        {"scale": scale},
    )


def _sdp_flash_attn_bwd_aten(
    dOut_bshd,
    Q_bshd,
    K_bshd,
    V_bshd,
    out_bshd,
    lse,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale=None,
):
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        out_bshd,
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
    )


def _sdp_flash_attn_bwd_gems(
    dOut_bshd,
    Q_bshd,
    K_bshd,
    V_bshd,
    out_bshd,
    lse,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale=None,
):
    return flag_gems.ops.scaled_dot_product_flash_attention_backward(
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        out_bshd,
        lse,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.scaled_dot_product_flash_attention_backward
@pytest.mark.parametrize("is_causal", [False, True])
def test_perf_scaled_dot_product_flash_attention_backward(is_causal):
    def input_fn(config, dtype, device):
        batch, num_heads, seq_len, head_size, _ = config
        if _is_unsafe_decode_causal(seq_len, is_causal):
            return
        yield from sdp_flash_attn_bwd_input_fn(
            (batch, num_heads, seq_len, head_size, is_causal), dtype, device
        )

    bench = SdpFlashAttentionBackwardBenchmark(
        op_name="scaled_dot_product_flash_attention_backward",
        input_fn=input_fn,
        torch_op=_sdp_flash_attn_bwd_aten,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(_sdp_flash_attn_bwd_gems)
    bench.run()


class CudnnAttentionBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = []

        for head_size in _SAFE_HEAD_SIZES:
            for is_causal in (False, True):
                self.shapes.append((4, 16, 1024, head_size, is_causal, False))
                self.shapes.append((2, 16, 2048, head_size, is_causal, False))

        for head_size in (64, 128):
            self.shapes.append((2, 8, 512, head_size, False, True))
            self.shapes.append((2, 8, 1024, head_size, False, True))

        for batch, num_heads, seq_len in (
            (4, 8, 512),
            (2, 16, 512),
            (4, 16, 1024),
            (1, 32, 1024),
            (8, 8, 1024),
            (1, 8, 2048),
            (2, 16, 2048),
        ):
            self.shapes.append((batch, num_heads, seq_len, 128, False, False))
            self.shapes.append((batch, num_heads, seq_len, 128, True, False))

        self.shapes.append((1, 8, 4096, 64, False, False))
        self.shapes.append((1, 8, 4096, 128, False, False))
        self.shapes.append((1, 8, 4096, 128, True, False))
        self.shapes.append((2, 16, 4096, 128, False, False))
        self.shapes.append((4, 16, 4096, 128, False, False))

    def set_more_shapes(self):
        return None


def cudnn_attn_bwd_input_fn(config, dtype, device):
    batch, num_heads, seq_len, head_size, is_causal, has_bias = config
    if _is_unsafe_decode_causal(seq_len, is_causal):
        return
    scale = 1.0 / math.sqrt(head_size)

    dev_idx = _device_index(device)
    set_philox_state(1234567890, 0, dev_idx)

    Q_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    K_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    V_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    dOut_bhsd = torch.randn(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    )
    bias = (
        torch.randn(batch, num_heads, seq_len, seq_len, device=device, dtype=dtype)
        * 0.1
        if has_bias
        else None
    )

    out_bhsd, lse, philox_seed, philox_offset = cudnn_attn_forward_native(
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        attn_bias=bias,
        is_causal=is_causal,
        softmax_scale=scale,
    )
    lse_4d = lse.unsqueeze(-1)

    yield (
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        out_bhsd,
        lse_4d,
        lse,
        philox_seed,
        philox_offset,
        bias,
        None,
        None,
        seq_len,
        seq_len,
        0.0,
        is_causal,
        {"scale": scale},
    )


def _cudnn_attn_bwd_aten(
    dOut_bhsd,
    Q_bhsd,
    K_bhsd,
    V_bhsd,
    out_bhsd,
    lse_4d,
    lse,
    philox_seed,
    philox_offset,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    scale=None,
):
    return torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        out_bhsd,
        lse_4d,
        philox_seed,
        philox_offset,
        attn_bias,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        scale=scale,
    )


def _cudnn_attn_bwd_gems(
    dOut_bhsd,
    Q_bhsd,
    K_bhsd,
    V_bhsd,
    out_bhsd,
    lse_4d,
    lse,
    philox_seed,
    philox_offset,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    scale=None,
):
    return flag_gems.ops.scaled_dot_product_cudnn_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        out_bhsd,
        lse_4d,
        philox_seed,
        philox_offset,
        attn_bias,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        scale=scale,
        bias_requires_grad=(attn_bias is not None),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.scaled_dot_product_cudnn_attention_backward
def test_perf_scaled_dot_product_cudnn_attention_backward():
    bench = CudnnAttentionBackwardBenchmark(
        op_name="scaled_dot_product_cudnn_attention_backward",
        input_fn=cudnn_attn_bwd_input_fn,
        torch_op=_cudnn_attn_bwd_aten,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(_cudnn_attn_bwd_gems)
    bench.run()


class EfficientAttentionBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = []

        for head_size in _SAFE_HEAD_SIZES:
            self.shapes.append((4, 16, 1024, head_size, 0, False, False))
            self.shapes.append((4, 16, 1024, head_size, 1, False, False))
            self.shapes.append((2, 16, 2048, head_size, 0, False, False))
            self.shapes.append((2, 16, 2048, head_size, 1, False, False))

        for head_size in (64, 128):
            self.shapes.append((2, 8, 512, head_size, 0, True, False))
            self.shapes.append((2, 8, 1024, head_size, 0, True, False))

        for batch, num_heads, seq_len in (
            (2, 16, 512),
            (4, 16, 1024),
            (1, 32, 1024),
            (8, 8, 1024),
            (1, 8, 2048),
            (2, 16, 2048),
        ):
            self.shapes.append((batch, num_heads, seq_len, 128, 0, False, False))
            self.shapes.append((batch, num_heads, seq_len, 128, 1, False, False))

        self.shapes.append((1, 8, 4096, 128, 0, False, False))
        self.shapes.append((1, 8, 4096, 64, 1, False, False))
        self.shapes.append((2, 16, 4096, 128, 0, False, False))
        self.shapes.append((4, 16, 4096, 128, 0, False, False))

    def set_more_shapes(self):
        return None


def efficient_attn_bwd_input_fn(config, dtype, device):
    batch, num_heads, seq_len, head_size, custom_mask_type, has_bias, _ = config
    is_causal = custom_mask_type == 1
    if _is_unsafe_decode_causal(seq_len, is_causal):
        return
    scale = 1.0 / math.sqrt(head_size)

    dev_idx = _device_index(device)
    set_philox_state(1234567890, 0, dev_idx)

    Q_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    K_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    V_bhsd = torch.empty(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    bias = (
        torch.randn(batch, num_heads, seq_len, seq_len, device=device, dtype=dtype)
        * 0.1
        if has_bias
        else None
    )
    dOut_bhsd = torch.randn(
        batch, num_heads, seq_len, head_size, dtype=dtype, device=device
    )

    (
        out_bhsd_tmp,
        lse_aligned,
        lse,
        philox_seed,
        philox_offset,
    ) = efficient_attn_forward_native(
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        bias=bias,
        is_causal=is_causal,
        softmax_scale=scale,
    )

    Q_bshd = Q_bhsd.permute(0, 2, 1, 3).contiguous()
    K_bshd = K_bhsd.permute(0, 2, 1, 3).contiguous()
    V_bshd = V_bhsd.permute(0, 2, 1, 3).contiguous()
    out_bshd = out_bhsd_tmp.permute(0, 2, 1, 3).contiguous()
    dOut_bshd = dOut_bhsd.permute(0, 2, 1, 3).contiguous()

    yield (
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        bias,
        out_bshd,
        None,
        None,
        seq_len,
        seq_len,
        lse_aligned,
        0.0,
        philox_seed,
        philox_offset,
        custom_mask_type,
        has_bias,
        lse,
        {"scale": scale},
    )


def _efficient_attn_bwd_aten(
    dOut_bshd,
    Q_bshd,
    K_bshd,
    V_bshd,
    attn_bias,
    out_bshd,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_aligned,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    lse,
    scale=None,
):
    return torch.ops.aten._efficient_attention_backward(
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        attn_bias,
        out_bshd,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        lse_aligned,
        dropout_p,
        philox_seed,
        philox_offset,
        custom_mask_type,
        bias_requires_grad,
        scale=scale,
        num_splits_key=0,
    )


def _efficient_attn_bwd_gems(
    dOut_bshd,
    Q_bshd,
    K_bshd,
    V_bshd,
    attn_bias,
    out_bshd,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_aligned,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    lse,
    scale=None,
):
    return flag_gems.ops.efficient_attention_backward(
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        attn_bias,
        out_bshd,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        logsumexp=lse,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        custom_mask_type=custom_mask_type,
        bias_requires_grad=bias_requires_grad,
        scale=scale,
        num_splits_key=None,
        window_size=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.efficient_attention_backward
def test_perf_efficient_attention_backward():
    bench = EfficientAttentionBackwardBenchmark(
        op_name="efficient_attention_backward",
        input_fn=efficient_attn_bwd_input_fn,
        torch_op=_efficient_attn_bwd_aten,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(_efficient_attn_bwd_gems)
    bench.run()


class SdpEfficientAttentionBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = []
        _full_grad = (True, True, True, False)
        _bias_grad = (True, True, True, True)

        for head_size in _SAFE_HEAD_SIZES:
            for is_causal in (False, True):
                self.shapes.append(
                    (4, 16, 16, 1024, head_size, is_causal, False, _full_grad)
                )
                self.shapes.append(
                    (2, 16, 16, 2048, head_size, is_causal, False, _full_grad)
                )

        for mask in (
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (True, True, False, False),
            (True, False, True, False),
            (False, True, True, False),
        ):
            self.shapes.append((2, 16, 16, 1024, 64, False, False, mask))

        for num_heads in (8, 16, 32):
            self.shapes.append(
                (2, num_heads, num_heads, 1024, 64, False, False, _full_grad)
            )
            self.shapes.append(
                (2, num_heads, num_heads, 1024, 128, True, False, _full_grad)
            )

        for seq_len in (512, 1024):
            self.shapes.append((2, 8, 8, seq_len, 64, False, True, _bias_grad))
            self.shapes.append((2, 8, 8, seq_len, 128, False, True, _bias_grad))

        for batch, num_heads, seq_len in (
            (2, 16, 512),
            (4, 16, 1024),
            (1, 32, 1024),
            (8, 8, 1024),
            (1, 8, 2048),
            (2, 16, 2048),
        ):
            self.shapes.append(
                (batch, num_heads, num_heads, seq_len, 128, False, False, _full_grad)
            )

        self.shapes.append((1, 8, 8, 4096, 128, False, False, _full_grad))
        self.shapes.append((1, 8, 8, 4096, 64, True, False, _full_grad))
        self.shapes.append((2, 16, 16, 4096, 128, False, False, _full_grad))
        self.shapes.append((4, 16, 16, 4096, 128, False, False, _full_grad))
        self.shapes.append((1, 32, 32, 4096, 128, False, False, _full_grad))

    def set_more_shapes(self):
        return None


def sdp_eff_attn_bwd_input_fn(config, dtype, device):
    batch, H_q, H_k, seq_len, head_size, is_causal, has_bias, grad_input_mask = config
    if _is_unsafe_decode_causal(seq_len, is_causal):
        return
    if H_q != H_k:
        return
    scale = 1.0 / math.sqrt(head_size)

    dev_idx = _device_index(device)
    set_philox_state(1234567890, 0, dev_idx)

    Q_bhsd = torch.empty(
        batch, H_q, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    K_bhsd = torch.empty(
        batch, H_k, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    V_bhsd = torch.empty(
        batch, H_k, seq_len, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    dOut_bhsd = torch.randn(batch, H_q, seq_len, head_size, dtype=dtype, device=device)
    bias = (
        torch.randn(batch, H_q, seq_len, seq_len, device=device, dtype=dtype) * 0.1
        if has_bias
        else None
    )

    out_bhsd, lse, philox_seed, philox_offset = efficient_attn_sdp_forward_native(
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        attn_bias=bias,
        is_causal=is_causal,
        softmax_scale=scale,
    )

    yield (
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        bias,
        out_bhsd,
        lse,
        philox_seed,
        philox_offset,
        0.0,
        list(grad_input_mask),
        is_causal,
        {"scale": scale},
    )


def _sdp_eff_attn_bwd_aten(
    dOut_bhsd,
    Q_bhsd,
    K_bhsd,
    V_bhsd,
    attn_bias,
    out_bhsd,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask,
    is_causal,
    scale=None,
):
    return torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        attn_bias,
        out_bhsd,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        list(grad_input_mask),
        is_causal,
        scale=scale,
    )


def _sdp_eff_attn_bwd_gems(
    dOut_bhsd,
    Q_bhsd,
    K_bhsd,
    V_bhsd,
    attn_bias,
    out_bhsd,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask,
    is_causal,
    scale=None,
):
    return flag_gems.ops.scaled_dot_product_efficient_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        attn_bias,
        out_bhsd,
        logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        grad_input_mask=grad_input_mask,
        is_causal=is_causal,
        scale=scale,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.scaled_dot_product_efficient_attention_backward
def test_perf_scaled_dot_product_efficient_attention_backward():
    bench = SdpEfficientAttentionBackwardBenchmark(
        op_name="scaled_dot_product_efficient_attention_backward",
        input_fn=sdp_eff_attn_bwd_input_fn,
        torch_op=_sdp_eff_attn_bwd_aten,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(_sdp_eff_attn_bwd_gems)
    bench.run()
