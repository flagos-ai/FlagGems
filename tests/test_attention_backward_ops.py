import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils


def make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device):
    set_philox_state(1234567890, 0, device)
    Q = torch.empty(
        batch, q_seq_len, num_head, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    K = torch.empty(
        batch, kv_seq_len, num_head, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    V = torch.empty(
        batch, kv_seq_len, num_head, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    return Q, K, V


def make_philox(device):
    return (
        torch.tensor(0, dtype=torch.int64, device=device),
        torch.tensor(0, dtype=torch.int64, device=device),
    )


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
    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    kwargs = dict(scale=scale)
    if window_size_left >= 0:
        kwargs["window_size_left"] = window_size_left
    if window_size_right >= 0:
        kwargs["window_size_right"] = window_size_right

    output, lse, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        Q,
        K,
        V,
        None,
        None,
        Q.shape[1],
        K.shape[1],
        dropout_p,
        is_causal,
        False,
        **kwargs,
    )
    philox_seed = rng_state[0]
    philox_offset = rng_state[1]

    return output.contiguous(), lse.float(), philox_seed, philox_offset


def cudnn_attn_forward_native(
    Q,
    K,
    V,
    attn_bias=None,
    is_causal=False,
    dropout_p=0.0,
    softmax_scale=None,
):
    Q_bhsd = Q.permute(0, 2, 1, 3).contiguous()
    K_bhsd = K.permute(0, 2, 1, 3).contiguous()
    V_bhsd = V.permute(0, 2, 1, 3).contiguous()

    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    results = torch.ops.aten._scaled_dot_product_cudnn_attention(
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        attn_bias,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=False,
        scale=scale,
    )
    out_bhsd = results[0]
    lse_4d = results[1]
    philox_seed = results[2]
    philox_offset = results[3]

    out = out_bhsd.permute(0, 2, 1, 3).contiguous()
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
    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    results = torch.ops.aten._efficient_attention_forward(
        Q,
        K,
        V,
        bias,
        None,
        None,
        Q.shape[1],
        K.shape[1],
        dropout_p=dropout_p,
        custom_mask_type=1 if is_causal else 0,
        compute_log_sumexp=True,
        scale=scale,
    )
    out = results[0]
    lse_aligned = results[1]
    philox_seed = results[2]
    philox_offset = results[3]

    lse = lse_aligned[:, :, : Q.shape[1]].float()
    return out.contiguous(), lse_aligned.float(), lse, philox_seed, philox_offset


def efficient_attn_sdp_forward_native(
    Q,
    K,
    V,
    attn_bias=None,
    is_causal=False,
    dropout_p=0.0,
    softmax_scale=None,
):
    Q_bhsd = Q.permute(0, 2, 1, 3).contiguous()
    K_bhsd = K.permute(0, 2, 1, 3).contiguous()
    V_bhsd = V.permute(0, 2, 1, 3).contiguous()

    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    results = torch.ops.aten._scaled_dot_product_efficient_attention(
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        attn_bias,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    out_bhsd = results[0]
    lse = results[1]
    philox_seed = results[2]
    philox_offset = results[3]

    return out_bhsd, Q_bhsd, K_bhsd, V_bhsd, lse.float(), philox_seed, philox_offset


@pytest.mark.flash_attention_backward
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len",
    [
        (2, 4, 512, 512),
        (1, 2, 1024, 1024),
        (1, 1, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "dtype, is_causal, window_size_left, window_size_right, head_size",
    [
        (torch.float16, False, None, None, 64),
        (torch.float16, False, None, None, 128),
        (torch.float16, True, None, None, 64),
        (torch.float16, True, None, None, 128),
        (torch.float16, False, 128, 0, 64),
        (torch.float16, False, 128, 0, 128),
        (torch.float16, False, 64, 64, 64),
        (torch.float16, False, 64, 64, 128),
        (torch.bfloat16, False, None, None, 64),
        (torch.bfloat16, True, None, None, 64),
        (torch.bfloat16, False, 128, 0, 64),
        (torch.bfloat16, False, 64, 64, 64),
        (torch.bfloat16, False, None, None, 128),
    ],
)
def test_flash_attention_backward(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    dtype,
    is_causal,
    window_size_left,
    window_size_right,
    head_size,
):
    dev = torch_device_fn.current_device()
    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, dev)
    dOut = torch.randn(
        batch, q_seq_len, num_head, head_size, dtype=dtype, device=flag_gems.device
    )

    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right

    out, lse, philox_seed, philox_offset = flash_attn_forward_native(
        Q,
        K,
        V,
        is_causal=is_causal,
        softmax_scale=scale,
        window_size_left=wl,
        window_size_right=wr,
    )

    extra_bwd = {}
    if window_size_left is not None:
        extra_bwd["window_size_left"] = window_size_left
    if window_size_right is not None:
        extra_bwd["window_size_right"] = window_size_right

    ref_dQ, ref_dK, ref_dV = torch.ops.aten._flash_attention_backward(
        dOut,
        Q,
        K,
        V,
        out,
        lse,
        None,
        None,
        q_seq_len,
        kv_seq_len,
        0.0,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
        **extra_bwd,
    )

    with flag_gems.use_gems():
        dQ, dK, dV = torch.ops.aten._flash_attention_backward(
            dOut,
            Q,
            K,
            V,
            out,
            lse,
            None,
            None,
            q_seq_len,
            kv_seq_len,
            0.0,
            is_causal,
            philox_seed,
            philox_offset,
            scale=scale,
            **extra_bwd,
        )

    utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True)
    utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True)
    utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True)


@pytest.mark.scaled_dot_product_cudnn_attention_backward
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len",
    [
        (2, 4, 512, 512),
        (1, 2, 512, 1024),
        (1, 1, 64, 64),
        (4, 8, 128, 128),
        (2, 4, 128, 512),
        (1, 2, 768, 768),
    ],
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("has_attn_bias", [False, True])
@pytest.mark.parametrize("bias_requires_grad", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_cudnn_attention_backward(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    has_attn_bias,
    bias_requires_grad,
    dtype,
):
    if is_causal and has_attn_bias:
        pytest.skip("causal + attn_bias combination: skip")
    if bias_requires_grad and not has_attn_bias:
        pytest.skip("bias_requires_grad=True requires has_attn_bias=True")

    dev = torch_device_fn.current_device()
    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, dev)
    dOut = torch.randn(
        batch, q_seq_len, num_head, head_size, dtype=dtype, device=flag_gems.device
    )

    attn_bias = None
    if has_attn_bias:
        attn_bias = (
            torch.randn(
                batch,
                num_head,
                q_seq_len,
                kv_seq_len,
                dtype=dtype,
                device=flag_gems.device,
            )
            * 0.1
        )

    out, lse, philox_seed, philox_offset = cudnn_attn_forward_native(
        Q,
        K,
        V,
        attn_bias=attn_bias,
        is_causal=is_causal,
        softmax_scale=scale,
    )

    Q_bhsd = Q.permute(0, 2, 1, 3).contiguous()
    K_bhsd = K.permute(0, 2, 1, 3).contiguous()
    V_bhsd = V.permute(0, 2, 1, 3).contiguous()
    out_bhsd = out.permute(0, 2, 1, 3).contiguous()
    dOut_bhsd = dOut.permute(0, 2, 1, 3).contiguous()
    lse_4d = lse.unsqueeze(-1)

    (
        ref_dQ_bhsd,
        ref_dK_bhsd,
        ref_dV_bhsd,
    ) = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        out_bhsd,
        lse_4d,
        philox_seed,
        philox_offset,
        attn_bias,
        None,
        None,
        q_seq_len,
        kv_seq_len,
        0.0,
        is_causal,
        scale=scale,
    )
    ref_dQ = ref_dQ_bhsd.permute(0, 2, 1, 3).contiguous()
    ref_dK = ref_dK_bhsd.permute(0, 2, 1, 3).contiguous()
    ref_dV = ref_dV_bhsd.permute(0, 2, 1, 3).contiguous()

    with flag_gems.use_gems():
        (
            dQ_bhsd,
            dK_bhsd,
            dV_bhsd,
        ) = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            dOut_bhsd,
            Q_bhsd,
            K_bhsd,
            V_bhsd,
            out_bhsd,
            lse_4d,
            philox_seed,
            philox_offset,
            attn_bias,
            None,
            None,
            q_seq_len,
            kv_seq_len,
            0.0,
            is_causal,
            scale=scale,
        )

    dQ = dQ_bhsd.permute(0, 2, 1, 3).contiguous()
    dK = dK_bhsd.permute(0, 2, 1, 3).contiguous()
    dV = dV_bhsd.permute(0, 2, 1, 3).contiguous()

    utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True)
    utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True)
    utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True)


@pytest.mark.efficient_attention_backward
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len",
    [
        (2, 4, 512, 512),
        (1, 2, 1024, 1024),
        (1, 2, 128, 256),
        (2, 4, 384, 384),
        (1, 1, 64, 64),
    ],
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize(
    "custom_mask_type, expected_causal",
    [
        (0, False),
        (1, True),
    ],
)
@pytest.mark.parametrize("has_bias", [False])
@pytest.mark.parametrize("bias_requires_grad", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_efficient_attention_backward(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    custom_mask_type,
    expected_causal,
    has_bias,
    bias_requires_grad,
    dtype,
):
    if not has_bias and bias_requires_grad:
        pytest.skip("bias_requires_grad=True requires has_bias=True")

    dev = torch_device_fn.current_device()
    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, dev)
    dOut = torch.randn(
        batch, q_seq_len, num_head, head_size, dtype=dtype, device=flag_gems.device
    )

    bias = None
    if has_bias:
        bias = (
            torch.randn(
                batch,
                num_head,
                q_seq_len,
                kv_seq_len,
                dtype=dtype,
                device=flag_gems.device,
            )
            * 0.1
        )

    out, lse_aligned, lse, philox_seed, philox_offset = efficient_attn_forward_native(
        Q,
        K,
        V,
        bias=bias,
        is_causal=expected_causal,
        softmax_scale=scale,
    )

    ref_dQ, ref_dK, ref_dV, ref_dBias = torch.ops.aten._efficient_attention_backward(
        dOut,
        Q,
        K,
        V,
        bias,
        out,
        None,
        None,
        q_seq_len,
        kv_seq_len,
        lse_aligned,
        0.0,
        philox_seed,
        philox_offset,
        custom_mask_type,
        bias_requires_grad and has_bias,
        scale=scale,
        num_splits_key=None,
    )

    with flag_gems.use_gems():
        dQ, dK, dV, dBias_gems = torch.ops.aten._efficient_attention_backward(
            dOut,
            Q,
            K,
            V,
            bias,
            out,
            None,
            None,
            q_seq_len,
            kv_seq_len,
            lse_aligned,
            0.0,
            philox_seed,
            philox_offset,
            custom_mask_type,
            bias_requires_grad and has_bias,
            scale=scale,
            num_splits_key=None,
        )

    utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True)
    utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True)
    utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True)

    if has_bias and bias_requires_grad:
        assert dBias_gems is not None, "dBias should not be None"
        assert ref_dBias is not None, "ref dBias should not be None"
        utils.gems_assert_close(dBias_gems, ref_dBias, dtype, equal_nan=True)


@pytest.mark.scaled_dot_product_efficient_attention_backward
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len",
    [
        (2, 4, 512, 512),
        (1, 2, 1024, 1024),
        (1, 2, 128, 256),
        (2, 4, 384, 384),
        (1, 1, 64, 64),
    ],
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("has_attn_bias", [False, True])
@pytest.mark.parametrize(
    "grad_input_mask",
    [
        (True, True, True, False),
        (True, True, True, True),
        (True, False, True, False),
        (False, True, False, False),
        (True, False, False, False),
        (False, True, True, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_efficient_attention_backward(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    has_attn_bias,
    grad_input_mask,
    dtype,
):
    need_dq, need_dk, need_dv, need_dbias = grad_input_mask

    if is_causal and has_attn_bias:
        pytest.skip("causal + attn_bias: skip")
    if need_dbias and not has_attn_bias:
        pytest.skip("need_dbias=True requires has_attn_bias=True")

    dev = torch_device_fn.current_device()
    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, dev)
    dOut = torch.randn(
        batch, q_seq_len, num_head, head_size, dtype=dtype, device=flag_gems.device
    )

    attn_bias = None
    if has_attn_bias:
        attn_bias = (
            torch.randn(
                batch,
                num_head,
                q_seq_len,
                kv_seq_len,
                dtype=dtype,
                device=flag_gems.device,
            )
            * 0.1
        )

    (
        out_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        lse,
        philox_seed,
        philox_offset,
    ) = efficient_attn_sdp_forward_native(
        Q,
        K,
        V,
        attn_bias=attn_bias,
        is_causal=is_causal,
        softmax_scale=scale,
    )
    dOut_bhsd = dOut.permute(0, 2, 1, 3).contiguous()

    (
        ref_dQ_bhsd,
        ref_dK_bhsd,
        ref_dV_bhsd,
        ref_dBias,
    ) = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        attn_bias,
        out_bhsd,
        lse,
        philox_seed,
        philox_offset,
        0.0,
        grad_input_mask,
        is_causal,
        scale=scale,
    )
    ref_dQ = ref_dQ_bhsd.permute(0, 2, 1, 3).contiguous()
    ref_dK = ref_dK_bhsd.permute(0, 2, 1, 3).contiguous()
    ref_dV = ref_dV_bhsd.permute(0, 2, 1, 3).contiguous()

    with flag_gems.use_gems():
        (
            dQ_bhsd_gems,
            dK_bhsd_gems,
            dV_bhsd_gems,
            dBias_gems,
        ) = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            dOut_bhsd,
            Q_bhsd,
            K_bhsd,
            V_bhsd,
            attn_bias,
            out_bhsd,
            lse,
            philox_seed,
            philox_offset,
            0.0,
            grad_input_mask,
            is_causal,
            scale=scale,
        )

    dQ = dQ_bhsd_gems.permute(0, 2, 1, 3).contiguous()
    dK = dK_bhsd_gems.permute(0, 2, 1, 3).contiguous()
    dV = dV_bhsd_gems.permute(0, 2, 1, 3).contiguous()

    if need_dq:
        utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True)
    else:
        assert torch.all(dQ_bhsd_gems == 0), (
            f"dQ should be zero when need_dq=False, "
            f"got max abs={dQ_bhsd_gems.abs().max().item():.6f}"
        )

    if need_dk:
        utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True)
    else:
        assert torch.all(dK_bhsd_gems == 0), (
            f"dK should be zero when need_dk=False, "
            f"got max abs={dK_bhsd_gems.abs().max().item():.6f}"
        )

    if need_dv:
        utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True)
    else:
        assert torch.all(dV_bhsd_gems == 0), (
            f"dV should be zero when need_dv=False, "
            f"got max abs={dV_bhsd_gems.abs().max().item():.6f}"
        )

    if need_dbias and has_attn_bias:
        assert dBias_gems is not None, "dBias should not be None"
        assert ref_dBias is not None, "ref dBias should not be None"
        utils.gems_assert_close(dBias_gems, ref_dBias, dtype, equal_nan=True)
    else:
        assert dBias_gems is None, (
            f"dBias should be None when need_dbias=False or no bias, "
            f"got type={type(dBias_gems)}"
        )
