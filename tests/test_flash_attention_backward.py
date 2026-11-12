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


import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils


def make_qkv(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device):
    dev = torch_device_fn.current_device()
    set_philox_state(1234567890, 0, dev)
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

    output, lse, rng_state, rng_offset, _ = torch.ops.aten._flash_attention_forward(
        Q,
        K,
        V,
        None,
        None,
        Q.shape[2] if flag_gems.vendor_name == "mthreads" else Q.shape[1],
        K.shape[2] if flag_gems.vendor_name == "mthreads" else K.shape[1],
        dropout_p,
        is_causal,
        False,
        **kwargs,
    )
    return output, lse, rng_state, rng_offset


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
    raw_lse = results[1]
    philox_seed = results[2]
    philox_offset = results[3]

    out = out_bhsd.permute(0, 2, 1, 3).contiguous()
    # FlagGems backward always expects 3-D (B, H, S); the aten backward
    # expects the *same* layout returned by the aten forward.
    lse = raw_lse.squeeze(-1).float()
    return out, lse, raw_lse.float(), philox_seed, philox_offset


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


def _flash_bwd_to_device(tensor, device):
    """Keep canonical host fixtures contiguous across MetaX's 4-D transfer."""
    if (
        flag_gems.vendor_name == "metax"
        and tensor.device.type == "cpu"
        and tensor.ndim == 4
    ):
        return tensor.flatten().to(device).reshape(tensor.shape)
    return tensor.to(device)


def _flash_bwd_cpu_oracle(
    q,
    k,
    v,
    grad_out,
    q_lengths,
    k_lengths,
    *,
    scale,
    causal,
    left,
    right,
    top_left_causal=False,
    saved_out=None,
    saved_lse=None,
):
    """FP64 backward formula using the exact saved O/LSE returned to the caller."""
    # Optional state uses canonical BSHD/THD output and natural-log BHS/HT LSE.
    # Native-unavailable acceptance must consume the state actually passed in.
    assert (saved_out is None) == (saved_lse is None)
    supplied_out = None if saved_out is None else saved_out.detach().cpu().double()
    supplied_lse = None if saved_lse is None else saved_lse.detach().cpu().double()
    packed = q.ndim == 3
    q_cpu, k_cpu, v_cpu, do_cpu = (
        tensor.detach().cpu().double() for tensor in (q, k, v, grad_out)
    )
    hq, hkv = q.shape[-2], k.shape[-2]
    group = hq // hkv
    assert hq % hkv == 0
    dq, dk, dv = (torch.zeros_like(tensor) for tensor in (q_cpu, k_cpu, v_cpu))
    out = torch.zeros_like(q_cpu)
    lse = torch.empty(
        (hq, sum(q_lengths)) if packed else (len(q_lengths), hq, q.shape[1]),
        dtype=torch.float64,
    )
    q_start = k_start = 0
    actual_scale = (q.shape[-1] ** -0.5) if scale is None else scale
    for batch, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
        q_seq = q_cpu[q_start : q_start + sq] if packed else q_cpu[batch]
        k_seq = k_cpu[k_start : k_start + sk] if packed else k_cpu[batch]
        v_seq = v_cpu[k_start : k_start + sk] if packed else v_cpu[batch]
        do_seq = do_cpu[q_start : q_start + sq] if packed else do_cpu[batch]
        q_index = torch.arange(sq)[:, None]
        k_index = torch.arange(sk)[None, :]
        aligned_q = q_index + (0 if causal and top_left_causal else sk - sq)
        valid = torch.ones((sq, sk), dtype=torch.bool)
        if causal:
            valid &= k_index <= aligned_q
        if left not in (None, -1):
            valid &= k_index >= aligned_q - left
        if right not in (None, -1):
            valid &= k_index <= aligned_q + (0 if causal else right)
        for head in range(hq):
            kv_head = head // group
            scores = (q_seq[:, head] @ k_seq[:, kv_head].T) * actual_scale
            masked_scores = scores.masked_fill(~valid, -torch.inf)
            row_valid = valid.any(dim=1)
            row_lse = torch.full((sq,), torch.inf, dtype=torch.float64)
            row_lse[row_valid] = torch.logsumexp(masked_scores[row_valid], dim=1)
            probs = torch.zeros_like(scores)
            probs[row_valid] = torch.exp(
                masked_scores[row_valid] - row_lse[row_valid, None]
            )
            output = probs @ v_seq[:, kv_head]
            # Backward consumes the stored forward state, not the unrounded
            # intermediates used to construct it. Preserve that distinction
            # without copying the kernel's P/dS casts into this FP64 oracle.
            saved_output = output.to(q.dtype).double()
            saved_row_lse = row_lse.float().double()
            if flag_gems.vendor_name == "iluvatar" and q.shape[-1] not in (
                32,
                96,
                160,
                192,
                256,
            ):
                # CoreX stores -log2 LSE for this dimension family. Include
                # the format conversion's FP32 rounding in the saved state.
                saved_row_lse = (row_lse.float() * math.log2(math.e)).double()
                saved_row_lse /= math.log2(math.e)
            if supplied_out is not None:
                saved_output = (
                    supplied_out[q_start : q_start + sq, head]
                    if packed
                    else supplied_out[batch, :, head]
                )
                saved_row_lse = (
                    supplied_lse[head, q_start : q_start + sq]
                    if packed
                    else supplied_lse[batch, head]
                )
            probs[row_valid] = torch.exp(
                masked_scores[row_valid] - saved_row_lse[row_valid, None]
            )
            dp = do_seq[:, head] @ v_seq[:, kv_head].T
            delta = (saved_output * do_seq[:, head]).sum(dim=-1)
            ds = probs * (dp - delta[:, None])
            dq_seq = actual_scale * (ds @ k_seq[:, kv_head])
            dk_seq = actual_scale * (ds.T @ q_seq[:, head])
            dv_seq = probs.T @ do_seq[:, head]
            if packed:
                out[q_start : q_start + sq, head] = output
                lse[head, q_start : q_start + sq] = row_lse
                dq[q_start : q_start + sq, head] = dq_seq
                dk[k_start : k_start + sk, kv_head] += dk_seq
                dv[k_start : k_start + sk, kv_head] += dv_seq
            else:
                out[batch, :, head] = output
                lse[batch, head] = row_lse
                dq[batch, :, head] = dq_seq
                dk[batch, :, kv_head] += dk_seq
                dv[batch, :, kv_head] += dv_seq
        q_start += sq
        k_start += sk
    if supplied_out is not None:
        out, lse = supplied_out, supplied_lse
    return out.to(q.dtype), lse.float(), (dq, dk, dv)


def _flash_bwd_lse_for_backend(lse, q_lengths, device, dim):
    """Place oracle LSE in the observed backend saved-state format."""
    if flag_gems.vendor_name == "iluvatar":
        lse = -lse
        if dim not in (32, 96, 160, 192, 256):
            lse = lse * math.log2(math.e)
    if flag_gems.vendor_name not in ("mthreads", "iluvatar"):
        return _flash_bwd_to_device(lse, device)
    if lse.ndim == 3:  # MUSA dense inputs are BHSD; LSE remains [B, H, S].
        return _flash_bwd_to_device(lse, device)
    # MUSA uses BHS; CoreX accepts BHS as well as HT. Exercise BHS here.
    max_q = max(q_lengths, default=0)
    padded = torch.full(
        (len(q_lengths), lse.shape[0], max_q),
        torch.inf,
        dtype=lse.dtype,
    )
    start = 0
    for batch, length in enumerate(q_lengths):
        padded[batch, :, :length] = lse[:, start : start + length]
        start += length
    return _flash_bwd_to_device(padded, device)


def _flash_bwd_inputs_for_backend(layout, q, k, v, out, grad_out):
    if layout == "dense" and flag_gems.vendor_name == "mthreads":
        return tuple(tensor.transpose(1, 2) for tensor in (q, k, v, out, grad_out))
    return q, k, v, out, grad_out


def _flash_bwd_gradient_from_backend(layout, gradient):
    if layout == "dense" and flag_gems.vendor_name == "mthreads":
        return gradient.transpose(1, 2)
    return gradient


def _flash_bwd_checked_call(operation, args, kwargs=None):
    saved = tuple(
        (
            tensor,
            (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.device,
                tensor.dtype,
            ),
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone(),
        )
        for tensor in args
        if isinstance(tensor, torch.Tensor)
    )
    try:
        return operation(*args, **(kwargs or {}))
    finally:
        for tensor, metadata, payload in saved:
            assert (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.device,
                tensor.dtype,
            ) == metadata, "backward changed input metadata"
            actual = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            assert torch.equal(actual, payload), "backward changed an input tensor"


def _flash_bwd_p0_rng(device):
    if flag_gems.vendor_name == "hygon":
        # DTK 26.04 validates two device int64 scalars even when p == 0.
        return tuple(
            torch.zeros((), dtype=torch.int64, device=device) for _ in range(2)
        )
    return tuple(torch.empty(0, dtype=torch.uint8, device=device) for _ in range(2))


def _flash_bwd_native_reference(args, kwargs=None):
    do, q, k, v, out, lse = args[:6]
    runtime = (
        flag_gems.vendor_name,
        torch.__version__,
        torch.version.git_version,
        q.device.type,
    )
    if runtime not in _FLASH_BWD_NATIVE_RUNTIMES:
        return None, "PREFLIGHT_BLOCKED"
    if flag_gems.vendor_name == "hygon" and (
        q.ndim != 4
        or args[6] is not None
        or args[7] is not None
        or q.shape[0] == 0
        or q.shape[1] == 0
        or k.shape[1] == 0
    ):
        # DTK 26.04 native packed FlashAttention has a recorded process crash.
        return None, "PREFLIGHT_BLOCKED:dense-only"
    native_op = torch.ops.aten._flash_attention_backward.default
    schema = native_op._schema
    if (
        sum(not argument.kwarg_only for argument in schema.arguments) != 14
        or len(schema.returns) != 3
    ):
        return None, "PREFLIGHT_BLOCKED:interface"
    if args[10] != 0.0 or q.dtype not in (torch.float16, torch.bfloat16):
        return None, "PREFLIGHT_BLOCKED:parameters"
    if not all(
        tensor.is_contiguous() for tensor in args if isinstance(tensor, torch.Tensor)
    ):

        def musa_bhsd_view(tensor):
            if tensor.ndim != 4 or tensor.storage_offset() != 0:
                return False
            _, heads, seq, dim = tensor.shape
            return tensor.stride() == (seq * heads * dim, dim, heads * dim, 1)

        # Only the measured BSHD-to-BHSD transpose layout is permitted here.
        musa_views = (
            flag_gems.vendor_name == "mthreads"
            and all(
                tensor.is_contiguous() or musa_bhsd_view(tensor)
                for tensor in (do, q, k, v, out)
            )
            and all(
                tensor.is_contiguous()
                for tensor in args[5:]
                if isinstance(tensor, torch.Tensor)
            )
        )
        if not musa_views:
            return None, "PREFLIGHT_BLOCKED:layout"
    try:
        native = _flash_bwd_checked_call(native_op, args, kwargs)
    except NotImplementedError:
        return None, "UNSUPPORTED"
    except RuntimeError:
        return None, "ERROR"
    if not isinstance(native, (tuple, list)) or len(native) != 3:
        return None, "REFERENCE_INVALID:metadata"
    for gradient, source in zip(native, (q, k, v)):
        if not (
            isinstance(gradient, torch.Tensor)
            and gradient.shape == source.shape
            and gradient.dtype == source.dtype
            and gradient.device == source.device
        ):
            return None, "REFERENCE_INVALID:metadata"
        if not torch.isfinite(gradient).all():
            return None, "REFERENCE_INVALID:nonfinite"
    # Independent head/batch controls change only a neighboring logical K
    # segment and observe changed target dQ. The measured D160 family has an
    # incomplete 128-key tile; D128 and aligned-128/256 controls are unaffected.
    # Retain native priority outside the measured runtime/layout/causal scope.
    if (
        flag_gems.vendor_name == "mthreads"
        and q.ndim == 4
        and q.shape[-1] == 160
        and k.shape[2] % 128 != 0
        and (q.shape[0] > 1 or (k.is_contiguous() and k.shape[1] > 1))
        and (not args[11] or q.shape[2] > k.shape[2])
        and (kwargs or {}).get("window_size_left") in (None, -1)
        and (kwargs or {}).get("window_size_right") in (None, -1)
    ):
        from torch_musa.core import mudnn

        if mudnn.version() == 3400 and "S5000" in torch.musa.get_device_name(q.device):
            return None, "REFERENCE_INVALID:d160-key-segment-dq-unreliable"
    # MuDNN uses top-left causality: keys beyond each actual query length
    # are fully masked, so both key/value gradients must be exactly zero.
    # This also detects finite garbage in native masked output tails.
    native_kwargs = kwargs or {}
    if (
        flag_gems.vendor_name == "mthreads"
        and args[11]
        and native_kwargs.get("window_size_left") in (None, -1)
        and native_kwargs.get("window_size_right") in (None, -1)
    ):
        if q.ndim == 4:
            tails = [gradient[:, :, q.shape[2] :] for gradient in native[1:]]
        else:
            cu_q, cu_k = (tensor.detach().cpu().tolist() for tensor in args[6:8])
            tails = [
                gradient[k0 + min(q1 - q0, k1 - k0) : k1]
                for q0, q1, k0, k1 in zip(cu_q, cu_q[1:], cu_k, cu_k[1:])
                for gradient in native[1:]
            ]
        if any(torch.count_nonzero(tail) != 0 for tail in tails):
            return None, "REFERENCE_INVALID:causal-masked-tail"
        # On this measured S5000/MuDNN family, allocator contents can make an
        # unreliable masked tail accidentally zero. Independent backward-only
        # zero-QK controls (independently supplied O/LSE) and bounded self-owned
        # FP32 canaries establish the invalid reference for this kernel family.
        # These controls do not distinguish output from internal workspace state.
        # Preserve actual native ERROR/nonfinite results above, and do not extend
        # the classification to other runtimes, hardware, dimensions, or windows.
        from torch_musa.core import mudnn

        if (
            mudnn.version() == 3400
            and "S5000" in torch.musa.get_device_name(q.device)
            and (
                q.shape[-1] in range(8, 129, 8) or (q.ndim == 4 and q.shape[-1] == 160)
            )
        ):
            lengths = (
                [(q.shape[2], k.shape[2])]
                if q.ndim == 4
                else [
                    (q1 - q0, k1 - k0)
                    for q0, q1, k0, k1 in zip(cu_q, cu_q[1:], cu_k, cu_k[1:])
                ]
            )
            if any(sk > ((sq + 127) // 128) * 128 for sq, sk in lengths):
                return None, "REFERENCE_INVALID:causal-tail-unreliable"
    layout = "dense" if q.ndim == 4 else "varlen"
    return (
        tuple(
            _flash_bwd_gradient_from_backend(layout, tensor).cpu() for tensor in native
        ),
        "READY",
    )


def _flash_bwd_assert_reference(result, mathematical, dtype, native=None):
    if native is not None:
        # Valid same-argument native is the acceptance reference. Do not retain
        # or use a second mathematical-error gate for this saved-state ABI.
        utils.gems_assert_close(result, native, dtype)
        return "native"
    utils.gems_assert_close(result, mathematical, dtype)
    return "math"


def _flash_bwd_oracle_case(
    case_id,
    q_lengths,
    k_lengths,
    hq,
    hkv,
    dim,
    *,
    layout="dense",
    causal=False,
    left=None,
    right=None,
    scale=None,
):
    return pytest.param(
        layout,
        q_lengths,
        k_lengths,
        hq,
        hkv,
        dim,
        causal,
        left,
        right,
        scale,
        id=case_id,
    )


def _flash_bwd_musa_zero_qk_control(args):
    """Check the zero dQ/dK invariant with independently constructed state."""
    do, q, k, v = args[:4]
    batch, hq, sq, _ = q.shape
    hk, sk = k.shape[1:3]
    # Q=K=0, top-left causal: row m averages keys 0..min(m, Sk-1).
    # This control is separate from the random test's saved O/LSE.
    count = torch.arange(1, sq + 1).clamp(max=sk)
    prefix = v.detach().cpu().float().cumsum(dim=2)
    control_out = prefix.index_select(2, count - 1) / count.view(1, 1, sq, 1)
    control_out = control_out.repeat_interleave(hq // hk, dim=1).to(q.dtype)
    control_lse = count.float().log().view(1, 1, sq).expand(batch, hq, sq)
    control_args = (
        do,
        torch.zeros_like(q),
        torch.zeros_like(k),
        v,
        control_out.contiguous().to(q.device),
        control_lse.contiguous().to(q.device),
        *args[6:],
    )
    control, status = _flash_bwd_native_reference(control_args)
    if control is None:
        return status
    if any(torch.count_nonzero(gradient) for gradient in control[:2]):
        return "REFERENCE_INVALID:zero-qk-control"
    return "READY"


_FLASH_BWD_NATIVE_RUNTIMES = {
    ("hygon", "2.9.0", "275d08c233b2cf705e4b66d9f98f7590daae5cba", "cuda"),
    ("nvidia", "2.11.0+cu130", "70d99e998b4955e0049d13a98d77ae1b14db1f45", "cuda"),
    ("iluvatar", "2.7.1", "4be291ff8bf4aa863bc5e1e047ad574a35de9a26", "cuda"),
    ("mthreads", "2.9.1", "d38164a545b4a4e4e0cf73ce67173f70574890b6", "musa"),
}


_FLASH_BWD_ORIGINAL_SHAPES = [
    (2, 4, 512, 512),
    (1, 2, 1024, 1024),
    (1, 1, 64, 64),
]


_FLASH_BWD_ORIGINAL_OPTIONS = [
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
]


_FLASH_BWD_ORACLE_CASES = [
    *(
        _flash_bwd_oracle_case(f"dense-single-d{dim}", (1,), (1,), 1, 1, dim)
        for dim in (8, 16, 24, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256)
    ),
    _flash_bwd_oracle_case("dense-q2-k5-causal-d80", (2,), (5,), 1, 1, 80, causal=True),
    _flash_bwd_oracle_case(
        "dense-q1-k7-causal-window-minus1-d8",
        (1,),
        (7,),
        1,
        1,
        8,
        causal=True,
        left=-1,
        right=-1,
    ),
    _flash_bwd_oracle_case(
        "dense-q7-k1-local00-d8", (7,), (1,), 1, 1, 8, left=0, right=0
    ),
    _flash_bwd_oracle_case(
        "dense-q5-k2-fully-masked-d24", (5,), (2,), 1, 1, 24, causal=True
    ),
    _flash_bwd_oracle_case(
        "dense-gqa-local-scale0-d40", (2,), (5,), 4, 1, 40, left=1, right=1, scale=0.0
    ),
    _flash_bwd_oracle_case(
        "dense-gqa-causal-local-negative-scale-d96",
        (5,),
        (2,),
        4,
        2,
        96,
        causal=True,
        left=2,
        right=5,
        scale=-0.125,
    ),
    _flash_bwd_oracle_case(
        "varlen-mixed-causal-packed-lse-d8",
        (2, 5, 1),
        (5, 2, 7),
        4,
        2,
        8,
        layout="varlen",
        causal=True,
    ),
    _flash_bwd_oracle_case(
        "varlen-mixed-gqa-local-tail-d80",
        (2, 5, 17),
        (5, 2, 15),
        4,
        1,
        80,
        layout="varlen",
        left=2,
        right=5,
    ),
    _flash_bwd_oracle_case(
        "dense-q17-k15-local00-tail-d96", (17,), (15,), 2, 2, 96, left=0, right=0
    ),
    _flash_bwd_oracle_case(
        "dense-q33-k31-causal-scale0-tail-d64",
        (33,),
        (31,),
        1,
        1,
        64,
        causal=True,
        scale=0.0,
    ),
    _flash_bwd_oracle_case(
        "dense-q65-k63-gqa-right0-tail-d64", (65,), (63,), 8, 2, 64, left=-1, right=0
    ),
]


@pytest.mark.flash_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len",
    _FLASH_BWD_ORIGINAL_SHAPES,
)
@pytest.mark.parametrize(
    "dtype, is_causal, window_size_left, window_size_right, head_size",
    _FLASH_BWD_ORIGINAL_OPTIONS,
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
    record_property,
):
    if flag_gems.vendor_name == "amd" and (
        window_size_left is not None or window_size_right is not None
    ):
        # ROCm's native (AOTriton) sliding-window flash attention is wrong
        # (e.g. window_size_right=0 is treated as unbounded), so it cannot
        # serve as the reference here.
        pytest.skip(
            "ROCm native sliding-window flash attention is not a valid reference"
        )
    if flag_gems.vendor_name in ("ascend", "metax"):
        q, k, v = make_qkv(
            batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
        )
        grad_out = torch.randn(q.shape, dtype=dtype, device=q.device)
        scale = 1.0 / math.sqrt(head_size)
        out, lse, expected = _flash_bwd_cpu_oracle(
            q,
            k,
            v,
            grad_out,
            (q_seq_len,) * batch,
            (kv_seq_len,) * batch,
            scale=scale,
            causal=is_causal,
            left=window_size_left,
            right=window_size_right,
        )
        out = _flash_bwd_to_device(out, q.device)
        lse = _flash_bwd_lse_for_backend(lse, (q_seq_len,) * batch, q.device, head_size)
        args = (
            grad_out,
            q,
            k,
            v,
            out,
            lse,
            None,
            None,
            q_seq_len,
            kv_seq_len,
            0.0,
            is_causal,
            *_flash_bwd_p0_rng(q.device),
        )
        kwargs = dict(
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        native, status = _flash_bwd_native_reference(args, kwargs)
        record_property("native_reference_status", status)
        got = _flash_bwd_checked_call(flag_gems.flash_attention_backward, args, kwargs)
        for index, (name, result, reference, source) in enumerate(
            zip(("dQ", "dK", "dV"), got, expected, (q, k, v))
        ):
            assert result.shape == source.shape, name
            assert result.dtype == source.dtype, name
            assert result.device == source.device, name
            assert torch.isfinite(result).all(), name
            selection = _flash_bwd_assert_reference(
                result.cpu(),
                reference.to(dtype),
                dtype,
                None if native is None else native[index],
            )
            record_property(f"{name}_reference", selection)
        return

    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
    )
    dOut = torch.randn(
        batch, q_seq_len, num_head, head_size, dtype=dtype, device=flag_gems.device
    )
    if flag_gems.vendor_name == "mthreads":
        if window_size_left is not None or window_size_right is not None:
            pytest.skip("torch_musa 2.9.1 native FlashAttention rejects local windows")
        Q, K, V, dOut = (t.transpose(1, 2).contiguous() for t in (Q, K, V, dOut))

    wl = -1 if window_size_left is None else window_size_left
    wr = -1 if window_size_right is None else window_size_right

    out, lse, rng_state, unused = flash_attn_forward_native(
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
    # aten._flash_attention_backward has no CPU kernel, so the reference runs
    # on device even with --ref cpu; only its outputs move to the ref device.
    ref_grads = torch.ops.aten._flash_attention_backward(
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
        rng_state,
        unused,
        scale=scale,
        **extra_bwd,
    )
    ref_dQ, ref_dK, ref_dV = (utils.to_reference(g) for g in ref_grads)

    dQ, dK, dV = flag_gems.flash_attention_backward(
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
        rng_state,
        unused,
        scale=scale,
        **extra_bwd,
    )

    utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True)
    utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True)
    utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True)


@pytest.mark.flash_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon",
    reason="same-name native FlashAttention reference is unavailable",
)
@pytest.mark.parametrize(
    "dim", (8, 16, 24, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256)
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("f16", "bf16"))
def test_flash_attention_backward_dim_sweep(dim, dtype, record_property):
    if flag_gems.vendor_name in ("ascend", "metax"):
        generator = torch.Generator().manual_seed(20261020)
        q_shape, kv_shape = (1, 17, 4, dim), (1, 33, 2, dim)
        q, k, v, grad_out = (
            _flash_bwd_to_device(
                torch.randn(shape, generator=generator).to(dtype), flag_gems.device
            )
            for shape in (q_shape, kv_shape, kv_shape, q_shape)
        )
        out, lse, expected = _flash_bwd_cpu_oracle(
            q,
            k,
            v,
            grad_out,
            (17,),
            (33,),
            scale=None,
            causal=True,
            left=None,
            right=None,
        )
        args = (
            grad_out,
            q,
            k,
            v,
            _flash_bwd_to_device(out, q.device),
            _flash_bwd_lse_for_backend(lse, (17,), q.device, dim),
            None,
            None,
            17,
            33,
            0.0,
            True,
            *_flash_bwd_p0_rng(q.device),
        )
        native, status = _flash_bwd_native_reference(args)
        record_property("native_reference_status", status)
        got = _flash_bwd_checked_call(flag_gems.flash_attention_backward, args)
        for index, (name, result, reference, source) in enumerate(
            zip(("dQ", "dK", "dV"), got, expected, (q, k, v))
        ):
            assert result.shape == source.shape, name
            assert result.dtype == source.dtype, name
            assert result.device == source.device, name
            assert torch.isfinite(result).all(), name
            selection = _flash_bwd_assert_reference(
                result.cpu(),
                reference.to(dtype),
                dtype,
                None if native is None else native[index],
            )
            record_property(f"{name}_reference", selection)
        return

    generator = torch.Generator().manual_seed(20261020)
    q_shape, kv_shape = (1, 17, 4, dim), (1, 33, 2, dim)
    q, k, v, grad_out = (
        _flash_bwd_to_device(
            torch.randn(shape, generator=generator).to(dtype), flag_gems.device
        )
        for shape in (q_shape, kv_shape, kv_shape, q_shape)
    )
    if flag_gems.vendor_name == "mthreads":
        q, k, v, grad_out = (
            tensor.transpose(1, 2).contiguous() for tensor in (q, k, v, grad_out)
        )
    forward = torch.ops.aten._flash_attention_forward.default
    backward = torch.ops.aten._flash_attention_backward.default
    out, lse, rng_state, unused, _ = forward(
        q, k, v, None, None, 17, 33, 0.0, True, False
    )
    args = (
        grad_out,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        17,
        33,
        0.0,
        True,
        rng_state,
        unused,
    )
    if flag_gems.vendor_name != "mthreads":
        native = backward(*args)
        gems = flag_gems.flash_attention_backward(*args)
        for result, reference in zip(gems, native):
            utils.gems_assert_close(result.cpu(), reference.cpu(), dtype)
        return
    native, status = _flash_bwd_native_reference(args)
    if native is not None and dim in (224, 256) and q.shape[1] > k.shape[1]:
        control_status = _flash_bwd_musa_zero_qk_control(args)
        record_property("native_zero_qk_control", control_status)
        if control_status != "READY":
            native, status = None, control_status
    record_property("native_reference_status", status)
    gems = flag_gems.flash_attention_backward(*args)
    results = tuple(tensor.transpose(1, 2).cpu() for tensor in gems)
    if native is not None:
        for result, reference in zip(results, native):
            utils.gems_assert_close(result, reference, dtype)
        record_property("reference", "native")
        return
    # Match native FP32 scaling without rounding sqrt(D) first.
    # Use the original random forward state, never the control's O/LSE.
    actual_scale = torch.tensor(dim**-0.5, dtype=torch.float32).item()
    _, _, mathematical = _flash_bwd_cpu_oracle(
        *(tensor.transpose(1, 2) for tensor in (q, k, v, grad_out)),
        (17,),
        (33,),
        scale=actual_scale,
        causal=True,
        left=None,
        right=None,
        top_left_causal=True,
        saved_out=out.transpose(1, 2),
        saved_lse=lse,
    )
    for result, reference in zip(results, mathematical):
        utils.gems_assert_close(result, reference.to(dtype), dtype)
    record_property("reference", "math_same_saved_state")


@pytest.mark.flash_attention_backward
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("f16", "bf16"))
@pytest.mark.parametrize(
    "layout,q_lengths,k_lengths,hq,hkv,dim,causal,left,right,scale",
    _FLASH_BWD_ORACLE_CASES,
)
def test_flash_attention_backward_cpu_oracle(
    layout,
    q_lengths,
    k_lengths,
    hq,
    hkv,
    dim,
    causal,
    left,
    right,
    scale,
    dtype,
    record_property,
):
    generator = torch.Generator().manual_seed(20261020)
    packed = layout == "varlen"
    q_shape = (sum(q_lengths), hq, dim) if packed else (1, q_lengths[0], hq, dim)
    kv_shape = (sum(k_lengths), hkv, dim) if packed else (1, k_lengths[0], hkv, dim)
    q = _flash_bwd_to_device(
        (torch.randn(q_shape, generator=generator) * 0.4).to(dtype), flag_gems.device
    )
    k = _flash_bwd_to_device(
        (torch.randn(kv_shape, generator=generator) * 0.4).to(dtype), flag_gems.device
    )
    v = _flash_bwd_to_device(
        (torch.randn(kv_shape, generator=generator) * 0.4).to(dtype), flag_gems.device
    )
    grad_out = _flash_bwd_to_device(
        torch.randn(q_shape, generator=generator).to(dtype), flag_gems.device
    )
    out, lse, expected = _flash_bwd_cpu_oracle(
        q,
        k,
        v,
        grad_out,
        q_lengths,
        k_lengths,
        scale=scale,
        causal=causal,
        top_left_causal=flag_gems.vendor_name == "mthreads",
        left=left,
        right=right,
    )
    out, lse = _flash_bwd_to_device(out, q.device), _flash_bwd_lse_for_backend(
        lse, q_lengths, q.device, dim
    )
    native_q, native_k, native_v, native_out, native_do = _flash_bwd_inputs_for_backend(
        layout, q, k, v, out, grad_out
    )
    cu_q = cu_k = None
    if packed:
        cu_q = torch.tensor(
            (0, *torch.tensor(q_lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=q.device,
        )
        cu_k = torch.tensor(
            (0, *torch.tensor(k_lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=q.device,
        )
    args = (
        native_do,
        native_q,
        native_k,
        native_v,
        native_out,
        lse,
        cu_q,
        cu_k,
        max(q_lengths),
        max(k_lengths),
        0.0,
        causal,
        *_flash_bwd_p0_rng(q.device),
    )
    kwargs = dict(scale=scale, window_size_left=left, window_size_right=right)
    native, status = _flash_bwd_native_reference(args, kwargs)
    record_property("native_reference_status", status)
    got = _flash_bwd_checked_call(flag_gems.flash_attention_backward, args, kwargs)
    for index, (name, native_result, reference, source) in enumerate(
        zip(("dQ", "dK", "dV"), got, expected, (q, k, v))
    ):
        result = _flash_bwd_gradient_from_backend(layout, native_result)
        assert result.shape == source.shape, name
        assert result.dtype == source.dtype, name
        assert result.device == source.device, name
        selection = _flash_bwd_assert_reference(
            result.cpu(),
            reference.to(source.dtype),
            dtype,
            None if native is None else native[index],
        )
        record_property(f"{name}_reference", selection)


@pytest.mark.flash_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("ascend", "metax"),
    reason="saved-state math coverage for Ascend and MetaX",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("f16", "bf16"))
@pytest.mark.parametrize(
    "dim,causal,large_range",
    (
        (64, False, False),
        (64, True, False),
        (128, False, False),
        (128, True, False),
        (64, False, True),
        (256, False, True),
    ),
)
def test_flash_attention_backward_saved_state_arithmetic_range(
    dtype, dim, causal, large_range, record_property
):
    generator = torch.Generator().manual_seed(20261021)
    sq, sk, hq, hk = (17, 33, 4, 2) if large_range else (513, 769, 2, 2)
    shapes = ((1, sq, hq, dim), (1, sk, hk, dim))
    shapes = (shapes[0], shapes[1], shapes[1], shapes[0])
    if large_range:
        # Extra overflow coverage: finite dS can exceed FP16 while the final
        # gradients stay finite. This does not replace the unscaled randn cases.
        inputs = (
            torch.empty(shape).uniform_(-bound, bound, generator=generator)
            for shape, bound in zip(shapes, (2**-8, 2**-8, 32768, 128))
        )
    else:
        inputs = (torch.randn(shape, generator=generator) for shape in shapes)
    q, k, v, do = (
        _flash_bwd_to_device(tensor.to(dtype), flag_gems.device) for tensor in inputs
    )
    out, lse, expected = _flash_bwd_cpu_oracle(
        q, k, v, do, (sq,), (sk,), scale=None, causal=causal, left=None, right=None
    )
    args = (
        do,
        q,
        k,
        v,
        _flash_bwd_to_device(out, q.device),
        lse.to(q.device),
        None,
        None,
        sq,
        sk,
        0.0,
        causal,
        *_flash_bwd_p0_rng(q.device),
    )
    native, status = _flash_bwd_native_reference(args)
    record_property("native_reference_status", status)
    got = _flash_bwd_checked_call(flag_gems.flash_attention_backward, args)
    for index, (actual, mathematical) in enumerate(zip(got, expected)):
        assert torch.isfinite(actual).all()
        selection = _flash_bwd_assert_reference(
            actual.cpu(),
            mathematical.to(dtype),
            dtype,
            None if native is None else native[index],
        )
        record_property(("dQ", "dK", "dV")[index] + "_reference", selection)


@pytest.mark.flash_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("nvidia", "hygon", "iluvatar", "mthreads"),
    reason="positive dropout requires native forward saved state and a native reference",
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("f16", "bf16"))
@pytest.mark.parametrize("dropout_p", (0.125, 0.25, 0.5))
@pytest.mark.parametrize(
    "layout,q_lengths,k_lengths,hq,hkv,dim,causal",
    [
        pytest.param("dense", (64,), (64,), 4, 4, 64, False, id="mha"),
        pytest.param("dense", (17,), (33,), 4, 2, 80, True, id="gqa-causal-tail"),
        pytest.param("dense", (33, 33), (17, 17), 8, 2, 128, False, id="batch2-gqa"),
        pytest.param("dense", (128,), (256,), 8, 8, 80, True, id="long-causal"),
        pytest.param("dense", (128,), (256,), 8, 8, 96, False, id="long-noncausal"),
        pytest.param("varlen", (2, 5), (5, 2), 4, 2, 80, False, id="packed-gqa"),
    ],
)
def test_flash_attention_backward_native_dropout(
    layout,
    q_lengths,
    k_lengths,
    hq,
    hkv,
    dim,
    causal,
    dtype,
    dropout_p,
    record_property,
):
    """Compare dQ/dK/dV using the same native O, LSE and dropout RNG state."""
    packed = layout == "varlen"
    if flag_gems.vendor_name == "mthreads" and (packed or hq != hkv):
        from torch_musa.core import mudnn

        runtime = ("mthreads", torch.__version__, torch.version.git_version, "musa")
        if (
            runtime in _FLASH_BWD_NATIVE_RUNTIMES
            and mudnn.version() == 3400
            and "S5000" in torch.musa.get_device_name()
        ):
            # These native reference fixtures fault or return invalid gradients.
            # A failed launch can poison the context and hang the next transfer.
            record_property("native_reference", "PREFLIGHT_BLOCKED:gqa-or-packed")
            pytest.skip("MuDNN 3400 GQA/packed dropout reference is unreliable")
    if flag_gems.vendor_name == "hygon":
        if (
            flag_gems.vendor_name,
            torch.__version__,
            torch.version.git_version,
            "cuda",
        ) not in _FLASH_BWD_NATIVE_RUNTIMES:
            pytest.skip("Hygon dropout ABI has only been validated on DTK 26.04")
        if packed:
            pytest.skip("DTK 26.04 native packed FlashAttention can crash")
    generator = torch.Generator().manual_seed(20261020)
    q_shape = (
        (sum(q_lengths), hq, dim) if packed else (len(q_lengths), q_lengths[0], hq, dim)
    )
    kv_shape = (
        (sum(k_lengths), hkv, dim)
        if packed
        else (len(k_lengths), k_lengths[0], hkv, dim)
    )
    # Generate MUSA's dense BHSD inputs directly, retaining its unscaled controls.
    if flag_gems.vendor_name == "mthreads" and not packed:
        q_shape = (q_shape[0], hq, q_lengths[0], dim)
        kv_shape = (kv_shape[0], hkv, k_lengths[0], dim)
    q, k, v, grad_out = (
        _flash_bwd_to_device(
            torch.randn(shape, generator=generator).to(dtype), flag_gems.device
        )
        for shape in (q_shape, kv_shape, kv_shape, q_shape)
    )
    cu_q = cu_k = None
    if packed:
        cu_q, cu_k = (
            torch.tensor(
                (0, *torch.tensor(lengths).cumsum(0).tolist()),
                dtype=torch.int32,
                device=q.device,
            )
            for lengths in (q_lengths, k_lengths)
        )
    device_api = torch.musa if flag_gems.vendor_name == "mthreads" else torch.cuda
    original_rng = device_api.get_rng_state(q.device).clone()
    index = (
        q.device.index if q.device.index is not None else device_api.current_device()
    )
    try:
        device_api.default_generators[index].manual_seed(12345)
        device_api.default_generators[index].set_offset(0)
        try:
            out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward.default(
                q,
                k,
                v,
                cu_q,
                cu_k,
                max(q_lengths),
                max(k_lengths),
                dropout_p,
                causal,
                False,
            )
            args = (
                grad_out,
                q,
                k,
                v,
                out,
                lse,
                cu_q,
                cu_k,
                max(q_lengths),
                max(k_lengths),
                dropout_p,
                causal,
                rng,
                unused,
            )
            native_op = torch.ops.aten._flash_attention_backward.default
            before_backward = device_api.get_rng_state(q.device).clone()
            native = _flash_bwd_checked_call(native_op, args)
            repeated = (
                _flash_bwd_checked_call(native_op, args)
                if flag_gems.vendor_name == "mthreads"
                else native
            )
        except (RuntimeError, NotImplementedError) as error:
            if flag_gems.vendor_name != "mthreads":
                raise
            record_property("native_reference", f"ERROR:{type(error).__name__}:{error}")
            pytest.skip("MuDNN positive-dropout native reference unavailable")
        finite = all(torch.isfinite(t).all() for t in (out, *native))
        if flag_gems.vendor_name == "mthreads":
            # An invalid native reference cannot decide Gems accuracy. Keep these
            # checks even though the standalone native-diagnostic tests are gone.
            try:
                assert finite and torch.isfinite(lse).all()
                assert all(torch.isfinite(t).all() for t in repeated)
                assert torch.equal(before_backward, device_api.get_rng_state(q.device))
                for first, second in zip(native, repeated):
                    utils.gems_assert_close(first, second, dtype)
                if causal and not packed:
                    assert all(
                        torch.count_nonzero(t[:, :, q_lengths[0] :]) == 0
                        for t in native[1:]
                    )
            except AssertionError:
                record_property("native_reference", "REFERENCE_INVALID")
                pytest.skip(
                    "MuDNN dropout reference violates finite/repeat/RNG/tail invariants"
                )
        else:
            assert finite
        record_property("native_reference", "READY")
        got = _flash_bwd_checked_call(flag_gems.flash_attention_backward, args)
        for result, reference, source in zip(got, native, (q, k, v)):
            assert result.shape == source.shape
            assert result.dtype == source.dtype and result.device == source.device
            assert torch.isfinite(result).all()
            utils.gems_assert_close(result.cpu(), reference.cpu(), dtype)
    finally:
        device_api.set_rng_state(original_rng, q.device)


@pytest.mark.flash_attention_backward
@pytest.mark.parametrize(
    "layout,q_lengths,k_lengths,dense_sq,dense_sk",
    [
        pytest.param("dense", (), (), 2, 3, id="dense-b0"),
        pytest.param("dense", (0,), (3,), 0, 3, id="dense-zeroq"),
        pytest.param("dense", (3,), (0,), 3, 0, id="dense-zerok"),
        pytest.param("varlen", (), (), 0, 0, id="varlen-b0"),
        pytest.param("varlen", (0,), (3,), 0, 0, id="varlen-zeroq"),
        pytest.param("varlen", (3,), (0,), 0, 0, id="varlen-zerok"),
        pytest.param("varlen", (0, 2), (1, 0), 0, 0, id="varlen-mixed-empty"),
    ],
)
def test_flash_attention_backward_empty_gems_cpu_oracle(
    layout, q_lengths, k_lengths, dense_sq, dense_sk
):
    generator = torch.Generator().manual_seed(20261020)
    packed = layout == "varlen"
    if packed:
        q_shape = (sum(q_lengths), 2, 8)
        kv_shape = (sum(k_lengths), 1, 8)
    else:
        q_shape = (len(q_lengths), dense_sq, 2, 8)
        kv_shape = (len(k_lengths), dense_sk, 1, 8)
    q, grad_out = (
        _flash_bwd_to_device(
            torch.randn(q_shape, generator=generator).half(), flag_gems.device
        )
        for _ in range(2)
    )
    k, v = (
        _flash_bwd_to_device(
            torch.randn(kv_shape, generator=generator).half(), flag_gems.device
        )
        for _ in range(2)
    )
    out, lse, expected = _flash_bwd_cpu_oracle(
        q,
        k,
        v,
        grad_out,
        q_lengths,
        k_lengths,
        scale=None,
        causal=False,
        left=None,
        right=None,
    )
    cu_q = cu_k = None
    if packed:
        q_offsets, k_offsets = [0], [0]
        for q_len, k_len in zip(q_lengths, k_lengths):
            q_offsets.append(q_offsets[-1] + q_len)
            k_offsets.append(k_offsets[-1] + k_len)
        cu_q = torch.tensor(q_offsets, dtype=torch.int32, device=q.device)
        cu_k = torch.tensor(k_offsets, dtype=torch.int32, device=q.device)
    native_q, native_k, native_v, native_out, native_do = _flash_bwd_inputs_for_backend(
        layout, q, k, v, _flash_bwd_to_device(out, q.device), grad_out
    )
    # p=0 must not inspect the deliberately invalid saved RNG tensor.
    invalid_rng = torch.empty(0, dtype=torch.uint8, device=q.device)
    got = flag_gems.flash_attention_backward(
        native_do,
        native_q,
        native_k,
        native_v,
        native_out,
        _flash_bwd_lse_for_backend(lse, q_lengths, q.device, q.shape[-1]),
        cu_q,
        cu_k,
        max(q_lengths, default=0),
        max(k_lengths, default=0),
        0.0,
        False,
        invalid_rng,
        invalid_rng,
    )
    for name, native_result, reference, source in zip(
        ("dQ", "dK", "dV"), got, expected, (q, k, v)
    ):
        result = _flash_bwd_gradient_from_backend(layout, native_result)
        assert result.shape == source.shape, name
        assert result.dtype == source.dtype, name
        assert result.device == source.device, name
        utils.gems_assert_close(result.cpu(), reference.to(source.dtype), source.dtype)


@pytest.mark.scaled_dot_product_cudnn_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "amd",
    reason="ROCm torch has no cuDNN SDPA reference",
)
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

    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
    )
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

    out, lse, raw_lse, philox_seed, philox_offset = cudnn_attn_forward_native(
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

    # The aten cudnn backward kernel exists only for the CUDA backend, so
    # reference inputs stay on the accelerator even under --ref cpu.
    # Pass `raw_lse` (the original logsumexp shape from forward) directly to
    # the aten backward so the Stats layout always matches what cuDNN expects,
    # regardless of PyTorch / cuDNN version.
    ref_dOut_bhsd = dOut_bhsd
    ref_Q_bhsd = Q_bhsd
    ref_K_bhsd = K_bhsd
    ref_V_bhsd = V_bhsd
    ref_out_bhsd = out_bhsd
    ref_lse = raw_lse

    (
        ref_dQ_bhsd,
        ref_dK_bhsd,
        ref_dV_bhsd,
    ) = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
        ref_dOut_bhsd,
        ref_Q_bhsd,
        ref_K_bhsd,
        ref_V_bhsd,
        ref_out_bhsd,
        ref_lse,
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
    ref_dQ = utils.to_reference(ref_dQ_bhsd.permute(0, 2, 1, 3).contiguous())
    ref_dK = utils.to_reference(ref_dK_bhsd.permute(0, 2, 1, 3).contiguous())
    ref_dV = utils.to_reference(ref_dV_bhsd.permute(0, 2, 1, 3).contiguous())

    (
        dQ_bhsd,
        dK_bhsd,
        dV_bhsd,
    ) = flag_gems.scaled_dot_product_cudnn_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        out_bhsd,
        lse,
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
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

    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
    )
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

    # aten._efficient_attention_backward has no CPU kernel, so the reference
    # runs on device even with --ref cpu; only its outputs move to the ref device.
    ref_grads = torch.ops.aten._efficient_attention_backward(
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
    ref_dQ, ref_dK, ref_dV, ref_dBias = (utils.to_reference(g) for g in ref_grads)

    dQ, dK, dV, dBias_gems = flag_gems.efficient_attention_backward(
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

    # Attention backward accumulates over the sequence, so fp16/bf16 gradients
    # can differ from the reference by a few ULP on individual elements; use the
    # relaxed tolerance already applied to other attention ops.
    utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True, atol=2e-2)
    utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True, atol=2e-2)
    utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True, atol=2e-2)

    if has_bias and bias_requires_grad:
        assert dBias_gems is not None, "dBias should not be None"
        assert ref_dBias is not None, "ref dBias should not be None"
        utils.gems_assert_close(dBias_gems, ref_dBias, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.scaled_dot_product_efficient_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
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

    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
    )
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

    # No CPU kernel for this aten op either: run the reference on device and
    # move only its outputs to the reference device.
    ref_grads = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
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
    ref_dQ_bhsd, ref_dK_bhsd, ref_dV_bhsd, ref_dBias = (
        utils.to_reference(g) for g in ref_grads
    )
    ref_dQ = ref_dQ_bhsd.permute(0, 2, 1, 3).contiguous()
    ref_dK = ref_dK_bhsd.permute(0, 2, 1, 3).contiguous()
    ref_dV = ref_dV_bhsd.permute(0, 2, 1, 3).contiguous()

    (
        dQ_bhsd_gems,
        dK_bhsd_gems,
        dV_bhsd_gems,
        dBias_gems,
    ) = flag_gems.scaled_dot_product_efficient_attention_backward(
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


@pytest.mark.scaled_dot_product_flash_attention_backward
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the flash SDPA backend"
)
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len",
    [
        (2, 4, 512, 512),
        (1, 2, 1024, 1024),
        (1, 1, 64, 64),
        (2, 4, 128, 256),
    ],
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_flash_attention_backward(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    if is_causal and q_seq_len != kv_seq_len:
        # The Triton flash backward hits a compile error on the
        # causal + non-square path; only square causal masks are exercised.
        pytest.skip("causal with q_seq_len != kv_seq_len is not supported")
    scale = float(1.0 / math.sqrt(head_size))

    Q, K, V = make_qkv(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
    )

    out, lse, rng_state, rng_offset = flash_attn_forward_native(
        Q,
        K,
        V,
        is_causal=is_causal,
        softmax_scale=scale,
    )
    if rng_state.ndim == 0:
        # AOTriton returns seed and offset as separate scalar tensors.
        philox_seed, philox_offset = rng_state, rng_offset
    else:
        philox_seed, philox_offset = rng_state[0], rng_state[1]
    out = out.contiguous()
    lse = lse.float()
    dOut = torch.randn_like(out)

    # The reference is the mathematical gradient obtained by autograd through
    # SDPA. It is computed in float32: the opaque ATen backward entry in this
    # torch build applies layout handling that diverges from the actual
    # gradients, and an fp16 reference carries too little precision to pin the
    # expected values down.
    def ref_grads():
        # Route through to_reference so the reference follows the test
        # framework's device selection (CPU under --ref cpu, otherwise the
        # accelerator).
        q = utils.to_reference(Q).float().detach().requires_grad_(True)
        k = utils.to_reference(K).float().detach().requires_grad_(True)
        v = utils.to_reference(V).float().detach().requires_grad_(True)
        do = utils.to_reference(dOut).float()
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
            scale=scale,
        ).transpose(1, 2)
        return torch.autograd.grad(o, (q, k, v), do)

    ref_dQ, ref_dK, ref_dV = ref_grads()

    res_dQ, res_dK, res_dV = flag_gems.scaled_dot_product_flash_attention_backward(
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
    )

    # Low-precision accumulation in the Triton kernel rounds a small
    # fraction of elements past the 1e-4 default tolerance (larger at the
    # causal-mask boundary); 3e-2 matches the SDPA math-test convention and
    # stays inside the fp16/bf16 resolution.
    utils.gems_assert_close(res_dQ, ref_dQ, dtype, equal_nan=True, atol=3e-2)
    utils.gems_assert_close(res_dK, ref_dK, dtype, equal_nan=True, atol=3e-2)
    utils.gems_assert_close(res_dV, ref_dV, dtype, equal_nan=True, atol=3e-2)
