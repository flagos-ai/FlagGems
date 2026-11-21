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


from __future__ import annotations

import math
from functools import lru_cache
from itertools import accumulate, product
from typing import NamedTuple

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils

# The #9447 reference-routing fixtures in this file probe native attention
# kernels at collection time. On kunlunxin the native efficient-attention
# kernel rejects max_seqlen_q != max_seqlen_k (ValueError), and the GEMS XPU
# attention-backward kernels are not yet validated against the FP64 CPU
# oracle, so the whole suite is skipped on this backend (tracked upstream).
pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="XPU native attention probe/reference coverage unavailable",
)


@pytest.fixture(scope="module", autouse=flag_gems.vendor_name == "hygon")
def _hygon_reference_threads():
    # DTK exposes all host CPUs to PyTorch. The per-head FP64 reductions run
    # faster with one thread inside the CI container's CPU quota.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


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


def _flash_bwd_to_device(tensor, device):
    """Keep canonical host fixtures contiguous across MetaX's 4-D transfer."""
    if (
        flag_gems.vendor_name == "metax"
        and tensor.device.type == "cpu"
        and tensor.ndim == 4
    ):
        return tensor.flatten().to(device).reshape(tensor.shape)
    return tensor.to(device)


def _attention_mask(sq, sk, causal, top_left_causal, left, right):
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
    return valid


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
    bias=None,
    need_dbias=False,
    corex_state=False,
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
    bias_cpu = None if bias is None else bias.detach().cpu().double()
    dbias = torch.zeros_like(bias_cpu) if need_dbias else None
    do_cpu = None if grad_out is None else grad_out.detach().cpu().double()
    q_cpu, k_cpu, v_cpu = (tensor.detach().cpu().double() for tensor in (q, k, v))
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
        if do_cpu is not None:
            do_seq = do_cpu[q_start : q_start + sq] if packed else do_cpu[batch]
        valid = _attention_mask(sq, sk, causal, top_left_causal, left, right)
        for head in range(hq):
            kv_head = head // group
            scores = (q_seq[:, head] @ k_seq[:, kv_head].T) * actual_scale
            if bias_cpu is not None:
                scores += bias_cpu[batch, head]
            masked_scores = scores.masked_fill(~valid, -torch.inf)
            row_valid = valid.any(dim=1)
            if supplied_out is None:
                row_lse = torch.full((sq,), torch.inf, dtype=torch.float64)
                row_lse[row_valid] = torch.logsumexp(masked_scores[row_valid], dim=1)
                probs = torch.zeros_like(scores)
                probs[row_valid] = torch.exp(
                    masked_scores[row_valid] - row_lse[row_valid, None]
                )
                output = probs @ v_seq[:, kv_head]
                if packed:
                    out[q_start : q_start + sq, head] = output
                    lse[head, q_start : q_start + sq] = row_lse
                else:
                    out[batch, :, head] = output
                    lse[batch, head] = row_lse
                if do_cpu is None:
                    continue
                saved_output = output.to(q.dtype).double()
                saved_row_lse = row_lse.float().double()
                if corex_state and q.shape[-1] not in (32, 96, 160, 192, 256):
                    saved_row_lse = (
                        row_lse.float() * math.log2(math.e)
                    ).double() / math.log2(math.e)
            elif packed:
                saved_output = supplied_out[q_start : q_start + sq, head]
                saved_row_lse = supplied_lse[head, q_start : q_start + sq]
            else:
                saved_output = supplied_out[batch, :, head]
                saved_row_lse = supplied_lse[batch, head]
            probs = torch.zeros_like(scores)
            probs[row_valid] = torch.exp(
                masked_scores[row_valid] - saved_row_lse[row_valid, None]
            )
            dp = do_seq[:, head] @ v_seq[:, kv_head].T
            delta = (saved_output * do_seq[:, head]).sum(dim=-1)
            ds = probs * (dp - delta[:, None])
            if dbias is not None:
                dbias[batch, head] = ds
            dq_seq = actual_scale * (ds @ k_seq[:, kv_head])
            dk_seq = actual_scale * (ds.T @ q_seq[:, head])
            dv_seq = probs.T @ do_seq[:, head]
            if packed:
                dq[q_start : q_start + sq, head] = dq_seq
                dk[k_start : k_start + sk, kv_head] += dk_seq
                dv[k_start : k_start + sk, kv_head] += dv_seq
            else:
                dq[batch, :, head] = dq_seq
                dk[batch, :, kv_head] += dk_seq
                dv[batch, :, kv_head] += dv_seq
        q_start += sq
        k_start += sk
    if supplied_out is not None:
        out, lse = supplied_out, supplied_lse
    return (
        out.to(q.dtype),
        lse.float(),
        ((dq, dk, dv, dbias) if need_dbias else (dq, dk, dv)),
    )


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


# These names pair each public test with its native saved-state ABI.
_FLASH = "flash_attention_backward"
_CUDNN = "scaled_dot_product_cudnn_attention_backward"
_EFFICIENT = "efficient_attention_backward"
_SDP_EFFICIENT = "scaled_dot_product_efficient_attention_backward"
_SDP_FLASH = "scaled_dot_product_flash_attention_backward"
_FORWARD = {
    _FLASH: "_flash_attention_forward",
    _CUDNN: "_scaled_dot_product_cudnn_attention",
    _EFFICIENT: "_efficient_attention_forward",
    _SDP_EFFICIENT: "_scaled_dot_product_efficient_attention",
    _SDP_FLASH: "_scaled_dot_product_flash_attention",
}


class _Case(NamedTuple):
    op: str
    dtype: torch.dtype
    q_lengths: tuple
    k_lengths: tuple
    hq: int
    hk: int
    dim: int
    causal: bool
    packed: bool = False
    left: int | None = None
    right: int | None = None
    scale: float | None = None
    dropout: float = 0.0
    bias: bool = False
    mask: tuple = (True, True, True, False)
    strided: bool = False


class _State(NamedTuple):
    out: torch.Tensor  # canonical BSHD or packed THD
    lse: torch.Tensor  # raw backend layout for the native backward
    rng: torch.Tensor
    offset: torch.Tensor
    cu_q: torch.Tensor | None
    cu_k: torch.Tensor | None
    lse_to_ln: float = 1.0
    native_out: torch.Tensor | None = None


class _Route(NamedTuple):
    forward: bool
    backward: bool
    reason: str
    lse_to_ln: float = 1.0
    forward_family: str | None = None


def _layout(case, tensor, *, native, strided=False):
    if tensor is None or case.packed:
        return tensor
    bhsd = case.op in (_CUDNN, _SDP_EFFICIENT)
    if case.op == _SDP_FLASH:
        bhsd = native  # the public Gems wrapper takes BSHD
    if case.op == _FLASH:
        bhsd = flag_gems.vendor_name == "mthreads"
    if not bhsd:
        return tensor
    view = tensor.transpose(1, 2)
    return view if strided else view.contiguous()


def _offsets(lengths, device):
    return torch.tensor((0, *accumulate(lengths)), dtype=torch.int32, device=device)


def _window(case):
    return {
        key: value
        for key, value in (
            ("window_size_left", case.left),
            ("window_size_right", case.right),
        )
        if value is not None
    }


def _native_forward(case, q, k, v, bias, *, backward_op=None):
    fn = getattr(torch.ops.aten, _FORWARD[case.op]).default
    q, k, v = (_layout(case, t, native=True) for t in (q, k, v))
    sq, sk = max(case.q_lengths), max(case.k_lengths)
    cq = _offsets(case.q_lengths, q.device) if case.packed else None
    ck = _offsets(case.k_lengths, q.device) if case.packed else None
    if case.op == _FLASH:
        ret = fn(
            q,
            k,
            v,
            cq,
            ck,
            sq,
            sk,
            case.dropout,
            case.causal,
            False,
            scale=case.scale,
            **_window(case),
        )
    elif case.op == _EFFICIENT:
        ret = fn(
            q,
            k,
            v,
            bias,
            None,
            None,
            sq,
            sk,
            case.dropout,
            int(case.causal),
            True,
            scale=case.scale,
        )
    elif case.op == _SDP_FLASH:
        ret = fn(q, k, v, case.dropout, case.causal, False, scale=case.scale)
    else:
        ret = fn(q, k, v, bias, True, case.dropout, case.causal, scale=case.scale)
    # cuDNN and flash SDPA expose cu_seqlens before seed/offset in the
    # nine-result ABI; vendor builds also ship a four-result cuDNN ABI.
    if case.op in (_CUDNN, _SDP_FLASH) and len(ret) == 9:
        cq, ck = ret[2:4]
        rng, offset = ret[6:8]
    else:
        rng, offset = ret[2:4]
    native_out = ret[0]
    out = _layout(case, native_out, native=True)
    if backward_op is not None and backward_op != case.op:
        # A low-level flash forward can feed an SDPA backward with different axes.
        view = _layout(case, native_out, native=True, strided=True)
        native_out = _layout(
            case._replace(op=backward_op), view, native=True, strided=True
        )
    return _State(out, ret[1], rng, offset, cq, ck, native_out=native_out)


def _canonical_lse(case, state):
    lse = state.lse.detach().cpu()
    if case.op == _CUDNN and lse.ndim == 4:
        lse = lse.squeeze(-1)
    if case.packed and lse.ndim == 3:
        lse = torch.cat([row[:, :n] for row, n in zip(lse, case.q_lengths)], dim=1)
    if not case.packed:
        lse = lse[..., : max(case.q_lengths)]
    return lse.double() * state.lse_to_ln


def _math(case, q, k, v, do=None, bias=None, state=None):
    return _flash_bwd_cpu_oracle(
        q,
        k,
        v,
        do,
        case.q_lengths,
        case.k_lengths,
        scale=case.scale,
        causal=case.causal,
        left=case.left,
        right=case.right,
        top_left_causal=case.op != _FLASH or flag_gems.vendor_name == "mthreads",
        saved_out=None if state is None else state.out,
        saved_lse=None if state is None else _canonical_lse(case, state),
        corex_state=case.op == _FLASH and flag_gems.vendor_name == "iluvatar",
        bias=bias,
        need_dbias=case.mask[3] and bias is not None,
    )


def _math_state(case, q, k, v, bias):
    out, lse, _ = _math(case, q, k, v, bias=bias)
    if case.op == _FLASH:
        lse = _flash_bwd_lse_for_backend(lse, case.q_lengths, q.device, case.dim)
    else:
        if case.op in (_EFFICIENT, _SDP_EFFICIENT):
            lse = torch.nn.functional.pad(lse, (0, -lse.shape[-1] % 32), value=math.inf)
        lse = lse.to(q.device)
    rng, offset = (
        torch.zeros((), dtype=torch.int64, device=q.device) for _ in range(2)
    )
    if case.op == _FLASH:
        rng, offset = _flash_bwd_p0_rng(q.device)
    return _State(
        _flash_bwd_to_device(out, q.device),
        lse,
        rng,
        offset,
        _offsets(case.q_lengths, q.device) if case.packed else None,
        _offsets(case.k_lengths, q.device) if case.packed else None,
        (
            (-1.0 if case.dim in (32, 96, 160, 192, 256) else -math.log(2))
            if case.op == _FLASH and flag_gems.vendor_name == "iluvatar"
            else 1.0
        ),
    )


def _backward(case, q, k, v, do, bias, state, *, native):
    fn = (
        getattr(torch.ops.aten, "_" + case.op).default
        if native
        else getattr(flag_gems, case.op)
    )
    args = tuple(
        _layout(case, t, native=native, strided=case.strided) for t in (do, q, k, v)
    )
    if native and state.native_out is not None:
        out = state.native_out
    else:
        out = _layout(case, state.out, native=native, strided=case.strided)
    lse, rng, offset, cq, ck = state[1:6]
    if not native and not case.packed:
        cq = ck = None
    sq, sk = max(case.q_lengths), max(case.k_lengths)
    kw = {"scale": case.scale}
    if case.op in (_FLASH, _SDP_FLASH):
        if case.op == _FLASH:
            kw.update(_window(case))
        args += (out, lse, cq, ck, sq, sk, case.dropout, case.causal, rng, offset)
    elif case.op == _CUDNN:
        if not native and lse.ndim == 4:
            lse = lse.squeeze(-1)
        args += (out, lse, rng, offset, bias, cq, ck, sq, sk, 0.0, case.causal)
    elif case.op == _EFFICIENT:
        args += (
            bias,
            out,
            None,
            None,
            sq,
            sk,
            lse,
            0.0,
            rng,
            offset,
            int(case.causal),
            case.mask[3],
        )
    else:
        args += (bias, out, lse, rng, offset, 0.0, case.mask, case.causal)
    if case.op == _FLASH:
        result = _flash_bwd_checked_call(fn, args, kw)
    else:
        result = fn(*args, **kw)
    return tuple(_layout(case, t, native=native) for t in result[:3]) + tuple(
        result[3:]
    )


def _saved_state(case, route, q, k, v, bias):
    if not route.forward:
        return _math_state(case, q, k, v, bias)
    source = case._replace(op=route.forward_family or case.op)
    state = _native_forward(source, q, k, v, bias, backward_op=case.op)
    if case.op == _SDP_FLASH and route.lse_to_ln != 1.0:
        return state._replace(lse=state.lse * route.lse_to_ln)
    return state._replace(lse_to_ln=route.lse_to_ln)


def _reference(case, route, q, k, v, do, bias=None, state=None):
    if state is None:
        state = _saved_state(case, route, q, k, v, bias)
    if route.backward:
        expected = _backward(case, q, k, v, do, bias, state, native=True)
    else:
        assert case.dropout == 0, "no independent dropout-mask mapping for this case"
        _, _, expected = _math(case, q, k, v, do, bias, state)
    actual = _backward(case, q, k, v, do, bias, state, native=False)
    if case.op != _FLASH:
        expected = tuple(utils.to_reference(g) for g in expected)
    return actual, expected


def _assert_flash_gradients(got, expected, sources, dtype):
    for name, result, ref, src in zip(("dQ", "dK", "dV"), got, expected, sources):
        assert result.shape == src.shape and result.dtype == src.dtype, name
        assert result.device == src.device and torch.isfinite(result).all(), name
        utils.gems_assert_close(result.cpu(), ref.cpu().to(dtype), dtype)


def _case(op, p):
    """Translate the existing pytest axes to a saved-state ABI."""
    if "q_seq_len" in p:
        return _Case(
            op,
            p["dtype"],
            (p["q_seq_len"],) * p["batch"],
            (p["kv_seq_len"],) * p["batch"],
            p["num_head"],
            p["num_head"],
            p["head_size"],
            p.get("is_causal", p.get("expected_causal", False)),
            left=p.get("window_size_left"),
            right=p.get("window_size_right"),
            scale=1.0 / math.sqrt(p["head_size"]),
            bias=p.get("has_attn_bias", p.get("has_bias", False)),
            mask=p.get(
                "grad_input_mask",
                (
                    True,
                    True,
                    True,
                    op == _EFFICIENT and p.get("bias_requires_grad", False),
                ),
            ),
        )
    if "q_lengths" in p:
        return _Case(
            op,
            p["dtype"],
            p["q_lengths"],
            p["k_lengths"],
            p["hq"],
            p["hkv"],
            p["dim"],
            p["causal"],
            p["layout"] == "varlen",
            p.get("left"),
            p.get("right"),
            p.get("scale"),
            p.get("dropout_p", 0.0),
            strided="dropout_p" not in p and flag_gems.vendor_name == "mthreads",
        )
    if "large_range" in p and not p["large_range"]:
        return _Case(op, p["dtype"], (513,), (769,), 2, 2, p["dim"], p["causal"])
    return _Case(op, p["dtype"], (17,), (33,), 4, 2, p["dim"], p.get("causal", True))


@lru_cache(None)
def _registered(name):
    op = getattr(torch.ops.aten, name, None)
    if op is None:
        return False
    key = torch._C._dispatch_key_for_device(torch.device(flag_gems.device).type)
    return torch._C._dispatch_has_computed_kernel_for_dispatch_key("aten::" + name, key)


def _try_native(call):
    # An unavailable dispatch/build/shape can reject the fixture. Surface
    # device faults, allocation errors and bugs; never change routes mid-test.
    try:
        result = call()
        torch_device_fn.synchronize()
        return result, ""
    except (RuntimeError, ValueError) as exc:
        # ATen uses RuntimeError for absent compiled features as well as schema
        # and input constraints; vendor native kernels also signal unsupported
        # shapes with ValueError (e.g. the XPU efficient-attention kernel
        # rejects max_seqlen_q != max_seqlen_k). A failed probe is recorded on
        # the test node. Device faults invalidate subsequent probes on the same
        # context.
        message = str(exc)
        if isinstance(exc, getattr(torch, "OutOfMemoryError", MemoryError)) or any(
            token in message.lower()
            for token in (
                "out of memory",
                "illegal memory",
                "illegal address",
                "misaligned address",
                "device-side assert",
                "device lost",
                "device has been lost",
                "launch failure",
            )
        ):
            raise
        return None, message.splitlines()[0] if message else type(exc).__name__


def _uniform_control(case, score):
    # Identical logits yield 1/n on each of a row's n visible keys. With V=0
    # and dOut=1, dQ=dK=0 and dV is the column sum for each group of heads.
    rows, columns = [], []
    for sq, sk in zip(case.q_lengths, case.k_lengths):
        mask = _attention_mask(
            sq,
            sk,
            case.causal,
            case.op != _FLASH or flag_gems.vendor_name == "mthreads",
            case.left,
            case.right,
        )
        count = mask.sum(1).double()
        rows.append((count.log() + score).expand(case.hq, -1))
        columns.append(
            (mask / count.clamp_min(1)[:, None]).sum(0) * (case.hq // case.hk)
        )
    lse = torch.cat(rows, 1) if case.packed else torch.stack(rows)
    dv = torch.cat(columns) if case.packed else torch.stack(columns)
    return lse, dv[..., None, None]


@lru_cache(None)
def _probe(case, forward_family=None):
    # Keep hazardous vendor kernels out of the import process. No independent
    # CPU dropout mask has been validated for any of the currently tested ABIs.
    vendor = flag_gems.vendor_name
    if vendor == "hygon" and case.packed:
        return _Route(False, False, "DTK packed FlashAttention can fault")
    if (
        case.dropout
        and vendor == "hygon"
        and torch.version.git_version != "275d08c233b2cf705e4b66d9f98f7590daae5cba"
    ):
        return _Route(False, False, "unverified Hygon dropout ABI")
    mudnn3400 = False
    if case.op == _FLASH and vendor == "mthreads":
        from torch_musa.core import mudnn

        mudnn3400 = mudnn.version() == 3400 and "S5000" in torch.musa.get_device_name()
        if mudnn3400 and case.dropout and (case.packed or case.hq != case.hk):
            return _Route(False, False, "MuDNN GQA/packed dropout reference can fault")
    source = case._replace(op=forward_family or case.op)
    forward = _registered(_FORWARD[source.op])
    backward = _registered("_" + case.op)
    if not forward and not backward:
        return _Route(False, False, "no device dispatch")
    if case.packed:
        qs = (sum(case.q_lengths), case.hq, case.dim)
        ks = (sum(case.k_lengths), case.hk, case.dim)
    else:
        qs = (len(case.q_lengths), case.q_lengths[0], case.hq, case.dim)
        ks = (len(case.k_lengths), case.k_lengths[0], case.hk, case.dim)
    q, k = (torch.zeros(s, dtype=case.dtype, device=flag_gems.device) for s in (qs, ks))
    if source.op == _FLASH and vendor == "iluvatar":
        # Distinguish ln from log2 even in a one-key row (log(1) == 0).
        q.fill_(0.25)
        k.fill_(0.25)
    v, do = torch.zeros_like(k), torch.ones_like(q)
    bias = q.new_zeros((qs[0], case.hq, qs[1], ks[1])) if case.bias else None
    state, why_f = None, "no forward dispatch"
    if forward:
        state, why_f = _try_native(
            lambda: _native_forward(source, q, k, v, bias, backward_op=case.op)
        )
    lse_to_ln = 1.0
    corex = vendor == "iluvatar" and source.op == _FLASH
    scale = case.dim**-0.5 if case.scale is None else case.scale
    target_lse, target_dv = _uniform_control(
        case, case.dim * scale / 16 if corex else 0.0
    )
    if state is not None:
        if (
            not isinstance(state.out, torch.Tensor)
            or state.out.shape != q.shape
            or state.out.device != q.device
            or state.out.dtype != q.dtype
            or not isinstance(state.lse, torch.Tensor)
            or state.lse.device != q.device
            or torch.count_nonzero(state.out)
        ):
            state, why_f = None, "invalid native forward output"
        else:
            # CoreX releases can select different log bases at the same D.
            observed = _canonical_lse(case, state)
            valid = torch.isfinite(target_lse)
            factors = (-1.0, -math.log(2), 1.0, math.log(2)) if corex else (1.0,)
            for factor in factors:
                if observed.shape == target_lse.shape and torch.allclose(
                    (observed * factor)[valid], target_lse[valid], atol=2e-5, rtol=2e-5
                ):
                    lse_to_ln = factor
                    break
            else:
                state, why_f = None, "native saved LSE disagrees with the probe mask"
    forward = state is not None
    if not forward and case.dropout:
        return _Route(False, False, why_f, lse_to_ln)
    if (
        mudnn3400
        and case.causal
        and case.dim in (224, 256)
        and case.q_lengths == (17,)
        and case.k_lengths == (33,)
        and (case.hq, case.hk) == (4, 2)
    ):
        backward = False  # ragged GQA tail can return finite garbage
    if not backward:
        return _Route(forward, False, why_f or "no eligible native backward", lse_to_ln)
    if state is None:
        state = _math_state(case, q, k, v, bias)
    elif case.op == _SDP_FLASH and lse_to_ln != 1.0:
        state = state._replace(lse=state.lse * lse_to_ln)
    result, why_b = _try_native(
        lambda: _backward(case, q, k, v, do, bias, state, native=True)
    )
    if result is None:
        return _Route(forward, False, why_b, lse_to_ln)
    # V=0 gives exactly zero dQ/dK. With p=0, a unit upstream
    # gradient also checks layout, causal alignment, GQA reduction and dV.
    grads = result[:3]
    for enabled, grad, inp in zip(case.mask, grads, (q, k, v)):
        if enabled and (
            grad is None
            or grad.shape != inp.shape
            or grad.device != inp.device
            or grad.dtype != inp.dtype
            or not torch.isfinite(grad).all()
        ):
            return _Route(forward, False, "invalid native backward output", lse_to_ln)
    if any(
        enabled and torch.count_nonzero(g)
        for enabled, g in zip(case.mask[:2], grads[:2])
    ):
        return _Route(forward, False, "invalid zero-QK native gradients", lse_to_ln)
    if case.mask[2] and not case.dropout:
        expected = target_dv.to(case.dtype).expand(grads[2].shape)
        if not torch.allclose(
            grads[2].cpu(),
            expected,
            atol=1e-4,
            rtol=0.001 if case.dtype == torch.float16 else 0.016,
        ):
            return _Route(
                forward, False, "invalid native probability or dV layout", lse_to_ln
            )
    return _Route(forward, True, why_f, lse_to_ln)


def _probe_shape(case):
    # Probe dense long-sequence families on a small aligned tile; retain
    # rectangular orientation. Keep short, varlen and dropout fixtures exact.
    if (
        not case.packed
        and not case.dropout
        and min(case.q_lengths + case.k_lengths) >= 64
    ):
        sq, sk = case.q_lengths[0], case.k_lengths[0]
        if sq % 64 or sk % 64:
            return case
        batch = min(2, len(case.q_lengths))
        return case._replace(
            q_lengths=(64 if sq <= sk else 128,) * batch,
            k_lengths=(64 if sk <= sq else 128,) * batch,
        )
    return case


def _case_route(case):
    representative = _probe_shape(case)
    route = _probe(representative)
    # The existing flash SDPA fixtures can also consume a low-level flash
    # forward. Some builds only register that entry (e.g. torch_musa).
    if case.op == _SDP_FLASH and not route.forward and _registered(_FORWARD[_FLASH]):
        alternative = _probe(representative, _FLASH)
        if alternative.forward:
            return alternative._replace(forward_family=_FLASH)
    return route


def _plan_references(op):
    """Enumerate the parametrization pytest collects, without any device work."""

    def decorate(fn):
        backend_skipped = any(
            m.name == "skipif" and m.args and m.args[0] is True for m in fn.pytestmark
        )
        axes = []
        for mark in fn.pytestmark:
            if mark.name != "parametrize":
                continue
            names = [name.strip() for name in mark.args[0].split(",")]
            axis = []
            for param in mark.args[1]:
                values = getattr(param, "values", param)
                if len(names) == 1:
                    values = (param,) if not hasattr(param, "values") else values
                axis.append(dict(zip(names, values)))
            axes.append(axis)
        cases = set()
        for row in () if backend_skipped else product(*axes):
            params = {k: v for axis in row for k, v in axis.items()}
            if (
                params.get("bias_requires_grad")
                and not params.get("has_attn_bias", params.get("has_bias", False))
                or params.get("is_causal")
                and params.get("has_attn_bias")
                or params.get("grad_input_mask", (False,) * 4)[3]
                and not params.get("has_attn_bias")
                or op == _SDP_FLASH
                and params["is_causal"]
                and params["q_seq_len"] != params["kv_seq_len"]
            ):
                continue
            cases.add(_case(op, params))
        fn.attention_cases = frozenset(cases)
        fn.attention_op = op
        return fn

    return decorate


@lru_cache(None)
def _planned_route(case):
    """Probe a planned case on first use, at most once per process.

    Importing this module happens for every ``pytest tests/`` invocation, even
    one whose marker deselects every test here, because pytest imports a module
    before it applies ``-m``. Probing while building the plans therefore charged
    each per-operator run several hundred native attention launches, so the
    probe is deferred to the fixture that consumes the route.
    """
    # Vendor dropout forward advances the generator, even on a failed probe.
    # Restore the caller's state so a test's own inputs stay reproducible.
    with torch.random.fork_rng(
        devices=[torch_device_fn.current_device()],
        device_type=torch.device(flag_gems.device).type,
    ):
        return _case_route(case)


@pytest.fixture
def attention_reference(request, record_property):
    case = _case(request.function.attention_op, request.node.callspec.params)
    # Invalid flag combinations are skipped by the test before using its route.
    route = _planned_route(case) if case in request.function.attention_cases else None
    if route is not None:
        fallback = "unavailable" if case.dropout else "cpu_fp64"
        record_property("forward_reference", "native" if route.forward else fallback)
        record_property(
            "backward_reference",
            "native" if route.backward else fallback,
        )
        if route.forward:
            record_property(
                "native_forward_op", _FORWARD[route.forward_family or case.op]
            )
        if route.reason:
            record_property("reference_probe", route.reason)
    return case, route


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


@_plan_references(_FLASH)
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
    attention_reference,
):

    case, route = attention_reference
    q, k, v = make_qkv(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
    )
    do = torch.randn(q.shape, dtype=dtype, device=q.device)
    got, expected = _reference(case, route, q, k, v, do)
    if flag_gems.vendor_name in ("ascend", "metax"):
        _assert_flash_gradients(got, expected, (q, k, v), dtype)
        return
    for result, ref in zip(got, expected):
        utils.gems_assert_close(result, utils.to_reference(ref), dtype, equal_nan=True)


@_plan_references(_FLASH)
@pytest.mark.flash_attention_backward
@pytest.mark.parametrize(
    "dim", (8, 16, 24, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256)
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("f16", "bf16"))
def test_flash_attention_backward_dim_sweep(dim, dtype, attention_reference):
    case, route = attention_reference

    generator = torch.Generator().manual_seed(20261020)
    q_shape, kv_shape = (1, 17, 4, dim), (1, 33, 2, dim)
    q, k, v, grad_out = (
        _flash_bwd_to_device(
            torch.randn(shape, generator=generator).to(dtype), flag_gems.device
        )
        for shape in (q_shape, kv_shape, kv_shape, q_shape)
    )
    got, expected = _reference(case, route, q, k, v, grad_out)

    _assert_flash_gradients(got, expected, (q, k, v), dtype)


@_plan_references(_FLASH)
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
    attention_reference,
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
    case, route = attention_reference
    got, expected = _reference(case, route, q, k, v, grad_out)

    _assert_flash_gradients(got, expected, (q, k, v), dtype)


@_plan_references(_FLASH)
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
    dtype, dim, causal, large_range, attention_reference
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
    case, route = attention_reference
    got, expected = _reference(case, route, q, k, v, do)

    _assert_flash_gradients(got, expected, (q, k, v), dtype)


@_plan_references(_FLASH)
@pytest.mark.flash_attention_backward
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
    attention_reference,
):

    case, route = attention_reference
    if not (route.forward and route.backward):
        pytest.skip(
            "native dropout pair unavailable; no verified CPU mask: " + route.reason
        )
    packed = layout == "varlen"
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

    if flag_gems.vendor_name == "mthreads" and not packed:
        q, k, v, grad_out = (t.transpose(1, 2) for t in (q, k, v, grad_out))
    device_api = torch_device_fn
    with torch.random.fork_rng(
        devices=[device_api.current_device()], device_type=q.device.type
    ):
        generator = device_api.default_generators[device_api.current_device()]
        generator.manual_seed(12345)
        generator.set_offset(0)
        state = _native_forward(case, q, k, v, None)
        before = device_api.get_rng_state(q.device).clone()
        got, expected = _reference(case, route, q, k, v, grad_out, state=state)
        assert torch.isfinite(state.out).all()
        if flag_gems.vendor_name == "mthreads":
            repeated = _backward(case, q, k, v, grad_out, None, state, native=True)
            assert torch.equal(before, device_api.get_rng_state(q.device))
            assert torch.isfinite(state.lse).all()
            for first, second in zip(expected, repeated):
                utils.gems_assert_close(first, second, dtype)
            if causal and not packed:
                assert all(
                    torch.count_nonzero(g[:, q_lengths[0] :]) == 0 for g in expected[1:]
                )

    _assert_flash_gradients(got, expected, (q, k, v), dtype)


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
    layout, q_lengths, k_lengths, dense_sq, dense_sk, record_property
):
    record_property("forward_reference", "cpu_fp64")
    record_property("backward_reference", "cpu_fp64")
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


@_plan_references(_CUDNN)
@pytest.mark.scaled_dot_product_cudnn_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
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
    attention_reference,
):
    if is_causal and has_attn_bias:
        pytest.skip("causal + attn_bias combination: skip")
    if bias_requires_grad and not has_attn_bias:
        pytest.skip("bias_requires_grad=True requires has_attn_bias=True")

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

    case, route = attention_reference
    (dQ, dK, dV), (ref_dQ, ref_dK, ref_dV) = _reference(
        case, route, Q, K, V, dOut, attn_bias
    )
    utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True)
    utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True)
    utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True)


@_plan_references(_EFFICIENT)
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
    attention_reference,
):
    if not has_bias and bias_requires_grad:
        pytest.skip("bias_requires_grad=True requires has_bias=True")

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

    case, route = attention_reference
    result, expected = _reference(case, route, Q, K, V, dOut, bias)
    dQ, dK, dV, dBias_gems = result
    ref_dQ, ref_dK, ref_dV = expected[:3]
    ref_dBias = expected[3] if len(expected) == 4 else None
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


@_plan_references(_SDP_EFFICIENT)
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
    attention_reference,
):
    need_dq, need_dk, need_dv, need_dbias = grad_input_mask

    if is_causal and has_attn_bias:
        pytest.skip("causal + attn_bias: skip")
    if need_dbias and not has_attn_bias:
        pytest.skip("need_dbias=True requires has_attn_bias=True")

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

    case, route = attention_reference
    result, expected = _reference(case, route, Q, K, V, dOut, attn_bias)
    dQ, dK, dV, dBias_gems = result
    ref_dQ, ref_dK, ref_dV = expected[:3]
    ref_dBias = expected[3] if len(expected) == 4 else None
    if need_dq:
        utils.gems_assert_close(dQ, ref_dQ, dtype, equal_nan=True)
    else:
        assert torch.all(dQ == 0), (
            f"dQ should be zero when need_dq=False, "
            f"got max abs={dQ.abs().max().item():.6f}"
        )

    if need_dk:
        utils.gems_assert_close(dK, ref_dK, dtype, equal_nan=True)
    else:
        assert torch.all(dK == 0), (
            f"dK should be zero when need_dk=False, "
            f"got max abs={dK.abs().max().item():.6f}"
        )

    if need_dv:
        utils.gems_assert_close(dV, ref_dV, dtype, equal_nan=True)
    else:
        assert torch.all(dV == 0), (
            f"dV should be zero when need_dv=False, "
            f"got max abs={dV.abs().max().item():.6f}"
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


@_plan_references(_SDP_FLASH)
@pytest.mark.scaled_dot_product_flash_attention_backward
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
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    dtype,
    attention_reference,
):
    if is_causal and q_seq_len != kv_seq_len:
        # The Triton flash backward hits a compile error on the
        # causal + non-square path; only square causal masks are exercised.
        pytest.skip("causal with q_seq_len != kv_seq_len is not supported")

    Q, K, V = make_qkv(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, flag_gems.device
    )

    case, route = attention_reference
    state = _saved_state(case, route, Q, K, V, None)
    dOut = torch.randn_like(state.out.contiguous())
    (res_dQ, res_dK, res_dV), (ref_dQ, ref_dK, ref_dV) = _reference(
        case, route, Q, K, V, dOut, state=state
    )
    # Low-precision accumulation in the Triton kernel rounds a small
    # fraction of elements past the 1e-4 default tolerance (larger at the
    # causal-mask boundary); 3e-2 matches the SDPA math-test convention and
    # stays inside the fp16/bf16 resolution.
    utils.gems_assert_close(res_dQ, ref_dQ, dtype, equal_nan=True, atol=3e-2)
    utils.gems_assert_close(res_dK, ref_dK, dtype, equal_nan=True, atol=3e-2)
    utils.gems_assert_close(res_dV, ref_dV, dtype, equal_nan=True, atol=3e-2)
