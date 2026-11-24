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
from importlib.metadata import version

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import base

_SAFE_HEAD_SIZES = (64, 128)
_NATIVE_SDPA_BHSD = flag_gems.vendor_name in ("hygon", "thead")


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
    scale = (
        softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Q.shape[-1]))
    )

    kwargs = dict(scale=scale)
    if window_size_left >= 0:
        kwargs["window_size_left"] = window_size_left
    if window_size_right >= 0:
        kwargs["window_size_right"] = window_size_right

    output, lse, rng_state, unused, _ = torch.ops.aten._flash_attention_forward(
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
    return output, lse, rng_state, unused


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
    philox_seed = rng_state[0]
    philox_offset = rng_state[1]

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

    lse = lse_4d.float()
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


def _missing_attention_baseline(exc):
    detail = str(exc)
    return any(
        message in detail
        for message in (
            "USE_FLASH_ATTENTION was not enabled",
            "USE_MEM_EFF_ATTENTION was not enabled",
            "not compiled with cuDNN Flash Attention enabled",
        )
    ) or (
        isinstance(exc, NotImplementedError)
        and detail.startswith(
            (
                "Could not run 'aten::_flash_attention_",
                "Could not run 'aten::_efficient_attention_",
                "Could not run 'aten::_scaled_dot_product_flash_attention",
                "Could not run 'aten::_scaled_dot_product_cudnn_attention",
                "Could not run 'aten::_scaled_dot_product_efficient_attention",
            )
        )
    )


class _AttentionBackwardBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype):
        inputs = super().get_input_iter(dtype)
        ppu_cudnn = (
            flag_gems.vendor_name == "thead"
            and self.op_name == "scaled_dot_product_cudnn_attention_backward"
        )
        if flag_gems.vendor_name not in ("metax", "hygon") and not ppu_cudnn:
            yield from inputs
            return

        # A native forward is required to construct the saved backward state.
        # Probe the matching backward separately, outside the timed section.
        while True:
            try:
                sample = next(inputs)
            except StopIteration:
                return
            except RuntimeError as exc:
                unsupported_descriptor = ppu_cudnn and (
                    "SetAttribute CUDNN_ATTR_TENSOR_IS_BY_VALUE Failed cudnn_status: "
                    "ACDNN_STATUS_BAD_PARAM" in str(exc)
                )
                if not (unsupported_descriptor or _missing_attention_baseline(exc)):
                    raise
                pytest.skip(
                    f"Native baseline: N/A (saved forward state unavailable: {exc})"
                )

            if not base.Config.skip_native:
                args, kwargs = self.unpack_to_args_kwargs(sample)
                try:
                    self.torch_op(*args, **kwargs)
                except RuntimeError as exc:
                    if not _missing_attention_baseline(exc):
                        raise
                    with pytest.MonkeyPatch.context() as local:
                        local.setattr(base.Config, "skip_native", True)
                        local.setattr(
                            base.Config, "native_baseline_skip_reason", str(exc)
                        )
                        yield sample
                    continue
            yield sample


class FlashAttentionBackwardBenchmark(_AttentionBackwardBenchmark):
    aggregate_results_by_dtype = True

    def __init__(self, *args, causal, head_size, **kwargs):
        self.causal = causal
        self.head_size = head_size
        super().__init__(*args, **kwargs)

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
        self.shapes = [
            shape
            for shape in self.shapes
            if shape[-2:] == (self.head_size, self.causal)
        ]

    def set_more_shapes(self):
        return None


def flash_attention_backward_input_fn(config, dtype, device):
    batch, num_heads, q_seq_len, kv_seq_len, head_size, is_causal = config
    scale = 1.0 / math.sqrt(head_size)

    dev_idx = _device_index(device)
    set_philox_state(1234567890, 0, dev_idx)

    Q = torch.empty(
        batch, q_seq_len, num_heads, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    K = torch.empty(
        batch, kv_seq_len, num_heads, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    V = torch.empty(
        batch, kv_seq_len, num_heads, head_size, dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    dOut = torch.randn(
        batch, q_seq_len, num_heads, head_size, dtype=dtype, device=device
    )
    if flag_gems.vendor_name == "mthreads":
        Q, K, V, dOut = (t.transpose(1, 2).contiguous() for t in (Q, K, V, dOut))

    out, lse, rng_state, unused = flash_attn_forward_native(
        Q,
        K,
        V,
        is_causal=is_causal,
        softmax_scale=scale,
    )
    yield (
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
        {"scale": scale},
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
    unused,
    scale=None,
):
    return flag_gems.flash_attention_backward(
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
        unused=unused,
        scale=scale,
    )


@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="CANN 8.5/9.1.1 torch_npu has no native ATen flash backward baseline",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
@pytest.mark.flash_attention_backward
@pytest.mark.parametrize("causal", (False, True), ids=("noncausal", "causal"))
@pytest.mark.parametrize("head_size", _SAFE_HEAD_SIZES, ids=lambda dim: f"d{dim}")
def test_perf_flash_attention_backward(causal, head_size):
    bench = FlashAttentionBackwardBenchmark(
        op_name="flash_attention_backward",
        input_fn=flash_attention_backward_input_fn,
        torch_op=torch.ops.aten._flash_attention_backward.default,
        dtypes=[torch.float16, torch.bfloat16],
        causal=causal,
        head_size=head_size,
    )
    bench.set_gems(_flash_attn_bwd_gems)
    bench.run()


_FLASH_BWD_NATIVE_FAMILIES = {
    "dense-p0": [
        ("dense", (128,), (128,), 4, 4, 64, 0.0, False, None, None),
        ("dense", (256,), (256,), 8, 2, 80, 0.0, True, None, None),
    ],
    "dense-dropout": [
        ("dense", (128,), (128,), 4, 4, 64, 0.25, False, None, None),
        ("dense", (128,), (256,), 8, 2, 80, 0.25, True, None, None),
    ],
    "varlen-p0": [
        ("varlen", (64, 128), (128, 64), 4, 2, 64, 0.0, False, None, None),
        ("varlen", (65, 127), (127, 65), 8, 2, 80, 0.0, False, 2, 5),
    ],
    "varlen-dropout": [
        ("varlen", (64, 128), (128, 64), 4, 2, 64, 0.25, False, None, None),
        ("varlen", (65, 127), (127, 65), 8, 2, 80, 0.25, False, 2, 5),
    ],
}


class FlashAttentionBackwardNativeFamilyBenchmark(_AttentionBackwardBenchmark):
    aggregate_results_by_dtype = True

    def __init__(self, *args, family, case_index, **kwargs):
        self.family = family
        self.case_index = case_index
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [_FLASH_BWD_NATIVE_FAMILIES[self.family][self.case_index]]

    def set_more_shapes(self):
        return None


def flash_attention_backward_native_family_input_fn(config, dtype, device):
    layout, q_lengths, k_lengths, hq, hkv, dim, dropout_p, causal, left, right = config
    packed = layout == "varlen"
    q_shape = (sum(q_lengths), hq, dim) if packed else (1, q_lengths[0], hq, dim)
    kv_shape = (sum(k_lengths), hkv, dim) if packed else (1, k_lengths[0], hkv, dim)
    Q = torch.randn(q_shape, dtype=dtype, device=device) * 0.1
    K = torch.randn(kv_shape, dtype=dtype, device=device) * 0.1
    V = torch.randn(kv_shape, dtype=dtype, device=device) * 0.1
    dOut = torch.randn(q_shape, dtype=dtype, device=device)
    if not packed and flag_gems.vendor_name == "mthreads":
        Q, K, V, dOut = (t.transpose(1, 2).contiguous() for t in (Q, K, V, dOut))
    cu_q = cu_k = None
    if packed:
        cu_q = torch.tensor(
            (0, *torch.tensor(q_lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        cu_k = torch.tensor(
            (0, *torch.tensor(k_lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
    kwargs = {"scale": dim**-0.5}
    if left is not None:
        kwargs["window_size_left"] = left
    if right is not None:
        kwargs["window_size_right"] = right
    out, lse, rng_state, unused, _ = torch.ops.aten._flash_attention_forward.default(
        Q,
        K,
        V,
        cu_q,
        cu_k,
        max(q_lengths),
        max(k_lengths),
        dropout_p,
        causal,
        False,
        **kwargs,
    )
    yield (
        dOut,
        Q,
        K,
        V,
        out,
        lse,
        cu_q,
        cu_k,
        max(q_lengths),
        max(k_lengths),
        dropout_p,
        causal,
        rng_state,
        unused,
        kwargs,
    )


def _flash_family_param(family, index, case):
    marks = ()
    if flag_gems.vendor_name == "mthreads" and family == "dense-dropout":
        from torch_musa.core import mudnn

        q_heads, kv_heads = case[3:5]
        if (
            q_heads != kv_heads
            and mudnn.version() == 3400
            and "S5000" in torch.musa.get_device_name()
        ):
            marks = pytest.mark.skip_native(
                vendors="mthreads",
                reason="MuDNN 3400 dense GQA dropout backward returns incorrect gradients on S5000",
            )
    return pytest.param(family, index, id=f"{family}-case{index}", marks=marks)


@pytest.mark.flash_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="CANN 8.5/9.1.1 torch_npu has no native ATen flash backward baseline",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
@pytest.mark.parametrize(
    "family,case_index",
    [
        _flash_family_param(family, index, case)
        for family, cases in _FLASH_BWD_NATIVE_FAMILIES.items()
        for index, case in enumerate(cases)
    ],
)
def test_perf_flash_attention_backward_native_families(family, case_index):
    if (
        flag_gems.vendor_name == "hygon"
        and "dtk2604" in version("torch")
        and family.startswith("varlen")
    ):
        pytest.skip("DTK 26.04 native varlen forward SIGSEGV in isolated probe")
    if flag_gems.vendor_name == "mthreads" and family.startswith("varlen"):
        if family == "varlen-dropout":
            pytest.skip("torch_musa 2.9.1 MuDNN native varlen dropout fails")
        if case_index == 1:
            pytest.skip("torch_musa 2.9.1 native FlashAttention rejects local windows")
    if (
        flag_gems.vendor_name == "iluvatar"
        and torch.__version__ == "2.7.1"
        and family == "varlen-p0"
        and case_index == 1
    ):
        pytest.skip("CoreX 4.4 native unpadded FlashAttention rejects local windows")
    bench = FlashAttentionBackwardNativeFamilyBenchmark(
        op_name="flash_attention_backward",
        input_fn=flash_attention_backward_native_family_input_fn,
        torch_op=torch.ops.aten._flash_attention_backward.default,
        dtypes=[torch.float16, torch.bfloat16],
        family=family,
        case_index=case_index,
    )
    bench.set_gems(flag_gems.flash_attention_backward)
    bench.run()


class SdpFlashAttentionBackwardBenchmark(_AttentionBackwardBenchmark):
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

    if _NATIVE_SDPA_BHSD:
        # Keep the matching SDPA producer's output layout and RNG ABI.
        saved = torch.ops.aten._scaled_dot_product_flash_attention(
            Q, K, V, 0.0, is_causal, False, scale=scale
        )
        out, lse, cum_seq_q, cum_seq_k = saved[:4]
        philox_seed, philox_offset = saved[6:8]
    else:
        cum_seq_q = cum_seq_k = None
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
    out_bshd = out.transpose(1, 2)
    if not _NATIVE_SDPA_BHSD:
        out_bshd = out_bshd.contiguous()
    dOut_bshd = dOut.transpose(1, 2).contiguous()

    yield (
        dOut_bshd,
        Q_bshd,
        K_bshd,
        V_bshd,
        out_bshd,
        lse,
        cum_seq_q,
        cum_seq_k,
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
    inputs = (dOut_bshd, Q_bshd, K_bshd, V_bshd, out_bshd)
    if _NATIVE_SDPA_BHSD:
        inputs = tuple(tensor.transpose(1, 2) for tensor in inputs)
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        *inputs,
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
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
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


class CudnnAttentionBackwardBenchmark(_AttentionBackwardBenchmark):
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

    yield (
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        out_bhsd,
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
        scale=scale,
    )


def _cudnn_attn_bwd_gems(
    dOut_bhsd,
    Q_bhsd,
    K_bhsd,
    V_bhsd,
    out_bhsd,
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
    # FlagGems Triton kernel expects 3-D lse (B, H, S); the forward now
    # returns the raw 4-D tensor so that the aten reference (cuDNN) is happy.
    lse_3d = lse.squeeze(-1) if lse.dim() == 4 else lse
    return flag_gems.ops.scaled_dot_product_cudnn_attention_backward(
        dOut_bhsd,
        Q_bhsd,
        K_bhsd,
        V_bhsd,
        out_bhsd,
        lse_3d,
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
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


class EfficientAttentionBackwardBenchmark(_AttentionBackwardBenchmark):
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


class SdpEfficientAttentionBackwardBenchmark(_AttentionBackwardBenchmark):
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
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
