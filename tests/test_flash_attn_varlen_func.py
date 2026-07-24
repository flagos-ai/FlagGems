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

from typing import List, Optional, Tuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

device = flag_gems.device
vendor_name = flag_gems.vendor_name
HOPPER_AVAILABLE = (
    vendor_name == "nvidia"
    and device == "cuda"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 9
)

FA_VERSION_CASES = [
    pytest.param(2, id="fa2"),
    pytest.param(
        3,
        id="fa3",
        marks=pytest.mark.skipif(
            not HOPPER_AVAILABLE,
            reason="FA3 requires an NVIDIA Hopper GPU",
        ),
    ),
]

if QUICK_MODE:
    NUM_HEADS = [(8, 2)]
    HEAD_SIZES = [128]
    FLOAT_DTYPES = [torch.float16]
    ALIBI = [False]
    SOFT_CAPS = [None]
    NUM_BLOCKS = [2048]
    OPTIMIZE_INIT = [False]
    SWAP_SOFT_CAPS = [None]
    NONCONTIG_DTYPES = [torch.float16]
    NONCONTIG_OPTIMIZE_INIT = [False]
else:
    NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
    HEAD_SIZES = [128, 192, 256]
    FLOAT_DTYPES = [torch.float16, torch.bfloat16]
    ALIBI = [False, True]
    SOFT_CAPS = [None, 10.0, 50.0]
    NUM_BLOCKS = [32768, 2048]
    OPTIMIZE_INIT = [False, True]
    SWAP_SOFT_CAPS = [None, 10.0]
    NONCONTIG_DTYPES = [torch.float16, torch.bfloat16]
    NONCONTIG_OPTIMIZE_INIT = [False, True]


def make_paged_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    non_contiguous: bool,
    device: str = device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (num_blocks, block_size, num_kv_heads, head_size)
    if not non_contiguous:
        key_cache = torch.randn(*shape, dtype=dtype, device=device)
        value_cache = torch.randn_like(key_cache)
        return key_cache, value_cache

    storage_shape = (num_blocks * 2, block_size, num_kv_heads, head_size)
    key_storage = torch.randn(*storage_shape, dtype=dtype, device=device)
    value_storage = torch.randn_like(key_storage)
    key_cache = key_storage[::2][:num_blocks]
    value_cache = value_storage[::2][:num_blocks]

    assert key_cache.shape == shape
    assert value_cache.shape == shape
    assert key_cache.stride() == value_cache.stride()
    assert key_cache.stride(-1) == 1
    assert key_cache.stride(0) != block_size * key_cache.stride(1)
    return key_cache, value_cache


# Following varlen and paged attn tests are copied from
# https://github.com/vllm-project/flash-attention/blob/main/tests/test_vllm_flash_attn.py
def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)

    if causal:
        v = torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32)
        return v * slopes

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)

    return -slopes * relative_pos.to(dtype=slopes.dtype)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor = None,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    s_aux: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    softmax_lses: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        if s_aux is None:
            q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        if s_aux is None:
            attn = torch.einsum("qhd,khd->hqk", q, k)
        else:
            attn = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))

        if attn_bias is not None:
            attn = attn + attn_bias[i, :, :query_len, :kv_len]

        if s_aux is not None:
            # The attention sink is a final, unscaled logit whose value vector is
            # zero. Drop its probability before the PV product.
            sink_logits = s_aux.float()[:, None, None].expand(-1, query_len, 1)
            attn = torch.cat((attn.float(), sink_logits), dim=-1)
            if return_softmax_lse:
                softmax_lses.append(torch.logsumexp(attn, dim=-1))
            attn = torch.softmax(attn, dim=-1)[..., :-1]
            out = torch.einsum("hqk,khd->qhd", attn, v.float()).to(v.dtype)
        else:
            if return_softmax_lse:
                softmax_lses.append(torch.logsumexp(attn.float(), dim=-1))
            attn = torch.softmax(attn, dim=-1).to(v.dtype)
            out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    output = torch.cat(outputs, dim=0)
    if return_softmax_lse:
        return output, torch.cat(softmax_lses, dim=1)
    return output


@pytest.mark.flash_attn_varlen_func
@pytest.mark.flash_attn_varlen_opt_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("alibi", ALIBI)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("optimize_init", OPTIMIZE_INIT)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@torch.inference_mode()
def test_flash_attn_varlen_func(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    alibi: bool,
    soft_cap: Optional[float],
    num_blocks: int,
    optimize_init: bool,
) -> None:
    # (Issue) numerical stability concern
    if alibi is True and soft_cap is not None:
        return

    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)

        if vendor_name == "cambricon":
            torch.manual_seed(123456)
            torch.mlu.manual_seed_all(123456)

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(
            sum(query_lens), num_query_heads, head_size, dtype=dtype, device=device
        )
        key_cache, value_cache = make_paged_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device=device,
            non_contiguous=False,
        )
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        causal = True

        if alibi:
            # alibi_slopes = torch.rand(num_seqs, num_query_heads, device=device, dtype=torch.float32) * 0.3
            alibi_slopes = (
                torch.ones(
                    num_seqs, num_query_heads, device=device, dtype=torch.float32
                )
                * 0.3
            )
            attn_bias = attn_bias_from_alibi_slopes(
                alibi_slopes, max_query_len, max_kv_len, causal=causal
            )
        else:
            alibi_slopes, attn_bias = None, None

        if vendor_name in ["cambricon", "sunrise"]:
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=causal,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                alibi_slopes=alibi_slopes,
                fa_version=2,
            )
        else:
            if optimize_init:
                output = flag_gems.flash_attn_varlen_opt_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )
            else:
                output = flag_gems.flash_attn_varlen_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            attn_bias=attn_bias,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        msg = f"{torch.max(torch.abs(output - ref_output))}"
        if vendor_name == "sunrise":
            torch.testing.assert_close(
                output, ref_output, atol=3e-2, rtol=1e-2, msg=msg
            )
        else:
            torch.testing.assert_close(
                output, ref_output, atol=2e-2, rtol=1e-2, msg=msg
            )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.parametrize("dtype", NONCONTIG_DTYPES)
@pytest.mark.parametrize("optimize_init", NONCONTIG_OPTIMIZE_INIT)
@torch.inference_mode()
def test_flash_attn_varlen_func_noncontiguous_kv_cache(
    monkeypatch,
    dtype: torch.dtype,
    optimize_init: bool,
) -> None:
    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)

        seq_lens = [(1, 1328), (5, 18), (129, 463)]
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_seqs = len(seq_lens)
        num_query_heads = 8
        num_kv_heads = 2
        head_size = 128
        block_size = 32
        num_blocks = 2048
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (-1, -1)
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache, value_cache = make_paged_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            non_contiguous=True,
            device=device,
        )
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        op = (
            flag_gems.flash_attn_varlen_opt_func
            if optimize_init
            else flag_gems.flash_attn_varlen_func
        )
        output = op(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=0,
            fa_version=2,
        )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
        )

        max_diff = torch.max(torch.abs(output - ref_output))
        msg = f"max_diff={max_diff}, k_stride={key_cache.stride()}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not working")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (1, 18), (1, 463)]])
@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("soft_cap", SWAP_SOFT_CAPS)
@pytest.mark.parametrize("num_blocks", [2048])
@pytest.mark.parametrize("fa_version", FA_VERSION_CASES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@torch.inference_mode()
def test_flash_attn_varlen_func_swap_qg(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    fa_version: int,
) -> None:
    with torch.device(flag_gems.device):
        utils.init_seed(1234567890)
        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(
            sum(query_lens), num_query_heads, head_size, dtype=dtype, device=device
        )
        key_cache, value_cache = make_paged_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device=device,
            non_contiguous=False,
        )
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        if vendor_name in ["cambricon", "sunrise"]:
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=fa_version,
            )
        else:
            output = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=fa_version,
            )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        torch.testing.assert_close(
            output, ref_output, atol=2e-2, rtol=1e-2
        ), f"{torch.max(torch.abs(output - ref_output))}"


def _make_fa3_s_aux_case(query_len: int, kv_len: int):
    num_query_heads = 8
    num_kv_heads = 2
    head_size = 128
    block_size = 16
    num_kv_blocks = (kv_len + block_size - 1) // block_size
    dtype = torch.bfloat16
    scale = head_size**-0.5

    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    key_cache, value_cache = make_paged_kv_cache(
        num_kv_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        non_contiguous=False,
        device=device,
    )
    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=device)
    block_tables = torch.arange(
        num_kv_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)
    s_aux = torch.tensor(
        [float("-inf"), -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        dtype=torch.bfloat16,
        device=device,
    )
    return (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    )


def _capture_fa3_plans(monkeypatch):
    """Observe scheduler results in tests without production instrumentation."""

    from importlib import import_module

    hopper_package = flag_gems.flash_attn_varlen_func.__module__.split(".ops.", 1)[0]
    FA3Scheduler = import_module(
        f"{hopper_package}.ops.attention_impl.scheduling"
    ).FA3Scheduler
    original_build = FA3Scheduler.build
    plans = []

    def build_and_capture(cls, inputs, config=None):
        plan = original_build(inputs, config)
        plans.append(plan)
        return plan

    monkeypatch.setattr(FA3Scheduler, "build", classmethod(build_and_capture))
    return FA3Scheduler, plans


def _plan_signature(plan) -> Tuple[str, int]:
    num_splits = plan.persistent_num_splits if plan.persistent_split_kv else 1
    return plan.kernel_name, num_splits


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@pytest.mark.parametrize(
    "route,kv_len,soft_cap,num_splits,expected_plans",
    [
        pytest.param(
            "direct",
            256,
            0.0,
            (1,),
            [("direct_packed_gqa", 1)],
            id="direct",
        ),
        pytest.param(
            "long",
            512,
            10.0,
            (1,),
            [("long_paged_prefill", 1)],
            id="persistent-long-softcap",
        ),
        pytest.param(
            "long",
            2048,
            0.0,
            (1, 2),
            [("long_paged_prefill", 1), ("persistent_splitkv_s2", 2)],
            id="split-kv",
        ),
    ],
)
@torch.inference_mode()
def test_flash_attn_varlen_fa3_s_aux(
    monkeypatch,
    route: str,
    kv_len: int,
    soft_cap: float,
    num_splits: Tuple[int, ...],
    expected_plans: List[Tuple[str, int]],
) -> None:
    FA3Scheduler, plans = _capture_fa3_plans(monkeypatch)

    utils.init_seed(1234567890)
    query_len = 2
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    ) = _make_fa3_s_aux_case(query_len, kv_len)

    ref_output, ref_lse = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap if soft_cap > 0 else None,
        s_aux=s_aux,
        return_softmax_lse=True,
    )

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", route)
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "on")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    FA3Scheduler.clear_config_cache()

    results = {}
    try:
        for split_count in num_splits:
            output, lse = flag_gems.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=query_len,
                max_seqlen_k=kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=(-1, -1),
                block_table=block_tables,
                softcap=soft_cap,
                s_aux=s_aux,
                return_softmax_lse=True,
                num_splits=split_count,
                fa_version=3,
            )
            results[split_count] = (output, lse)
    finally:
        FA3Scheduler.clear_config_cache()

    assert [_plan_signature(plan) for plan in plans] == expected_plans
    for output, lse in results.values():
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=1e-2)

    if len(num_splits) > 1:
        one_pass_output, one_pass_lse = results[1]
        split_output, split_lse = results[2]
        torch.testing.assert_close(split_output, one_pass_output, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(split_lse, one_pass_lse, atol=2e-2, rtol=1e-2)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@pytest.mark.parametrize("invalid_contract", ["dtype", "shape", "noncontiguous"])
@torch.inference_mode()
def test_flash_attn_varlen_fa3_s_aux_contract(invalid_contract: str) -> None:
    utils.init_seed(1234567890)
    query_len = 2
    kv_len = 64
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    ) = _make_fa3_s_aux_case(query_len, kv_len)

    if invalid_contract == "dtype":
        s_aux = s_aux.float()
    elif invalid_contract == "shape":
        s_aux = s_aux[:-1]
    else:
        s_aux = torch.empty(16, dtype=torch.bfloat16, device=device)[::2]
        assert not s_aux.is_contiguous()

    with pytest.raises(RuntimeError, match="s_aux must be a contiguous bf16"):
        flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            s_aux=s_aux,
            num_splits=1,
            fa_version=3,
        )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="FA3 requires an NVIDIA Hopper GPU")
@pytest.mark.parametrize(
    "route,expected_kernel",
    [
        pytest.param("direct", "direct_packed_gqa", id="direct"),
        pytest.param("long", "long_paged_prefill", id="persistent-long"),
    ],
)
@torch.inference_mode()
def test_flash_attn_varlen_fa3_s_aux_empty_kv(
    monkeypatch, route: str, expected_kernel: str
) -> None:
    FA3Scheduler, plans = _capture_fa3_plans(monkeypatch)

    utils.init_seed(1234567890)
    query_len = 2
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        seqused_k,
        block_tables,
        s_aux,
        scale,
    ) = _make_fa3_s_aux_case(query_len, kv_len=1)
    seqused_k.zero_()

    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DECODE_STRATEGY", route)
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK", "on")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER", "off")
    monkeypatch.setenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", "1")
    FA3Scheduler.clear_config_cache()

    try:
        output, lse = flag_gems.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=query_len,
            max_seqlen_k=1,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            s_aux=s_aux,
            return_softmax_lse=True,
            num_splits=1,
            fa_version=3,
        )
    finally:
        FA3Scheduler.clear_config_cache()

    assert [_plan_signature(plan) for plan in plans] == [(expected_kernel, 1)]
    assert torch.count_nonzero(output).item() == 0
    expected_lse = s_aux.float().unsqueeze(1).expand(-1, query_len)
    torch.testing.assert_close(lse, expected_lse, atol=1e-4, rtol=0)
