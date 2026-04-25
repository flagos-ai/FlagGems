import pytest
import torch

import flag_gems

from .performance_utils import GenericBenchmark


class GetSchedulerMetadataBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 8, 1024, 16, 4, 128, 128),
            (32, 32, 512, 8, 8, 64, 64),
            (256, 256, 2048, 32, 32, 128, 128),
            (512, 512, 4096, 32, 8, 128, 128),
            (1024, 1024, 8192, 64, 16, 128, 128),
        ]

    def set_more_shapes(self):
        return None


@pytest.mark.get_scheduler_metadata
def test_get_scheduler_metadata(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            get_scheduler_metadata as vllm_get_scheduler_metadata,
        )
    except ImportError:
        pytest.skip("vLLM is not available, skipping performance test")

    def input_kwargs(shape, dtype, device):
        (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            headdim_v,
        ) = shape
        cache_seqlens = torch.randint(
            1, max_seqlen_k + 1, (batch_size,), dtype=torch.int32, device=device
        )

        yield (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            cache_seqlens,
            dtype,  # qkv_dtype
            headdim_v,  # headdim_v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k_new
            None,  # cache_leftpad
            None,  # page_size
            0,  # max_seqlen_k_new
            False,  # causal
            (-1, -1),  # window_size
            False,  # has_softcap
            0,  # num_splits
            None,  # pack_gqa
            0,  # sm_margin
        )

    def flaggems_wrapper(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q=None,
        cu_seqlens_k_new=None,
        cache_leftpad=None,
        page_size=None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),
        has_softcap=False,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
    ):
        return flag_gems.ops.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads=num_heads_q,
            num_heads_k=num_heads_kv,
            headdim=headdim,
            headdim_v=headdim_v or headdim,
            qkv_dtype=qkv_dtype,
            seqused_k=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            cu_seqlens_k_new=cu_seqlens_k_new,
            seqused_q=None,
            leftpad_k=cache_leftpad,
            page_size=page_size,
            max_seqlen_k_new=max_seqlen_k_new,
            is_causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            has_softcap=has_softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )

    bench = GetSchedulerMetadataBenchmark(
        op_name="get_scheduler_metadata",
        input_fn=input_kwargs,
        torch_op=vllm_get_scheduler_metadata,
        gems_op=flaggems_wrapper,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
