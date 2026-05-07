import pytest
import torch

import flag_gems

try:
    import vllm.vllm_flash_attn.flash_attn_interface  # noqa: F401

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from flag_gems.runtime import torch_device_fn

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.get_scheduler_metadata
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.skipif(cfg.TO_CPU, reason="Skipping correctness test in CPU mode.")
@pytest.mark.parametrize("batch_size", [1, 8, 256, 512])
@pytest.mark.parametrize("max_seqlen_k", [512, 2048])
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("num_splits_static", [0, 4])
@pytest.mark.parametrize("seed", [42])
def test_scheduler_metadata_correctness(
    batch_size, max_seqlen_k, headdim, num_splits_static, seed
):
    device = torch_device_fn.current_device()
    utils.init_seed(seed)

    seqused_k = torch.randint(
        1, max_seqlen_k, (batch_size,), dtype=torch.int32, device=device
    )
    num_heads, num_heads_k = 32, 8
    headdim_v = headdim
    qkv_dtype = torch.float16

    # num_sm = torch.cuda.get_device_properties(device).multi_processor_count

    from vllm.vllm_flash_attn import flash_attn_interface as vllm_ops  # noqa: F401

    ref_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=qkv_dtype,
        seqused_k=seqused_k,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        leftpad_k=None,
        page_size=None,
        max_seqlen_k_new=0,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        has_softcap=False,
        num_splits=num_splits_static,
        pack_gqa=True,
        sm_margin=0,
    )

    with flag_gems.use_gems():
        gems_metadata = flag_gems.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            headdim=headdim,
            headdim_v=headdim_v,
            qkv_dtype=qkv_dtype,
            seqused_k=seqused_k,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            seqused_q=None,
            leftpad_k=None,
            page_size=None,
            max_seqlen_k_new=0,
            is_causal=False,
            window_size_left=-1,
            window_size_right=-1,
            has_softcap=False,
            num_splits=num_splits_static,
            pack_gqa=True,
            sm_margin=0,
        )

    utils.gems_assert_close(gems_metadata, ref_metadata, dtype=torch.int32)
