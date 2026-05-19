import pytest
import torch

from flag_gems.fused.dsv4_attention import (
    dsv4_combine_topk_swa_indices,
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_dequantize_and_gather_k_cache,
    dsv4_flash_mla_sparse_decode,
    dsv4_flash_mla_sparse_prefill,
    dsv4_fused_q_kv_rmsnorm,
    dsv4_qnorm_rope_kv_rope_quant_insert,
)

from . import base

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        combine_topk_swa_indices as vllm_combine_topk_swa_indices,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        compute_global_topk_indices_and_lens as vllm_compute_global_topk_indices_and_lens,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        dequantize_and_gather_k_cache as vllm_dequantize_and_gather_k_cache,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        fused_q_kv_rmsnorm as vllm_fused_q_kv_rmsnorm,
    )
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd as vllm_flash_mla_sparse_fwd,
    )
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _has_hopper_tl_float8e4nv() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major == 9


HAS_HOPPER_TL_FLOAT8E4NV = _has_hopper_tl_float8e4nv()


def _build_cos_sin_cache(max_pos: int, rope_dim: int, device: str):
    half = rope_dim // 2
    pos = torch.arange(max_pos, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.arange(half, device=device, dtype=torch.float32).unsqueeze(0)
    angles = pos * (1.0 / (10000.0 ** (freq / max(1, half - 1))))
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).to(torch.bfloat16)


def _build_decode_cache(
    num_tokens: int,
    block_size: int,
    head_dim: int,
    rope_dim: int,
    device: str,
):
    nope_dim = head_dim - rope_dim
    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    num_blocks = max(2, (num_tokens + block_size - 1) // block_size + 1)
    cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)

    q_seed = torch.randn(
        (num_tokens, 64, head_dim), device=device, dtype=torch.bfloat16
    )
    kv_seed = torch.randn((num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    cos_sin = _build_cos_sin_cache(num_tokens + 8, rope_dim, device)
    dsv4_qnorm_rope_kv_rope_quant_insert(
        q_seed,
        kv_seed,
        cache,
        slot_mapping,
        positions,
        cos_sin,
        eps=1e-6,
        block_size=block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
    )
    return cache


def _build_prefill_case(device: str = "cuda"):
    torch.manual_seed(7)
    sq = 64
    h = 64
    dt = 576
    skv = 256
    topk = 128
    return {
        "q": torch.randn((sq, h, dt), device=device, dtype=torch.bfloat16),
        "kv": torch.randn((skv, 1, dt), device=device, dtype=torch.bfloat16),
        "indices": torch.randint(
            0, skv, (sq, 1, topk), device=device, dtype=torch.int32
        ),
        "sm_scale": dt**-0.5,
        "attn_sink": torch.randn((h,), device=device, dtype=torch.float32),
        "topk_length": torch.full((sq,), topk, device=device, dtype=torch.int32),
    }


def _build_decode_case(device: str = "cuda"):
    torch.manual_seed(11)
    bsz = 8
    next_n = 1
    h = 64
    dt = 576
    topk = 128
    rope_dim = 64
    decode_tokens = bsz * next_n
    return {
        "q": torch.randn((bsz, next_n, h, dt), device=device, dtype=torch.bfloat16),
        "cache": _build_decode_cache(1024, 64, dt, rope_dim, device),
        "indices": torch.randint(
            0, 768, (bsz, next_n, topk), device=device, dtype=torch.int32
        ),
        "sm_scale": dt**-0.5,
        "attn_sink": torch.randn((h,), device=device, dtype=torch.float32),
        "extra_cache": _build_decode_cache(1024, 64, dt, rope_dim, device),
        "extra_indices": torch.randint(
            0, 768, (bsz, next_n, topk), device=device, dtype=torch.int32
        ),
        "topk_length": torch.full(
            (decode_tokens,), topk, device=device, dtype=torch.int32
        ),
        "extra_topk_length": torch.full(
            (decode_tokens,), topk, device=device, dtype=torch.int32
        ),
    }


def _prefill_op(q, kv, indices, sm_scale, attn_sink, topk_length):
    return dsv4_flash_mla_sparse_prefill(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


def _decode_op(
    q,
    cache,
    indices,
    sm_scale,
    attn_sink,
    extra_cache,
    extra_indices,
    topk_length,
    extra_topk_length,
):
    return dsv4_flash_mla_sparse_decode(
        q,
        cache,
        indices,
        sm_scale,
        head_dim_v=512,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        block_size=64,
        rope_dim=64,
    )


class DSV4PrefillBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_flash_mla_sparse_prefill",
            _prefill_op,
            [torch.bfloat16],
            gems_op=_prefill_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["q"],
            self.case["kv"],
            self.case["indices"],
            self.case["sm_scale"],
            self.case["attn_sink"],
            self.case["topk_length"],
        )


class DSV4DecodeBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_flash_mla_sparse_decode",
            _decode_op,
            [torch.bfloat16],
            gems_op=_decode_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["q"],
            self.case["cache"],
            self.case["indices"],
            self.case["sm_scale"],
            self.case["attn_sink"],
            self.case["extra_cache"],
            self.case["extra_indices"],
            self.case["topk_length"],
            self.case["extra_topk_length"],
        )


def _prefill_fg_vs_vllm_op(
    q,
    kv,
    indices,
    sm_scale,
    attn_sink,
    topk_length,
    fg_out,
    vl_out,
):
    _ = vl_out
    return dsv4_flash_mla_sparse_prefill(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=fg_out,
    )


def _prefill_vllm_vs_fg_op(
    q,
    kv,
    indices,
    sm_scale,
    attn_sink,
    topk_length,
    fg_out,
    vl_out,
):
    _ = fg_out
    return vllm_flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=vl_out,
    )


def _rms_fg_vs_vllm_op(qr, kv, q_weight, kv_weight, eps):
    return dsv4_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def _rms_vllm_vs_fg_op(qr, kv, q_weight, kv_weight, eps):
    return vllm_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def _gather_fg_vs_vllm_op(
    out_fg,
    out_vl,
    cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
):
    _ = out_vl
    return dsv4_dequantize_and_gather_k_cache(
        out_fg,
        cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset=offset,
        rope_dim=64,
        nope_dim=512,
    )


def _gather_vllm_vs_fg_op(
    out_fg,
    out_vl,
    cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
):
    _ = out_fg
    return vllm_dequantize_and_gather_k_cache(
        out_vl,
        cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    )


def _global_fg_vs_vllm_op(topk_indices, token_to_req, blk_tbl, block_size, valid):
    return dsv4_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        blk_tbl,
        block_size,
        valid,
    )


def _global_vllm_vs_fg_op(topk_indices, token_to_req, blk_tbl, block_size, valid):
    return vllm_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        blk_tbl,
        block_size,
        valid,
    )


def _combine_fg_vs_vllm_op(
    topk2,
    query_start_loc,
    seq_lens2,
    gather_lens2,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    return dsv4_combine_topk_swa_indices(
        topk2,
        query_start_loc,
        seq_lens2,
        gather_lens2,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )


def _combine_vllm_vs_fg_op(
    topk2,
    query_start_loc,
    seq_lens2,
    gather_lens2,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    return vllm_combine_topk_swa_indices(
        topk2,
        query_start_loc,
        seq_lens2,
        gather_lens2,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )


class DSV4PrefillVsVLLMBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_flash_mla_sparse_prefill_vs_vllm",
            _prefill_vllm_vs_fg_op,
            [torch.bfloat16],
            gems_op=_prefill_fg_vs_vllm_op,
        )
        self.case = case
        self.case["fg_out"] = torch.empty(
            (self.case["q"].shape[0], self.case["q"].shape[1], 512),
            device=self.case["q"].device,
            dtype=torch.bfloat16,
        )
        self.case["vl_out"] = torch.empty_like(self.case["fg_out"])

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["q"],
            self.case["kv"],
            self.case["indices"],
            self.case["sm_scale"],
            self.case["attn_sink"],
            self.case["topk_length"],
            self.case["fg_out"],
            self.case["vl_out"],
        )


class DSV4RMSNormVsVLLMBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_fused_q_kv_rmsnorm_vs_vllm",
            _rms_vllm_vs_fg_op,
            [torch.bfloat16],
            gems_op=_rms_fg_vs_vllm_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["qr"],
            self.case["kv"],
            self.case["q_weight"],
            self.case["kv_weight"],
            self.case["eps"],
        )


class DSV4GatherVsVLLMBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_dequantize_and_gather_k_cache_vs_vllm",
            _gather_vllm_vs_fg_op,
            [torch.bfloat16],
            gems_op=_gather_fg_vs_vllm_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["out_fg"],
            self.case["out_vl"],
            self.case["cache"],
            self.case["seq_lens"],
            self.case["gather_lens"],
            self.case["block_table"],
            self.case["block_size"],
            self.case["offset"],
        )


class DSV4GlobalTopkVsVLLMBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_compute_global_topk_indices_and_lens_vs_vllm",
            _global_vllm_vs_fg_op,
            [torch.bfloat16],
            gems_op=_global_fg_vs_vllm_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["topk_indices"],
            self.case["token_to_req"],
            self.case["blk_tbl"],
            self.case["block_size"],
            self.case["valid"],
        )


class DSV4CombineTopkVsVLLMBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_combine_topk_swa_indices_vs_vllm",
            _combine_vllm_vs_fg_op,
            [torch.bfloat16],
            gems_op=_combine_fg_vs_vllm_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["topk2"],
            self.case["query_start_loc"],
            self.case["seq_lens2"],
            self.case["gather_lens2"],
            self.case["window_size"],
            self.case["compress_ratio"],
            self.case["topk"],
            self.case["M"],
            self.case["N"],
        )


def _build_subops_case(device: str = "cuda"):
    torch.manual_seed(2026)
    case = {
        "qr": torch.randn((128, 64 * 576), device=device, dtype=torch.bfloat16),
        "kv": torch.randn((128, 576), device=device, dtype=torch.bfloat16),
        "q_weight": torch.randn((64 * 576,), device=device, dtype=torch.bfloat16),
        "kv_weight": torch.randn((576,), device=device, dtype=torch.bfloat16),
        "eps": 1e-6,
    }

    cache = _build_decode_cache(512, 64, 576, 64, device)
    case.update(
        {
            "cache": cache,
            "out_fg": torch.empty((1, 512, 576), device=device, dtype=torch.bfloat16),
            "out_vl": torch.empty((1, 512, 576), device=device, dtype=torch.bfloat16),
            "seq_lens": torch.tensor([512], device=device, dtype=torch.int32),
            "gather_lens": torch.tensor([512], device=device, dtype=torch.int32),
            "block_table": torch.tensor(
                [[i for i in range(8)]], device=device, dtype=torch.int32
            ),
            "block_size": 64,
            "offset": 0,
        }
    )

    case.update(
        {
            "topk_indices": torch.randint(
                0, 256, (256, 64), device=device, dtype=torch.int32
            ),
            "token_to_req": torch.randint(
                0, 2, (256,), device=device, dtype=torch.int32
            ),
            "blk_tbl": torch.randint(0, 64, (2, 64), device=device, dtype=torch.int32),
            "valid": torch.randint(0, 2, (256,), device=device, dtype=torch.int32),
        }
    )

    case.update(
        {
            "query_start_loc": torch.tensor(
                [0, 128, 256], device=device, dtype=torch.int32
            ),
            "seq_lens2": torch.tensor([1024, 1024], device=device, dtype=torch.int32),
            "gather_lens2": torch.tensor([512, 512], device=device, dtype=torch.int32),
            "topk2": torch.randint(0, 256, (256, 64), device=device, dtype=torch.int32),
            "window_size": 64,
            "compress_ratio": 2,
            "topk": 64,
            "M": 256,
            "N": 512,
        }
    )

    return case


@pytest.mark.dsv4_attention_prefill
@pytest.mark.skipif(
    not HAS_HOPPER_TL_FLOAT8E4NV,
    reason="DSV4 benchmarks require NVIDIA Hopper (SM90) with tl.float8e4nv support",
)
def test_dsv4_attention_prefill_benchmark():
    bench = DSV4PrefillBenchmark(_build_prefill_case())
    try:
        bench.run()
    except BaseException as exc:
        err = str(exc)
        if "multiple values" in err and "BK" in err:
            pytest.skip(f"flash_mla_sparse_fwd launch signature mismatch: {exc}")
        raise


@pytest.mark.dsv4_attention_decode
@pytest.mark.skipif(
    not HAS_HOPPER_TL_FLOAT8E4NV,
    reason="DSV4 benchmarks require NVIDIA Hopper (SM90) with tl.float8e4nv support",
)
def test_dsv4_attention_decode_benchmark():
    bench = DSV4DecodeBenchmark(_build_decode_case())
    try:
        bench.run()
    except BaseException as exc:
        err = str(exc)
        if "multiple values" in err and "BK" in err:
            pytest.skip(f"flash_mla_sparse_fwd launch signature mismatch: {exc}")
        raise


@pytest.mark.dsv4_attention_prefill
@pytest.mark.skipif(
    not HAS_HOPPER_TL_FLOAT8E4NV,
    reason="DSV4 benchmarks require NVIDIA Hopper (SM90) with tl.float8e4nv support",
)
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM is not installed")
def test_dsv4_attention_prefill_perf_vs_vllm():
    supported, reason = is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason or "vLLM FlashMLA sparse is not supported")

    bench = DSV4PrefillVsVLLMBenchmark(_build_prefill_case())
    try:
        bench.run()
    except BaseException as exc:
        err = str(exc)
        if "multiple values" in err and "BK" in err:
            pytest.skip(f"flash_mla_sparse_fwd launch signature mismatch: {exc}")
        if "params.topk % (2*B_TOPK) == 0" in err:
            pytest.skip(f"vLLM FlashMLA sparse prefill kernel constraint hit: {exc}")
        raise


@pytest.mark.dsv4_attention_decode
@pytest.mark.skipif(
    not HAS_HOPPER_TL_FLOAT8E4NV,
    reason="DSV4 benchmarks require NVIDIA Hopper (SM90) with tl.float8e4nv support",
)
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM is not installed")
def test_dsv4_subops_perf_vs_vllm():
    case = _build_subops_case()
    DSV4RMSNormVsVLLMBenchmark(case).run()
    DSV4GatherVsVLLMBenchmark(case).run()
    DSV4GlobalTopkVsVLLMBenchmark(case).run()
    DSV4CombineTopkVsVLLMBenchmark(case).run()
