from typing import Any, List, Optional

import pytest
import torch

import flag_gems

from tests.hopper_fa3_utils import (
    HAS_VLLM_FA,
    VLLM_FA_HAS_BLOCK_TABLE,
    VLLM_FA_HAS_SEQUSED_K,
    Shape as HopperFA3Shape,
    Tensors as HopperFA3Tensors,
    attn_flops as hopper_fa3_attn_flops,
    benchmark_shapes as hopper_fa3_benchmark_shapes,
    dispatch_source as hopper_fa3_dispatch_source,
    dispatches_to_hopper as dispatches_to_hopper_fa3,
    is_fa3_supported as is_hopper_fa3_supported,
    make_varlen as make_hopper_fa3_varlen,
    run_flag_gems as run_hopper_fa3,
    run_vllm_fa as run_vllm_hopper_fa3,
)

from . import base, consts, utils

vendor_name = flag_gems.vendor_name


def _selected_fa_version(pytestconfig) -> int:
    return pytestconfig.getoption("flash_attn_varlen_fa_version")


def _vllm_benchmark_enabled(pytestconfig) -> bool:
    return bool(pytestconfig.getoption("flash_attn_varlen_enable_vllm"))


def _selected_case(pytestconfig) -> str:
    return pytestconfig.getoption("flash_attn_varlen_case")


def _skip_unless_case_enabled(pytestconfig, case_name: str) -> None:
    selected_case = _selected_case(pytestconfig)
    if selected_case not in ("all", case_name):
        pytest.skip(f"{case_name} is disabled by --flash-attn-varlen-case.")


def _skip_unless_vllm_baseline_available() -> None:
    if utils.SkipVersion("vllm", "<0.9"):
        pytest.skip(
            "vLLM version prior to 0.9 does not include the "
            "flash_attn_varlen_func API."
        )
    if utils.SkipVersion("torch", "<2.7"):
        pytest.skip("Torch version prior to 2.7 is not compatible with VLLM.")


def _skip_unless_selected_fa_supported(pytestconfig) -> None:
    fa_version = _selected_fa_version(pytestconfig)
    if fa_version == 3 and not is_hopper_fa3_supported():
        pytest.skip("FA3 requires CUDA Hopper with Triton FA3 support.")


def _assert_flash_attn_varlen_uses_hopper_backend() -> None:
    if not dispatches_to_hopper_fa3():
        pytest.fail(
            "flag_gems.flash_attn_varlen_func is not routed to the Hopper "
            f"backend; source={hopper_fa3_dispatch_source()}"
        )


def _hopper_benchmark_shape_skip_reason(shape: HopperFA3Shape) -> Optional[str]:
    if shape.paged and not (VLLM_FA_HAS_BLOCK_TABLE and VLLM_FA_HAS_SEQUSED_K):
        return "vLLM flash-attention lacks paged KV benchmark support"
    return None


class FlashAttnVarlenBenchmark(base.Benchmark):
    """
    benchmark for flash_attn_varlen_func
    """

    fa_version = 2

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Collecting from qwen/Qwen3-1.7B
        # --random-input 512 --random-output 2048 --num-prompts 200 --request-rate inf
        # Format: (cu_seq_lens_q, seqused_k, num_heads, head_size, block_size,
        # num_blocks, alibi, soft_cap)

        all_cu_seq_lens_q = [
            (
                0,
                512,
            ),
            (
                0,
                1,
                2,
                72,
            ),
            tuple(range(0, 45))
            + (
                105,
                121,
                137,
                153,
                169,
                185,
                201,
                217,
                233,
                249,
                265,
            ),
            tuple(range(0, 196))
            + (
                211,
                226,
                240,
                253,
                265,
            ),
        ]
        all_seqused_k = [
            (512,),
            (
                1,
                1,
                70,
            ),
            (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
            (2333,)
            + (2331,) * 20
            + (2330,) * 20
            + (2329,) * 14
            + (2328,) * 18
            + (2327,) * 15
            + (2326,) * 17
            + (2325,) * 18
            + (2324,) * 21
            + (2323,) * 22
            + (2322,) * 24
            + (2321,) * 5
            + (
                2320,
                2319,
                2318,
                2317,
                2316,
            ),
        ]

        num_heads = 16
        num_heads_k = 8
        head_dim = 128
        block_size = 16
        num_blocks = 2000
        alibi = False
        soft_cap = None

        all_configs = [
            (
                cu_seq_lens_q,
                seqused_k,
                num_heads,
                num_heads_k,
                head_dim,
                block_size,
                num_blocks,
                alibi,
                soft_cap,
            )
            for cu_seq_lens_q, seqused_k in zip(all_cu_seq_lens_q, all_seqused_k)
        ]

        self.shapes = all_configs

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.flash_attn_varlen_input_fn(config, dtype, self.device)

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        """Input function for flash attention varlen benchmark"""
        (
            cu_query_lens,
            seqused_k,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            num_blocks,
            alibi,
            soft_cap,
        ) = config

        if alibi is True and soft_cap is not None:
            return

        num_seqs = len(cu_query_lens) - 1
        max_query_len = max(
            map(lambda x, y: x - y, cu_query_lens[1:], cu_query_lens[:-1])
        )
        max_kv_len = max(seqused_k)
        window_size = (-1, -1)
        scale = head_size**-0.5

        assert num_seqs == len(seqused_k)

        with torch.device(device):
            query = torch.randn(
                cu_query_lens[-1],
                num_query_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            out = torch.empty_like(query)
            key_cache = torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            value_cache = torch.randn_like(key_cache)
            cu_query_lens = torch.tensor(
                cu_query_lens, dtype=torch.int32, device=device
            )
            seqused_k = torch.tensor(seqused_k, dtype=torch.int32, device=device)

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
                alibi_slopes = (
                    torch.ones(
                        num_seqs, num_query_heads, device=device, dtype=torch.float32
                    )
                    * 0.3
                )
            else:
                alibi_slopes = None

        return (
            query,
            key_cache,
            value_cache,
            max_query_len,
            cu_query_lens,
            max_kv_len,
            None,
            seqused_k,
            None,
            0.0,
            scale,
            causal,
            window_size,
            soft_cap if soft_cap is not None else 0,
            alibi_slopes,
            False,
            False,
            block_tables,
            False,
            out,
            None,
            None,
            None,
            None,
            {
                "s_aux": None,
                "num_splits": 0,
                "cp_world_size": 1,
                "cp_rank": 0,
                "cp_tot_seqused_k": None,
                "fa_version": self.fa_version,
            },
        )


def _hopper_fa3_gems_wrapper(
    tensors: HopperFA3Tensors, shape: HopperFA3Shape, fa_version: int
):
    if fa_version == 2:
        kwargs = {
            "max_seqlen_q": tensors.max_seqlen_q,
            "cu_seqlens_q": tensors.cu_seqlens_q,
            "max_seqlen_k": tensors.max_seqlen_k,
            "softmax_scale": shape.head_dim**-0.5,
            "causal": shape.causal,
            "fa_version": fa_version,
        }
        if shape.paged:
            kwargs["seqused_k"] = tensors.seqused_k
            kwargs["block_table"] = tensors.block_table
        else:
            kwargs["cu_seqlens_k"] = tensors.cu_seqlens_k
        return flag_gems.ops.flash_attn_varlen_func(
            tensors.q, tensors.k, tensors.v, **kwargs
        )
    return run_hopper_fa3(tensors, shape, fa_version=fa_version)


def _hopper_fa3_vllm_wrapper(
    tensors: HopperFA3Tensors, shape: HopperFA3Shape, fa_version: int
):
    return run_vllm_hopper_fa3(tensors, shape, fa_version=fa_version)


class HopperFA3Benchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_SHAPE_DESC = (
        "name, seq_lens, num_query_heads, num_kv_heads, head_dim, causal, paged"
    )
    fa_version = 3
    use_vllm_baseline = False

    def set_shapes(self, shape_file_path=None):
        self.shapes = hopper_fa3_benchmark_shapes()

    def init_user_config(self):
        super().init_user_config()
        supported_shapes = []
        skipped_reasons = []
        for shape in self.shapes:
            skip_reason = (
                _hopper_benchmark_shape_skip_reason(shape)
                if self.use_vllm_baseline
                else None
            )
            if skip_reason:
                skipped_reasons.append(f"{shape.name}: {skip_reason}")
                continue
            supported_shapes.append(shape)
        if not supported_shapes:
            details = "; ".join(skipped_reasons) if skipped_reasons else "none"
            pytest.skip(f"No Hopper varlen benchmark shapes are supported ({details}).")
        self.shapes = supported_shapes

    def get_input_iter(self, dtype):
        for idx, shape in enumerate(self.shapes):
            tensors = make_hopper_fa3_varlen(shape, dtype, self.device, seed=2026 + idx)
            yield tensors, shape, self.fa_version

    def unpack_to_args_kwargs(self, input_tuple):
        return list(input_tuple), {}

    def record_shapes(
        self, tensors: HopperFA3Tensors, shape: HopperFA3Shape, fa_version: int
    ):
        return {
            "name": shape.name,
            "seq_lens": shape.seq_lens,
            "num_query_heads": shape.nh_q,
            "num_kv_heads": shape.nh_k,
            "head_dim": shape.head_dim,
            "causal": shape.causal,
            "paged": shape.paged,
            "max_seqlen_q": tensors.max_seqlen_q,
            "max_seqlen_k": tensors.max_seqlen_k,
            "fa_version": fa_version,
        }

    def get_tflops(
        self, op, tensors: HopperFA3Tensors, shape: HopperFA3Shape, fa_version: int
    ):
        return hopper_fa3_attn_flops(shape)


def flash_attn_varlen_legacy(*args, **kwargs):
    """
    Compatibility wrapper for running old flash_attn_varlen_func.
    """
    (
        query,
        key_cache,
        value_cache,
        max_query_len,
        cu_query_lens,
        max_kv_len,
        _,
        seqused_k,
        _,
        dropout_p,
        scale,
        causal,
        window_size,
        soft_cap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_tables,
        _,
        out,
        *_,
    ) = args

    k_flat = key_cache.reshape(-1, key_cache.shape[2], key_cache.shape[3])
    v_flat = value_cache.reshape(-1, value_cache.shape[2], value_cache.shape[3])
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
            torch.cumsum(seqused_k, dim=0),
        ]
    ).to(torch.int32)

    from flash_attn import flash_attn_varlen_func

    result = flash_attn_varlen_func(
        query,  # q
        k_flat,  # k (flattened from key_cache)
        v_flat,  # v (flattened from value_cache)
        cu_query_lens,  # cu_seqlens_q
        cu_seqlens_k,  # cu_seqlens_k (constructed from seqused_k)
        max_query_len,  # max_seqlen_q
        max_kv_len,  # max_seqlen_k
        dropout_p,  # dropout_p
        scale,  # softmax_scale
        causal,  # causal
        tuple(window_size),  # window_size
        float(soft_cap),  # softcap
        alibi_slopes,  # alibi_slopes
        deterministic,  # deterministic
        return_attn_probs,  # return_attn_probs
        block_tables,  # block_table
        alibi_slopes is not None,  # use_alibi (derived from alibi_slopes)
        0,  # alibi_mode
        1,  # imp_mode
        out=out,  # out
        bias=None,  # bias
    )
    return result


@pytest.mark.skipif(vendor_name == "hygon", reason="#2816: RuntimeError")
@pytest.mark.skipif(vendor_name == "cambricon", reason="#2886: TypeError")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_func(monkeypatch, pytestconfig):
    _skip_unless_case_enabled(pytestconfig, "qwenCase")
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    fa_version = _selected_fa_version(pytestconfig)
    _skip_unless_selected_fa_supported(pytestconfig)
    if fa_version == 3:
        _assert_flash_attn_varlen_uses_hopper_backend()

    use_vllm_baseline = _vllm_benchmark_enabled(pytestconfig)

    if not use_vllm_baseline:
        flash_attn_varlen_func = flag_gems.flash_attn_varlen_func
    else:
        _skip_unless_vllm_baseline_available()
        if vendor_name == "iluvatar":
            # iluvatar does not have updated vllm_flash_attn, use conversion wrapper
            flash_attn_varlen_func = flash_attn_varlen_legacy
        else:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                flash_attn_varlen_func,
            )

    gems_op = (
        flag_gems.flash_attn_varlen_func
        if fa_version == 3
        else flag_gems.ops.flash_attn_varlen_func
    )

    bench = FlashAttnVarlenBenchmark(
        op_name="flash_attn_varlen_func",
        torch_op=flash_attn_varlen_func,
        gems_op=gems_op,
        dtypes=[torch.float16, torch.bfloat16],
        fa_version=fa_version,
    )
    if not use_vllm_baseline:
        bench.metrics = ["latency"]
    bench.run()


@pytest.mark.hopper_fa3
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_func_hopper_fa3(pytestconfig):
    _skip_unless_case_enabled(pytestconfig, "prefillDecodePageCase")
    _skip_unless_selected_fa_supported(pytestconfig)
    use_vllm_baseline = _vllm_benchmark_enabled(pytestconfig)
    if use_vllm_baseline and not HAS_VLLM_FA:
        pytest.skip("requires vLLM flash-attention as the benchmark baseline")
    fa_version = _selected_fa_version(pytestconfig)
    if fa_version == 3:
        _assert_flash_attn_varlen_uses_hopper_backend()
    bench = HopperFA3Benchmark(
        op_name="hopper_fa3",
        torch_op=(
            _hopper_fa3_vllm_wrapper
            if use_vllm_baseline
            else _hopper_fa3_gems_wrapper
        ),
        gems_op=_hopper_fa3_gems_wrapper,
        dtypes=[torch.float16, torch.bfloat16],
        fa_version=fa_version,
        use_vllm_baseline=use_vllm_baseline,
    )
    if not use_vllm_baseline:
        bench.metrics = ["latency", "tflops"]
    bench.run()
