import random
from typing import Generator

import pytest
import torch
import triton

import flag_gems
from benchmark.attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
)
from benchmark.performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    GenericBenchmark4DOnly,
    GenericBenchmarkExcluse1D,
    GenericBenchmarkExcluse3D,
    SkipVersion,
    generate_tensor_input,
    vendor_name,
)


class GroupedTopKBenchmark(Benchmark):
    def __init__(
        self,
        op_name,
        torch_op,
        dtypes,
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func=0,
    ):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func

    def set_shapes(self, shape_file_path=None):
        grouped_topk_configs = [
            (1, 64, 8, 2, 8),
            (8, 64, 8, 2, 8),
            (32, 64, 8, 2, 8),
            (64, 64, 8, 2, 8),
            (128, 64, 8, 2, 8),
            (256, 64, 8, 2, 8),
            (32, 128, 8, 2, 8),
            (64, 128, 8, 2, 8),
            (128, 128, 8, 2, 8),
            (64, 64, 4, 2, 4),
            (64, 128, 16, 2, 8),
            (512, 64, 8, 2, 8),
            (1024, 64, 8, 2, 8),
            (2048, 64, 8, 2, 8),
        ]
        self.shapes = grouped_topk_configs

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.grouped_topk_input_fn(config, cur_dtype, self.device)

    def grouped_topk_input_fn(self, config, dtype, device):
        num_tokens, num_experts, n_group, topk_group, topk = config

        scores = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
        bias = torch.randn(num_experts, device=device, dtype=dtype)

        yield (
            scores,
            n_group,
            topk_group,
            topk,
            self.renormalize,
            self.routed_scaling_factor,
            bias,
            self.scoring_func,
        )


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the grouped_topk kernel.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.grouped_topk
def test_perf_grouped_topk():
    try:
        from vllm._custom_ops import grouped_topk as vllm_grouped_topk
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Skipped due to missing vLLM grouped_topk: {e}")

    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=vllm_grouped_topk,
        dtypes=[torch.float32, torch.bfloat16],
        renormalize=True,
        scoring_func=0,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the grouped_topk kernel.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.grouped_topk
def test_perf_grouped_topk_no_renorm():
    try:
        from vllm._custom_ops import grouped_topk as vllm_grouped_topk
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Skipped due to missing vLLM grouped_topk: {e}")

    bench = GroupedTopKBenchmark(
        op_name="grouped_topk_no_renorm",
        torch_op=vllm_grouped_topk,
        dtypes=[torch.float32, torch.bfloat16],
        renormalize=False,
        scoring_func=0,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the grouped_topk kernel.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.grouped_topk_sigmoid
def test_perf_grouped_topk_sigmoid():
    try:
        from vllm._custom_ops import grouped_topk as vllm_grouped_topk
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Skipped due to missing vLLM grouped_topk: {e}")

    bench = GroupedTopKBenchmark(
        op_name="grouped_topk_sigmoid",
        torch_op=vllm_grouped_topk,
        dtypes=[torch.float32, torch.bfloat16],
        renormalize=True,
        scoring_func=1,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()


def topk_input_fn(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    k = 5 if shape[-1] > 5 else shape[-1]
    yield {"x": x, "k": k, "dim": -1},
    # TODO:  Currently only support sorted == True and only support topk in last dimension
    # if Config.bench_level == BenchLevel.COMPREHENSIVE:
    #     k = 5 if shape[0] > 5 else shape[0]
    #     yield {"x": x, "k": k, "dim": 0},
    #     yield {"x": x, "k": k, "dim": -1, "sorted": False},


def resolve_neg_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    if vendor_name == "mthreads":
        yield x.conj(),
    else:
        yield x.conj().imag,


def resolve_conj_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj(),


special_operations = [
    # Sorting Operations
    ("topk", torch.topk, FLOAT_DTYPES, topk_input_fn),
    # Complex Operations
    ("resolve_neg", torch.resolve_neg, [torch.cfloat], resolve_neg_input_fn),
    ("resolve_conj", torch.resolve_conj, [torch.cfloat], resolve_conj_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes, input_fn",
    [
        pytest.param(
            op,
            fn,
            dtypes,
            input_fn,
            marks=getattr(pytest.mark, op, None),
        )
        for op, fn, dtypes, input_fn in special_operations
    ],
)
def test_special_operations_benchmark(op_name, torch_op, dtypes, input_fn):
    bench = GenericBenchmarkExcluse1D(
        input_fn=input_fn, op_name=op_name, dtypes=dtypes, torch_op=torch_op
    )
    bench.run()


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.isin
def test_isin_perf():
    def isin_input_fn(shape, dtype, device):
        elements = generate_tensor_input(shape, dtype, device)
        test_elements = generate_tensor_input(shape, dtype, device)
        yield elements, test_elements
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            # assume_unique set to True
            uniq_elements = torch.unique(generate_tensor_input(shape, dtype, device))
            uniq_test_elements = torch.unique(
                generate_tensor_input(shape, dtype, device)
            )
            yield uniq_elements, uniq_test_elements, {"assume_unique": True}

    bench = GenericBenchmark2DOnly(
        input_fn=isin_input_fn,
        op_name="isin",
        torch_op=torch.isin,
        dtypes=INT_DTYPES,
    )
    bench.run()


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.unique
def test_perf_unique():
    def unique_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},

    bench = GenericBenchmark2DOnly(
        input_fn=unique_input_fn,
        op_name="unique",
        torch_op=torch.unique,
        dtypes=INT_DTYPES,
    )
    bench.run()


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.sort
def test_perf_sort():
    class SortBenchmark(GenericBenchmark2DOnly):
        def set_more_shapes(self):
            return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]

    def sort_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp, {"dim": -1, "descending": False},

    bench = SortBenchmark(
        input_fn=sort_input_fn,
        op_name="sort",
        torch_op=torch.sort,
        dtypes=INT_DTYPES + FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.multinomial
def test_multinomial_with_replacement():
    def multinomial_input_fn(shape, dtype, device):
        dist = torch.rand(shape, dtype=dtype, device=device)
        n_samples = 10000
        yield dist, n_samples, True,

    bench = GenericBenchmark2DOnly(
        input_fn=multinomial_input_fn,
        op_name="multinomial",
        torch_op=torch.multinomial,
        dtypes=(torch.float16, torch.float32),
    )
    bench.run()


@pytest.mark.pad
def test_perf_pad():
    def pad_input_fn(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        rank = input.ndim
        pad_params = [random.randint(0, 10) for _ in range(rank * 2)]
        pad_value = float(torch.randint(0, 1024, [1]))
        yield input, {
            "pad": pad_params,
            "mode": "constant",
            "value": pad_value,
        },

    bench = GenericBenchmark(
        input_fn=pad_input_fn,
        op_name="pad",
        torch_op=torch.nn.functional.pad,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class EmbeddingBenchmark(GenericBenchmark2DOnly):
    def set_more_shapes(self):
        # TODO: add more shapes
        return None


def embedding_input_fn(shape, dtype, device):
    num_embeddings, embedding_dim = shape
    indices = torch.randint(0, num_embeddings, (num_embeddings,), device=device)
    weight = torch.randn((num_embeddings, embedding_dim), device=device, dtype=dtype)
    yield {"input": indices, "weight": weight},
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        indices_2d = torch.randint(
            0,
            num_embeddings,
            (num_embeddings, num_embeddings),
            device=device,
        )
        yield {"input": indices_2d, "weight": weight},


def embedding_backward_input_fn(shape, dtype, device):
    for forward_args in embedding_input_fn(shape, dtype, device):
        # print(f'forward_args = {forward_args}')
        input = forward_args[0]["input"]
        weight = forward_args[0]["weight"]
        # print(f'weight = {weight}')
        weight.requires_grad_(True)
        # import pudb; pudb.set_trace()
        # output = torch.nn.functional.embedding(input, weight)
        # grad_output = torch.randn_like(output)
        yield input, weight


@pytest.mark.embedding
def test_perf_embedding():
    bench = EmbeddingBenchmark(
        input_fn=embedding_input_fn,
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        dtypes=[
            torch.float32,
            torch.float16,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
    )
    bench.run()


@pytest.mark.embedding
def test_perf_embedding_backward():
    bench = EmbeddingBenchmark(
        input_fn=embedding_backward_input_fn,
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        dtypes=[
            torch.float32,
            torch.float16,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
        is_backward=True,
    )
    bench.run()


class EmbeddingDenseBackwardBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (32, 2048, 128, 8192),
            (16, 2048, 256, 16384),
            (8, 4096, 256, 32768),
        ]


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (flag_gems.device != "cuda"),
    reason="CUDA backend is not available for this benchmark.",
)
@pytest.mark.embedding_dense_backward
def test_perf_embedding_dense_backward():
    bench = EmbeddingDenseBackwardBenchmark(
        input_fn=embedding_dense_backward_input_fn,
        op_name="embedding_dense_backward",
        torch_op=torch.ops.aten.embedding_dense_backward,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


def embedding_dense_backward_input_fn(shape, dtype, device):
    B, M, D, num_weights = shape

    grad_output = torch.randn((B, M, D), device=device, dtype=dtype)
    indices = torch.randint(0, num_weights, (B, M), device=device, dtype=torch.long)

    def inject_padding_idx(cur_indices: torch.Tensor, padding_idx: int) -> torch.Tensor:
        if padding_idx < 0:
            return cur_indices
        mask = torch.rand((B, M), device=device) < 0.25
        return torch.where(mask, torch.full_like(cur_indices, padding_idx), cur_indices)

    test_cases = [(-1, False), (0, True), (5, False)]
    for padding_idx, scale_grad_by_freq in test_cases:
        cur_indices = inject_padding_idx(indices, padding_idx)
        yield grad_output, cur_indices, num_weights, padding_idx, scale_grad_by_freq


def lerp_input_fn(shape, dtype, device):
    input = torch.randn(*shape, device=device, dtype=dtype)
    end = input + 10
    weight = torch.randn(*shape, device=device, dtype=dtype)
    yield {"input": input, "end": end, "weight": weight},


class LerpBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


@pytest.mark.lerp
@pytest.mark.skipif(
    vendor_name == "kunlunxin" and SkipVersion("torch", "<2.5"),
    reason="The half dtype is only supported on torch >= 2.5.",
)
def test_perf_lerp():
    bench = LerpBenchmark(
        input_fn=lerp_input_fn,
        op_name="lerp",
        torch_op=torch.lerp,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.lerp_
@pytest.mark.skipif(
    vendor_name == "kunlunxin" and SkipVersion("torch", "<2.5"),
    reason="The half dtype is only supported on torch >= 2.5.",
)
def test_perf_lerp_inplace():
    bench = LerpBenchmark(
        input_fn=lerp_input_fn,
        op_name="lerp_",
        torch_op=lambda input, end, weight: input.lerp_(end, weight),
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


class UpsampleBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


@pytest.mark.upsample_bicubic2d_aa
def test_perf_upsample_bicubic2d_aa():
    def upsample_bicubic2d_aa_input_fn(shape, dtype, device):
        batch, channel, height, weight = shape
        input = torch.randn(size=shape, device=device, dtype=dtype)
        scale_factors = (2, 2)
        output_size = (
            int(height * scale_factors[0]),
            int(weight * scale_factors[1]),
        )
        yield {
            "input": input,
            "output_size": output_size,
            "align_corners": False,
            "scales_h": None,
            "scales_w": None,
        },

    if vendor_name == "cambricon":
        dtypes = [torch.float32]
    elif vendor_name == "kunlunxin":
        dtypes = [torch.float32, torch.float16]
    else:
        dtypes = FLOAT_DTYPES
    bench = UpsampleBenchmark(
        input_fn=upsample_bicubic2d_aa_input_fn,
        op_name="upsample_bicubic2d_aa",
        torch_op=torch._C._nn._upsample_bicubic2d_aa,
        dtypes=dtypes,
    )
    bench.run()


@pytest.mark.upsample_linear1d
@pytest.mark.parametrize("align_corners", [False, True])
def test_perf_upsample_linear1d(align_corners):
    def upsample_linear1d_input_fn(shape, dtype, device):
        batch, channel, height, width = shape
        length = height * width
        input = torch.randn((batch, channel, length), device=device, dtype=dtype)
        scale_factors = 2
        output_size = int(length * scale_factors)
        yield {
            "input": input,
            "output_size": (output_size,),
            "align_corners": align_corners,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_linear1d_input_fn,
        op_name=f"upsample_linear1d_align_{align_corners}",
        torch_op=torch._C._nn.upsample_linear1d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest1d
def test_perf_upsample_nearest1d():
    def upsample_nearest1d_input_fn(shape, dtype, device):
        batch, channel, height, width = shape
        length = height * width  # flatten spatial dims to 1D length
        input = torch.randn((batch, channel, length), device=device, dtype=dtype)
        scale_factors = 2
        output_size = int(length * scale_factors)
        yield {
            "input": input,
            "output_size": (output_size,),
            "scales": None,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_nearest1d_input_fn,
        op_name="upsample_nearest1d",
        torch_op=torch._C._nn.upsample_nearest1d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest2d
def test_perf_upsample_nearest2d():
    def upsample_nearest2d_input_fn(shape, dtype, device):
        batch, channel, height, weight = shape
        input = torch.randn(size=shape, device=device, dtype=dtype)
        scale_factors = (2, 2)
        output_size = (
            int(height * scale_factors[0]),
            int(weight * scale_factors[1]),
        )
        yield {
            "input": input,
            "output_size": output_size,
            "scales_h": None,
            "scales_w": None,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_nearest2d_input_fn,
        op_name="upsample_nearest2d",
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest3d
def test_perf_upsample_nearest3d():
    def upsample_nearest3d_input_fn(shape, dtype, device):
        batch, channel, height, width = shape
        depth = 4
        width = width // 4
        new_height = height // depth
        real_shape = (batch, channel, depth, new_height, width)

        input = torch.randn(size=real_shape, device=device, dtype=dtype)
        scale_factors = (2.0, 2.0, 2.0)
        output_size = (
            int(depth * scale_factors[0]),
            int(new_height * scale_factors[1]),
            int(width * scale_factors[2]),
        )

        yield {
            "input": input,
            "output_size": output_size,
            "scales_d": None,
            "scales_h": None,
            "scales_w": None,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_nearest3d_input_fn,
        op_name="upsample_nearest3d",
        torch_op=torch._C._nn.upsample_nearest3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.diag
def test_perf_diag():
    def diag_input_fn(shape, dtype, device):
        input = generate_tensor_input(shape, dtype, device)
        diagonal = random.randint(-4, 4)
        yield input, {
            "diagonal": diagonal,
        },

    bench = GenericBenchmarkExcluse3D(
        input_fn=diag_input_fn,
        op_name="diag",
        torch_op=torch.diag,
        dtypes=FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.diag_embed
def test_perf_diag_embed():
    def diag_embed_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield {"input": inp},

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            yield {"input": inp, "offset": 1, "dim1": 0, "dim2": -1},

    bench = EmbeddingBenchmark(
        input_fn=diag_embed_input_fn,
        op_name="diag_embed",
        torch_op=torch.diag_embed,
        dtypes=FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
    )

    bench.run()


@pytest.mark.diagonal
def test_perf_diagonal_backward():
    def diagonal_backward_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp,

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            yield inp, {"offset": 1, "dim1": 0, "dim2": -1},

    bench = GenericBenchmarkExcluse1D(
        input_fn=diagonal_backward_input_fn,
        op_name="diagonal",
        torch_op=torch.diagonal,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()


@pytest.mark.skipif(
    vendor_name == "kunlunxin" and SkipVersion("torch", "<2.5"),
    reason="only support torch >= 2.5.",
)
@pytest.mark.kron
def test_perf_kron():
    class KronBenchmark(GenericBenchmark2DOnly):
        def set_more_shapes(self):
            return None

    def kron_input_fn(shape, dtype, device):
        inp1 = generate_tensor_input(shape, dtype, device)
        inp2 = generate_tensor_input(shape, dtype, device)
        yield inp1, inp2

    bench = KronBenchmark(
        input_fn=kron_input_fn,
        op_name="kron",
        torch_op=torch.kron,
        dtypes=FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.contiguous
def test_perf_contiguous():
    def contiguous_input_fn(shape, dtype, device):
        if dtype in FLOAT_DTYPES:
            inp = torch.randn(shape, dtype=dtype, device=device)
        else:
            inp = torch.randint(
                low=-10000, high=10000, size=shape, dtype=dtype, device="cpu"
            ).to(device)
        inp = inp[::2]
        yield inp,

    bench = GenericBenchmark(
        input_fn=contiguous_input_fn,
        op_name="torch.Tensor.contiguous",
        torch_op=torch.Tensor.contiguous,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
    )

    bench.run()


class RWKVSparsityBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.rwkv_mm_sparsity
def test_perf_rwkv_mm_sparsity():
    def rwkv_mm_sparsity_input_fn(shape, dtype, device):
        n = 16384
        embedding_dim = 4096

        V_ = torch.randn(n, embedding_dim, dtype=dtype, device=device)
        sparsity_levels = [0.9]
        for target_sparsity in sparsity_levels:
            k_sparse = torch.randn(n, dtype=dtype, device=device)
            threshold = torch.quantile(
                k_sparse.abs().to(torch.float32), target_sparsity
            ).to(dtype)
            k_sparse = torch.relu(k_sparse - threshold)
            yield k_sparse, V_

    def torch_rwkv_mm_sparsity(k, v):
        return torch.mv(v.T, k)

    torch_op = torch_rwkv_mm_sparsity
    gems_op = flag_gems.rwkv_mm_sparsity

    bench = RWKVSparsityBenchmark(
        input_fn=rwkv_mm_sparsity_input_fn,
        op_name="rwkv_mm_sparsity",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


class RWKVBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.rwkv_ka_fusion
def test_perf_rwkv_ka_fusion():
    def rwkv_ka_fusion_input_fn(shape, dtype, device):
        T = shape[0]
        H = 8
        N = 64
        C = H * N

        k = torch.randn(T, C, dtype=dtype, device=device)
        kk = torch.randn(C, dtype=dtype, device=device)
        a = torch.randn(T, C, dtype=dtype, device=device)
        ka = torch.randn(C, dtype=dtype, device=device)

        yield k, kk, a, ka, H, N

    def torch_rwkv_ka(k, kk, a, ka, H, N):
        T, C = k.shape
        assert (
            C == H * N and kk.shape == (C,) and a.shape == (T, C) and ka.shape == (C,)
        )
        o_kk = torch.nn.functional.normalize(
            (k * kk).view(T, H, N), dim=-1, p=2.0
        ).view(T, H * N)
        o_k = k * (1 + (a - 1) * ka)
        o_kka = o_kk * a

        return o_k, o_kk, o_kka

    torch_op = torch_rwkv_ka
    gems_op = flag_gems.rwkv_ka_fusion

    bench = RWKVBenchmark(
        input_fn=rwkv_ka_fusion_input_fn,
        op_name="rwkv_ka_fusion",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.moe_sum
def test_perf_moe_sum():
    def moe_sum_input_fn(shape, dtype, device):
        shape = (shape[0], 1, shape[1]) if len(shape) == 2 else shape
        num_tokens, topk, hidden_size = shape
        input_tensor = torch.randn(
            num_tokens,
            topk,
            hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

        output_tensor = torch.empty(
            num_tokens, hidden_size, dtype=dtype, device=device, requires_grad=False
        )
        yield input_tensor, output_tensor

    def torch_op(input_tensor, output_tensor):
        output_tensor.copy_(input_tensor.sum(dim=1))

    gems_op = flag_gems.moe_sum

    bench = GenericBenchmarkExcluse1D(
        input_fn=moe_sum_input_fn,
        op_name="moe_sum",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_op)
    bench.run()


try:
    import os

    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    import vllm._custom_ops as vllm_ops

    HAS_VLLM = True
    WARP_SIZE = 32
except ImportError:
    HAS_VLLM = False


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_perf_moe_align_block_size():
    def moe_align_block_size_input_fn(shape, dtype, device):
        num_experts = shape[0]
        block_size = shape[1]
        dtype = torch.int32
        topk_ids = torch.randint(
            0, num_experts, (shape[2], shape[3]), dtype=dtype, device=device
        )
        max_num_tokens_padded = ((num_experts + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE

        # padded_num_experts in vllm._custom_ops.moe_align_block_size
        # must be less than 1024
        if max_num_tokens_padded >= 1024:
            return

        sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
        max_num_m_blocks = max_num_tokens_padded // block_size
        expert_ids = torch.empty((max_num_m_blocks,), dtype=dtype, device=device)
        num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

        yield (
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )

    class MoeAlignBlockSizeBenchmark(GenericBenchmark4DOnly):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_shapes(self, shape_file_path: None):
            moe_align_block_size_shape = [
                (512, 64, 16384, 10),
                (512, 64, 6152, 10),
                (512, 64, 4727, 10),
                (512, 64, 1905, 10),
                (512, 64, 11575, 10),
                (512, 64, 1032, 10),
                (512, 64, 4201, 10),
                (512, 64, 2056, 10),
                (512, 64, 7561, 10),
                (512, 64, 4104, 10),
                (512, 64, 14281, 10),
            ]
            self.shapes = moe_align_block_size_shape

        def set_more_shapes(self):
            return None

    gems_op = flag_gems.moe_align_block_size_triton
    bench = MoeAlignBlockSizeBenchmark(
        op_name="moe_align_block_size_triton",
        input_fn=moe_align_block_size_input_fn,
        torch_op=vllm_ops.moe_align_block_size,
        dtypes=[
            torch.int32,
        ],
    )

    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.replication_pad3d
def test_perf_replication_pad3d():
    def replication_pad3d_input_fn(shape, dtype, device):
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
        p = random.randint(1, 3)
        padding = (p, p, p, p, p, p)
        yield input_tensor, {"padding": padding}

    def torch_replication_pad3d(input, padding):
        return torch.nn.functional.pad(input, padding, mode="replicate")

    def gems_wrapper(input, padding):
        return flag_gems.replication_pad3d(input, padding)

    class ReplicationPad3dBenchmark(GenericBenchmarkExcluse3D):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_shapes(self, shape_file_path=None):
            replication_pad3d_shapes = [
                (1, 3, 16, 256, 256),
                (4, 16, 32, 64, 64),
                (8, 64, 8, 32, 32),
                (2, 32, 16, 128, 128),
                (1, 1, 64, 128, 128),
            ]
            self.shapes = replication_pad3d_shapes

        def set_more_shapes(self):
            return None

    bench = ReplicationPad3dBenchmark(
        input_fn=replication_pad3d_input_fn,
        op_name="replication_pad3d",
        torch_op=torch_replication_pad3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_wrapper)
    bench.run()


def torch_per_token_group_quant_fp8_ref(x, group_size, scale_ue8m0):
    dtype = flag_gems.SUPPORTED_FP8_DTYPE
    eps = 1e-10
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    if scale_ue8m0:
        min_val = torch.tensor(1e-10, dtype=x_s.dtype, device=x_s.device)
        x_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(x_s.abs(), min_val))))
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s


class PerTokenGroupQuantFp8Benchmark(GenericBenchmark):
    """
    benchmark for per_token_group_quant_fp8
    """

    def set_more_shapes(self):
        return None


@pytest.mark.per_token_group_quant_fp8
def test_perf_per_token_group_quant_fp8():
    def input_kwargs(shape, dtype, device):
        (
            num_tokens,
            d,
            group_size,
        ) = shape
        scale_ue8m0 = random.choice([True, False])
        x = torch.rand(num_tokens, d, dtype=dtype, device=device)

        yield (
            x,
            group_size,
            scale_ue8m0,
        )

    bench = PerTokenGroupQuantFp8Benchmark(
        op_name="per_token_group_quant_fp8",
        input_fn=input_kwargs,
        torch_op=torch_per_token_group_quant_fp8_ref,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.per_token_group_quant_fp8)
    bench.run()


@pytest.mark.conj_physical
def test_perf_conj_physical():
    def conj_physical_input_fn(shape, dtype, device):
        if dtype.is_complex:
            float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
            real = torch.randn(shape, dtype=float_dtype, device=device)
            imag = torch.randn(shape, dtype=float_dtype, device=device)
            input_tensor = torch.complex(real, imag).to(dtype)
        elif dtype.is_floating_point:
            input_tensor = torch.randn(shape, dtype=dtype, device=device)
        else:
            input_tensor = torch.randn(shape, device=device).to(dtype)
        yield (input_tensor,)

    def torch_conj_physical(input):
        return torch.conj_physical(input)

    def gems_wrapper(input):
        return flag_gems.conj_physical(input)

    class Conj_physicalBenchmark(GenericBenchmarkExcluse3D):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_shapes(self, shape_file_path=None):
            conj_physical_shapes = [
                (256,),
                (2048, 2048),
                (128, 512, 256),
                (32, 64),
                (512, 1024),
                (2, 3, 4),
            ]
            self.shapes = conj_physical_shapes

        def set_more_shapes(self):
            return None

    dtypes = FLOAT_DTYPES + INT_DTYPES + COMPLEX_DTYPES
    bench = Conj_physicalBenchmark(
        input_fn=conj_physical_input_fn,
        op_name="conj_physical",
        torch_op=torch_conj_physical,
        dtypes=dtypes,
    )
    bench.set_gems(flag_gems.conj_physical)
    bench.run()


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "shape",
    [
        (3, 33, 33),
        (2, 4, 32, 64),
        (8, 16, 64, 64),
        (32, 64, 128, 256),
        (16, 32, 64, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
    ],
)
def test_reflection_pad2d_benchmark_tensor(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad2d(ref_x, padding),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: flag_gems.reflection_pad2d(x, padding), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_bicubic2d
@pytest.mark.parametrize("align_corners", [False, True])
def test_perf_upsample_bicubic2d(align_corners):
    def upsample_bicubic2d_input_fn(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        scale_factors = [2.0, 2.0]
        output_size = None
        yield {
            "input": input,
            "output_size": output_size,
            "align_corners": align_corners,
            "scale_factors": scale_factors,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_bicubic2d_input_fn,
        op_name=f"upsample_bicubic2d_align_{align_corners}",
        torch_op=torch._C._nn.upsample_bicubic2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.reflection_pad1d
def test_perf_reflection_pad1d():
    def reflection_pad1d_input_fn(config, dtype, device):
        shape, padding = config
        x = torch.randn(shape, dtype=dtype, device=device)
        yield x, list(padding)

    class ReflectionPad1dBenchmark(Benchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [
                ((3, 33), (1, 1)),
                ((2, 4, 64), (3, 5)),
                ((8, 16, 256), (8, 8)),
                ((32, 64, 2048), (3, 5)),
            ]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield from reflection_pad1d_input_fn(config, cur_dtype, self.device)

    bench = ReflectionPad1dBenchmark(
        op_name="reflection_pad1d",
        torch_op=torch.ops.aten.reflection_pad1d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pixel_unshuffle
def test_perf_pixel_unshuffle():
    def pixel_unshuffle_input_fn(config, dtype, device):
        shape, downscale_factor = config
        x = torch.randn(shape, dtype=dtype, device=device)
        yield x, downscale_factor

    class PixelUnshuffleBenchmark(Benchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [
                ((1, 3, 8, 8), 2),
                ((2, 4, 12, 6), 3),
                ((4, 16, 64, 48), 4),
            ]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield from pixel_unshuffle_input_fn(config, cur_dtype, self.device)

    bench = PixelUnshuffleBenchmark(
        op_name="pixel_unshuffle",
        torch_op=torch.ops.aten.pixel_unshuffle,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.replication_pad1d
def test_perf_replication_pad1d():
    def replication_pad1d_input_fn(config, dtype, device):
        shape, padding = config
        x = torch.randn(shape, dtype=dtype, device=device)
        yield x, list(padding)

    class ReplicationPad1dBenchmark(Benchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [
                ((2, 3, 7), (1, 2)),
                ((4, 16, 64), (3, 1)),
                ((8, 32, 256), (1, 2)),
                ((32, 256), (3, 1)),
            ]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield from replication_pad1d_input_fn(config, cur_dtype, self.device)

    bench = ReplicationPad1dBenchmark(
        op_name="replication_pad1d",
        torch_op=torch.ops.aten.replication_pad1d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unfold
def test_perf_unfold_backward():
    def unfold_backward_input_fn(config, dtype, device):
        input_sizes, dim, size, step = config
        d = dim % len(input_sizes)
        num_windows = (input_sizes[d] - size) // step + 1
        grad_shape = (
            list(input_sizes[:d]) + [num_windows] + list(input_sizes[d + 1 :]) + [size]
        )
        grad_in = torch.randn(grad_shape, dtype=dtype, device=device)
        yield grad_in, list(input_sizes), dim, size, step

    class UnfoldBackwardBenchmark(Benchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [
                ((32, 64), 1, 16, 16),
                ((16, 33), 0, 5, 2),
                ((4, 8, 12), -1, 6, 4),
                ((7, 13), 1, 13, 3),
                ((6, 20), 1, 7, 4),
                ((2, 3, 17), -1, 9, 1),
                ((2, 17), 1, 4, 6),
            ]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield from unfold_backward_input_fn(config, cur_dtype, self.device)

    bench = UnfoldBackwardBenchmark(
        op_name="unfold_backward",
        torch_op=torch.ops.aten.unfold_backward,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.set_gems(flag_gems.unfold_backward)
    bench.run()


@pytest.mark.lift_fresh_copy
def test_perf_lift_fresh_copy():
    bench = GenericBenchmark(
        input_fn=lambda shape, dtype, device: (
            iter([(torch.randn(shape, dtype=dtype, device=device),)])
        ),
        op_name="lift_fresh_copy",
        torch_op=torch.ops.aten.lift_fresh_copy,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest_exact1d
def test_perf__upsample_nearest_exact1d():
    class UpsampleNearestExact1dBenchmark(Benchmark):
        def set_shapes(self, shape_file_path=None):
            self.shapes = [(2, 3, 16), (4, 8, 64), (8, 16, 256), (16, 32, 512)]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for shape in self.shapes:
                x = torch.randn(shape, dtype=cur_dtype, device=self.device)
                out_size = [shape[-1] * 2]
                yield x, out_size, None

    bench = UpsampleNearestExact1dBenchmark(
        op_name="_upsample_nearest_exact1d",
        torch_op=torch.ops.aten._upsample_nearest_exact1d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.margin_ranking_loss
def test_perf_margin_ranking_loss():
    def margin_ranking_loss_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=device)
        inp2 = torch.randn(shape, dtype=dtype, device=device)
        target = (
            torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1
        ).to(dtype)
        yield inp1, inp2, target, 0.5, 1

    bench = GenericBenchmark(
        input_fn=margin_ranking_loss_input_fn,
        op_name="margin_ranking_loss",
        torch_op=torch.ops.aten.margin_ranking_loss,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.soft_margin_loss
def test_perf_soft_margin_loss():
    def soft_margin_loss_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        target = (torch.randint(0, 2, shape, device=device).to(dtype) * 2) - 1
        yield inp, target

    bench = GenericBenchmark(
        input_fn=soft_margin_loss_input_fn,
        op_name="soft_margin_loss",
        torch_op=torch.ops.aten.soft_margin_loss,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class SafeSoftmaxBenchmark(Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, -1, None


@pytest.mark.safe_softmax
def test_perf__safe_softmax():
    bench = SafeSoftmaxBenchmark(
        op_name="_safe_softmax",
        torch_op=torch.ops.aten._safe_softmax,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class TCopyBenchmark(Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) == 2:
                inp = generate_tensor_input(shape, cur_dtype, self.device)
                yield inp,


@pytest.mark.t_copy
def test_perf_t_copy():
    bench = TCopyBenchmark(
        op_name="t_copy",
        torch_op=torch.ops.aten.t_copy,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class UpsampleBicubic2dAaBackwardBenchmark(Benchmark):
    def set_more_shapes(self):
        shapes = [(512, 1024, 32, 32), (256, 512, 64, 64)]
        shapes_4d = [(4, 16, 2**i, 2**i) for i in range(2, 8, 2)]
        shapes_rect = [(4, 16, 2**i, 2 ** (i + 1)) for i in range(2, 7, 2)]
        return shapes + shapes_4d + shapes_rect

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if len(shape) == 1:
                # Treat flat shape as a square spatial dim: (1, 1, sqrt(N), sqrt(N))
                side = int(shape[0] ** 0.5)
                shape_4d = (1, 1, side, side)
            elif len(shape) == 2:
                shape_4d = (1, 1, shape[0], shape[1])
            elif len(shape) == 3:
                shape_4d = (1, shape[0], shape[1], shape[2])
            else:
                shape_4d = shape
            for scale_factor in [0.25, 0.5, 2.0]:
                for align_corners in [False, True]:
                    N, C, H_in, W_in = shape_4d
                    H_out = max(1, int(H_in * scale_factor))
                    W_out = max(1, int(W_in * scale_factor))
                    if N * C * max(H_in * W_in, H_out * W_out) >= 2**30:
                        continue
                    grad = torch.randn(
                        [N, C, H_out, W_out], device=self.device, dtype=cur_dtype
                    )
                    yield grad, [H_out, W_out], [N, C, H_in, W_in], align_corners, None, None

    def get_tflops(self, op, *args, **kwargs):
        grad, output_size, input_size, align_corners, scales_h, scales_w = args
        return grad.numel() * 2


@pytest.mark.upsample_bicubic2d_aa_backward
@pytest.mark.parametrize(
    "dtype",
    [torch.float32], #FLOAT_DTYPES,
)
def test_upsample_bicubic2d_aa_backward_perf(dtype):
    bench = UpsampleBicubic2dAaBackwardBenchmark(
        op_name="upsample_bicubic2d_aa_backward",
        torch_op=torch.ops.aten._upsample_bicubic2d_aa_backward,
        dtypes=[dtype],
    )

    bench.run()
