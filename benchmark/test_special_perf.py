import random

import pytest
import torch

import flag_gems
from benchmark.attri_util import BOOL_DTYPES, FLOAT_DTYPES, INT_DTYPES, BenchLevel
from benchmark.performance_utils import (
    Config,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    GenericBenchmarkExcluse1D,
    GenericBenchmarkExcluse3D,
    SkipVersion,
    generate_tensor_input,
    vendor_name,
)


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


@pytest.mark.embedding_backward
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
