import torch

from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    binary_int_args,
    unary_int_arg,
)


def test_perf_embedding():
    def embedding_kwargs(dtype, batch, size):
        input = torch.randint(0, batch, (batch,), device="cuda")
        weight = torch.randn((batch + 1, size), device="cuda", dtype=dtype)
        return {"input": input, "weight": weight}

    bench = Benchmark(
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        arg_func=None,
        dtypes=[
            torch.float32,
            torch.float16,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=embedding_kwargs,
    )
    bench.run()


def test_perf_topk():
    def topk_kwargs(dtype, batch, size):
        x = torch.randn((batch, size), device="cuda", dtype=dtype)
        return {"x": x, "k": 5, "dim": -1}

    bench = Benchmark(
        op_name="topk",
        torch_op=torch.topk,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=topk_kwargs,
    )
    bench.run()


def test_perf_resolve_neg():
    def resolve_neg_arg(dtype, batch, size):
        x = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
        y = x.conj()
        z = y.imag
        return (z,)

    bench = Benchmark(
        op_name="resolve_neg",
        torch_op=torch.resolve_neg,
        arg_func=resolve_neg_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_resolve_conj():
    def resolve_conj_arg(dtype, batch, size):
        x = torch.randn(size=(size, batch), dtype=dtype, device="cuda")
        return (x.conj(),)

    bench = Benchmark(
        op_name="resolve_conj",
        torch_op=torch.resolve_conj,
        arg_func=resolve_conj_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_unique():
    def unique_kwargs(dtype, batch, size):
        return {"sorted": True, "return_inverse": True, "return_counts": False}

    bench = Benchmark(
        op_name="unique",
        torch_op=torch.unique,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=unique_kwargs,
    )
    bench.run()


def test_multinomial_with_replacement():
    def multinomial_args(dtype, batch, size):
        dist = torch.rand((batch, size), dtype=dtype, device="cuda")
        n_samples = 10000
        return (dist, n_samples, True)

    bench = Benchmark(
        op_name="multinomial",
        torch_op=torch.multinomial,
        arg_func=multinomial_args,
        dtypes=(torch.float16, torch.float32),
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_pad():
    def padding_kwargs(dtype, batch, size):
        input = torch.randn((batch, size), device="cuda", dtype=dtype)
        rank = input.ndim
        pad_params = tuple(torch.randint(0, 10, [rank * 2]))
        pad_value = float(torch.randint(0, 1024, [1]))
        return {
            "input": input,
            "pad": pad_params,
            "mode": "constant",
            "value": pad_value,
        }

    bench = Benchmark(
        op_name="padding",
        torch_op=torch.nn.functional.pad,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=padding_kwargs,
    )
    bench.run()


def test_perf_arange():
    def arange_kwargs(dtype, batch, size):
        return {
            "end": batch * size,
            "device": "cuda",
            "dtype": dtype,
        }

    bench = Benchmark(
        op_name="arange",
        torch_op=torch.arange,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=arange_kwargs,
    )
    bench.run()


def test_perf_isin():
    bench = Benchmark(
        op_name="isin",
        torch_op=torch.isin,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_fill():
    def fill_kwargs(dtype, batch, size):
        value = 1.0
        input = torch.empty(batch * size, dtype=dtype, device="cuda")
        return {
            "input": input,
            "value": value,
        }

    bench = Benchmark(
        op_name="fill",
        torch_op=torch.fill,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=fill_kwargs,
    )
    bench.run()


def test_perf_stack():
    def stack_args(dtype, batch, size):
        inp = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
        return {(inp,) * 3}

    bench = Benchmark(
        op_name="stack",
        torch_op=torch.stack,
        arg_func=stack_args,
        dtypes=FLOAT_DTYPES,
        batch=(512),
        sizes=SIZES,
    )
    bench.run()


def test_perf_hstack():
    def hstack_args(dtype, batch, size):
        inp = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
        return {(inp,) * 3}

    bench = Benchmark(
        op_name="hstack",
        torch_op=torch.hstack,
        arg_func=hstack_args,
        dtypes=FLOAT_DTYPES,
        batch=(512),
        sizes=SIZES,
    )
    bench.run()


def test_perf_cat():
    def cat_args(dtype, batch, size):
        inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
        return [[inp1, inp2]]

    def cat_kwargs(dtype, batch, size):
        return {"dim": 0}

    bench = Benchmark(
        op_name="cat",
        torch_op=torch.cat,
        arg_func=cat_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=cat_kwargs,
    )
    bench.run()


def test_perf_cat_int():
    def cat_args(dtype, batch, size):
        inp1 = torch.randint(
            low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
        )
        inp2 = torch.randint(
            low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
        )
        return [[inp1, inp2]]

    def cat_kwargs(dtype, batch, size):
        return {"dim": 0}

    bench = Benchmark(
        op_name="cat_int",
        torch_op=torch.cat,
        arg_func=cat_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=cat_kwargs,
    )
    bench.run()


def test_perf_vstack():
    def vstack_args(dtype, batch, size):
        inp1 = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
        inp2 = torch.randn(size=(batch + 1, size), dtype=dtype, device="cuda")
        inp3 = torch.randn(size=(batch + 2, size), dtype=dtype, device="cuda")
        return [[inp1, inp2, inp3]]

    bench = Benchmark(
        op_name="vstack",
        torch_op=torch.vstack,
        arg_func=vstack_args,
        dtypes=FLOAT_DTYPES,
        batch=(512),
        sizes=SIZES,
    )
    bench.run()


def test_perf_repeat_interleave_self_int():
    def repeat_interleave_self_int_arg(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        repeats = 2
        return inp, repeats

    bench = Benchmark(
        op_name="repeat_interleave_self_int",
        torch_op=torch.repeat_interleave,
        arg_func=repeat_interleave_self_int_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_repeat_interleave_tensor():
    def repeat_interleave_tensor_arg(dtype, batch, size):
        repeats = torch.randint(
            low=0,
            high=0x7F,
            size=[
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        return (repeats,)

    bench = Benchmark(
        op_name="repeat_interleave_tensor",
        torch_op=torch.repeat_interleave,
        arg_func=repeat_interleave_tensor_arg,
        dtypes=[torch.int32],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()
