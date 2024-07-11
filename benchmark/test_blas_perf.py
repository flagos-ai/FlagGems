import torch

from .performance_utils import BLAS_BATCH, DEFAULT_BATCH, FLOAT_DTYPES, SIZES, Benchmark


def test_perf_addmm():
    def addmm_args(dtype, batch, size):
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        inp1 = torch.randn([size, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([size, size], dtype=dtype, device="cuda")
        return bias, inp1, inp2

    bench = Benchmark(
        op_name="addmm",
        torch_op=torch.addmm,
        arg_func=addmm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_bmm():
    def bmm_args(dtype, batch, size):
        inp1 = torch.randn([batch, size, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([batch, size, size], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="bmm",
        torch_op=torch.bmm,
        arg_func=bmm_args,
        dtypes=FLOAT_DTYPES,
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_mm():
    def mm_args(dtype, batch, size):
        inp1 = torch.randn([size, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([size, size], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="mm",
        torch_op=torch.mm,
        arg_func=mm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_mv():
    def mv_args(dtype, batch, size):
        inp1 = torch.randn([size, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([size], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="mv",
        torch_op=torch.mv,
        arg_func=mv_args,
        dtypes=FLOAT_DTYPES,
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_outer():
    def outer_args(dtype, batch, size):
        inp1 = torch.randn([size], dtype=dtype, device="cuda")
        inp2 = torch.randn([size], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="outer",
        torch_op=torch.outer,
        arg_func=outer_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()
