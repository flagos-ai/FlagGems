import torch

from .performance_utils import Benchmark


def test_gcd_perf():
    def gcd_torch(x, y):
        return torch.gcd(x, y)

    def gcd_gems(x, y):
        return torch.gcd(x, y)

    bench = Benchmark(
        op_name="gcd",
        torch_op=gcd_torch,
        arg_func=lambda: (
            torch.randint(-1000, 1000, [1024, 1024], dtype=torch.int32, device="cuda"),
            torch.randint(-1000, 1000, [1024, 1024], dtype=torch.int32, device="cuda"),
        ),
        dtype_list=[torch.int32, torch.int64],
        shape_list=[(1024, 1024), (2048, 2048), (4096, 4096)],
    )
    bench.run()

