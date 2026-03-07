import torch

from .performance_utils import Benchmark


def test_log10_perf():
    def log10_torch(x):
        return torch.log10(x)

    def log10_gems(x):
        return torch.log10(x)

    bench = Benchmark(
        op_name="log10",
        torch_op=log10_torch,
        arg_func=lambda: (torch.randn([1024, 1024], dtype=torch.float32, device="cuda"),),
        dtype_list=[torch.float16, torch.float32, torch.bfloat16],
        shape_list=[(1024, 1024), (2048, 2048), (4096, 4096)],
    )
    bench.run()
