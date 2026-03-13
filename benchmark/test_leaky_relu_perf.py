import torch

from .performance_utils import Benchmark


def test_leaky_relu_perf():
    def leaky_relu_torch(x):
        return torch.nn.functional.leaky_relu(x, negative_slope=0.01)

    def leaky_relu_gems(x):
        return torch.nn.functional.leaky_relu(x, negative_slope=0.01)

    bench = Benchmark(
        op_name="leaky_relu",
        torch_op=leaky_relu_torch,
        arg_func=lambda: (
            torch.randn([1024, 1024], dtype=torch.float32, device="cuda"),
        ),
        dtype_list=[torch.float16, torch.float32, torch.bfloat16],
        shape_list=[(1024, 1024), (2048, 2048), (4096, 4096)],
    )
    bench.run()
