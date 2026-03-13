import torch

from .performance_utils import GenericBenchmark, generate_tensor_input


class Log10Benchmark(GenericBenchmark):
    input_generator = generate_tensor_input

    def set_more_shapes(self):
        return [
            (1024,),
            (1024 * 1024,),
            (4096, 4096),
            (128, 512, 512),
        ]

    def set_more_metrics(self):
        return []


bench_log10 = Log10Benchmark(
    op_name="log10",
    torch_op=torch.log10,
    dtypes=[torch.float16, torch.float32, torch.bfloat16],
)

if __name__ == "__main__":
    bench_log10.run(print_data=True)
