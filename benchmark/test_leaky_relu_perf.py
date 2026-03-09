import torch

from .performance_utils import GenericBenchmark, generate_tensor_input


class LeakyReluBenchmark(GenericBenchmark):
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


bench_leaky_relu = LeakyReluBenchmark(
    op_name="leaky_relu",
    torch_op=torch.nn.functional.leaky_relu,
    dtypes=[torch.float16, torch.float32, torch.bfloat16],
)

if __name__ == "__main__":
    bench_leaky_relu.run(print_data=True)
