import torch

from .performance_utils import GenericBenchmark


class Benchmark(GenericBenchmark):
    """Benchmark for _sobol_engine_scramble_ operator."""

    def set_more_shapes(self):
        """Set benchmark shapes for different dimensions."""
        self.shapes = [
            (100, 30),
            (500, 30),
            (1000, 30),
            (5000, 30),
        ]
        self.bench_fn_args = [
            (
                torch.randint(0, 2, shape, dtype=torch.long, device=self.device),
                torch.randint(
                    0, 2, (shape[0], 30, 30), dtype=torch.long, device=self.device
                ).tril(),
                shape[0],
            )
            for shape in self.shapes
        ]
        self.bench_fn_strs = [f"dimension={shape[0]}" for shape in self.shapes]

    def set_torch_fn(self):
        """Set PyTorch reference function."""

        def torch_fn(sobolstate, ltm, dimension):
            return torch._sobol_engine_scramble_(sobolstate.clone(), ltm, dimension)

        self.torch_fn = torch_fn

    def set_gems_fn(self):
        """Set FlagGems function."""

        def gems_fn(sobolstate, ltm, dimension):
            return torch._sobol_engine_scramble_(sobolstate.clone(), ltm, dimension)

        self.gems_fn = gems_fn

    def run(self):
        """Run the benchmark."""
        self.run_benchmark()
