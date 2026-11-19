import pytest
import torch

import flag_gems

from . import base
from .consts import BenchmarkCasePlan

# PDIST requires the input dim to be reasonably small; these shapes follow the upstream test suite.
PDIST_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]

# The backward defaults to the same matrices as the forward benchmark: pdist
# has no entry in core_shapes.yaml, so PdistBenchmark.set_shapes is the
# authoritative default shape list and the backward case reuses it unchanged.
PDIST_BACKWARD_SHAPES = PDIST_SHAPES


class PdistBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PDIST_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (x, 2.0)


@pytest.mark.pdist
def test_pdist():
    bench = PdistBenchmark(
        op_name="pdist",
        torch_op=torch.pdist,
        # pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
        dtypes=[torch.float32],
    )
    bench.run()


class PdistBackwardBenchmark(base.Benchmark):
    """Timed pdist backward through the real autograd path.

    ``is_backward=True`` builds the forward graph once in ``get_latency`` and
    times only ``torch.autograd.grad(out, (x,), (dout,), retain_graph=True)``,
    so input construction and the forward launch are outside the measurement.
    One case per ``PDIST_BACKWARD_SHAPES`` entry: the forward input and ``p``
    are the plain arguments, and the condensed gradient length
    ``N*(N-1)//2`` is derived inside ``get_latency`` from the forward output.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = PDIST_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (x, 2.0)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                BenchmarkCasePlan(shape={"input": list(shape)}, params={"p": 2.0}),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].shape["input"]
        return (
            torch.randn(shape, dtype=case.dtype, device=self.device),
            case.builder_args[0].params["p"],
        )


@pytest.mark.pdist_forward_backward
def test_pdist_backward():
    bench = PdistBackwardBenchmark(
        op_name="pdist",
        torch_op=torch.pdist,
        # pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
        dtypes=[torch.float32],
        is_backward=True,
    )
    bench.set_gems(flag_gems.pdist)
    bench.run()
