import pytest
import torch

import flag_gems

from . import base, consts


class VanderBenchmark(base.Benchmark):
    """
    Benchmark for vander operation.
    vander takes a 1D tensor and generates a 2D Vandermonde matrix.
    """

    def set_shapes(self, shape_file_path=None):
        # vander takes a 1-D input of length M and produces an M x M matrix, so
        # the shared core_shapes.yaml / DEFAULT_SHAPES (which include a
        # 1073741824-element 1-D tensor and multi-dim shapes) would OOM or be
        # invalid inputs. Override with 1-D lengths whose M x M output stays
        # within memory. Rule 14: override set_shapes so CI cannot clobber these.
        self.shapes = [(64,), (128,), (256,), (512,)]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from self._inputs_for_shape(shape, dtype)

    def _inputs_for_shape(self, shape, dtype):
        inp = base.generate_tensor_input(shape, dtype, flag_gems.device)
        yield inp, None, False  # (x, N=None, increasing=False)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        return next(self._inputs_for_shape(shape, case.dtype))

    def unpack_to_args_kwargs(self, input):
        x, N, increasing = input
        return (x,), {"N": N, "increasing": increasing}

    def record_shapes(self, x, N=None, increasing=False):
        M = x.shape[0]
        N_val = N if N is not None else M
        return f"[{M} -> {M}x{N_val}]"


@pytest.mark.vander
def test_vander():
    bench = VanderBenchmark(
        op_name="vander", torch_op=torch.vander, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
