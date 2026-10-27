import pytest
import torch

from . import base, consts

DEFAULT_SHAPE_DESC = "input shape"


# squeeze_ is a zero-copy metadata op, so the timed region never moves element
# data and the workload is dispatch + shape/stride bookkeeping. The shapes stay
# small (no 1e9-element grids) and each carries size-1 dims so the squeeze
# actually removes something; (1024, 1024) is the no-op control case. ATen
# spells the overloads squeeze_.dim / squeeze_.dims; pytest marks cannot
# contain a dot, so the dot-free ids squeeze__dim / squeeze__dims stand in for
# them (same convention as all.dim -> all_dim).
SQUEEZE_SHAPES = [
    (1,),  # all size-1 dims -> 0-d
    (1, 1024, 1),  # leading and trailing size-1 dims
    (1024, 1, 1024, 1),  # interleaved size-1 dims
    (1, 1, 1, 1, 1),  # every dim removed -> 0-d
    (1, 20, 320, 1, 15),  # 5-D with interior singleton
    (1, 16, 128, 1, 64),  # 5-D model-like feature map
    (32, 1, 512),  # batch x singleton x hidden
    (1024, 1024),  # no size-1 dim: no-op control
]


class SqueezeBenchmark(base.Benchmark):
    """Benchmark for the in-place squeeze family (zero-copy view operation)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SQUEEZE_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, dtype, self.device)
            yield inp,


class SqueezeDimBenchmark(SqueezeBenchmark):
    """Squeeze one dimension.

    dim=0 is in range for every shape above and stays in range after squeezing
    (a leading size-1 dim is the only thing that can disappear, and a 0-d
    tensor still accepts dim=0), which matters because the harness replays one
    input tuple for the whole timed region and the op mutates that tuple.
    """

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, dtype, self.device)
            yield inp, 0


class SqueezeDimsBenchmark(SqueezeBenchmark):
    """Squeeze the leading size-1 dimension of each shape.

    The harness replays the same (tensor, dims) tuple throughout the timed
    region and squeeze_ mutates the tensor in place, so the list must stay in
    range after the first squeeze. Listing only the *first* size-1 dimension
    keeps every list below in range: it becomes a no-op once removed, while
    (1, 1, 1, 1, 1) still collapses one dimension per call.
    """

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, dtype, self.device)
            dims = [index for index, size in enumerate(shape) if size == 1][:1]
            yield inp, dims


@pytest.mark.squeeze_
def test_squeeze_():
    bench = SqueezeBenchmark(
        op_name="squeeze_",
        torch_op=torch.ops.aten.squeeze_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.squeeze__dim
def test_squeeze__dim():
    bench = SqueezeDimBenchmark(
        op_name="squeeze__dim",
        torch_op=torch.ops.aten.squeeze_.dim,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.squeeze__dims
def test_squeeze__dims():
    bench = SqueezeDimsBenchmark(
        op_name="squeeze__dims",
        torch_op=torch.ops.aten.squeeze_.dims,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
