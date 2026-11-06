import pytest
import torch

import flag_gems

from . import base, consts


def _dims_case_fn(shape, dtype):
    del dtype
    # The list form: yield the plan only for shapes owning at least two
    # size-1 dims, since a single entry is identical work to the int form.
    singletons = [dim for dim, size in enumerate(shape) if size == 1]
    if len(singletons) > 1:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim": singletons},
            builder_args=(shape, singletons),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    # Metadata-only op: the values are never read, so an uninitialized buffer
    # avoids paying for a fill the benchmark would not measure.
    inp = torch.empty(shape, dtype=dtype, device=device)
    if dim is None:
        return (inp,)
    return inp, dim


def _case_fn(shape, dtype):
    del dtype
    # squeeze(x) drops every size-1 dim for all shapes; the .dim and .dims
    # forms are added only for shapes that own a size-1 dim, since a shape
    # without one only ever exercises the no-op path.
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dim": None},
        builder_args=(shape, None),
    )
    singletons = [dim for dim, size in enumerate(shape) if size == 1]
    if singletons:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim": singletons[0]},
            builder_args=(shape, singletons[0]),
        )
    if len(singletons) > 1:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim": singletons},
            builder_args=(shape, singletons),
        )


# squeeze only relabels dims: the measured work is dispatch plus view
# construction, and the input values are never read. A size-1 extent is the
# only thing the operator reacts to and the shared core/comprehensive grids
# contain none, so the size-1 boundaries are appended to the shared grid
# instead of replacing it. The large rank-5 entries keep the metadata cost from
# being hidden by timer noise.
_SQUEEZE_BOUNDARY_SHAPES = [
    (1,),  # all size-1 dims -> 0-d
    (1, 256),  # leading size-1 dim
    (256, 1),  # trailing size-1 dim
    (1, 1, 1),  # several size-1 dims -> 0-d
    (2, 1, 3),  # interleaved size-1 dim
    (1, 1024, 1024),  # leading size-1 dim, large payload
    (20, 1, 320, 15),  # interleaved size-1 dim, large payload
    (16, 1, 128, 64, 60),  # rank-5 with a size-1 dim
    (16, 7, 57, 1, 29),  # rank-5 with a trailing size-1 dim
    (1, 7, 1, 32, 1),  # rank-5, interleaved size-1 dims
]


class SqueezeBenchmark(base.GenericBenchmark):
    """Shared core/comprehensive shape grids plus the squeeze boundaries."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape) for shape in self.shapes + _SQUEEZE_BOUNDARY_SHAPES
            )
        )


@pytest.mark.squeeze
def test_squeeze():
    bench = SqueezeBenchmark(
        op_name="squeeze",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.squeeze,
        gems_op=flag_gems.squeeze,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# ``squeeze.dims`` contains a dot, so the literal @pytest.mark.squeeze.dims
# would chain off the ``squeeze`` MarkDecorator (which has no ``__getattr__``)
# and raise AttributeError; build the overload mark via getattr instead, the
# same way tests/test_or_scalar.py does for the ``or`` operator.


@getattr(pytest.mark, "squeeze.dims")
@pytest.mark.squeeze
def test_squeeze_dims_list():
    # The list form is measured on the same grids; _case_fn only yields the
    # list form for shapes that own at least two size-1 dims, since fewer
    # entries degenerate to the int form (one entry) or the no-op (none).
    # op_name stays "squeeze": both call forms are the same registered
    # operator, and the shape grids are looked up under that id.
    bench = SqueezeBenchmark(
        op_name="squeeze",
        case_fn=_dims_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.squeeze.dims,
        gems_op=flag_gems.squeeze,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# ``squeeze.dims`` contains a dot, so the literal @pytest.mark.squeeze.dims
# would chain off the ``squeeze`` MarkDecorator (which has no ``__getattr__``)
# and raise AttributeError; build the overload mark via getattr instead, the
# same way tests/test_or_scalar.py does for the ``or`` operator.
