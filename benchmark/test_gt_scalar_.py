import pytest

from . import base, consts, utils


@pytest.mark.gt_scalar_
def test_gt_scalar_():
    def _scalar_inplace_fn(shape, dtype, device):
        inp = utils.generate_tensor_input(shape, dtype, device)
        yield inp, 0.5

    bench = base.GenericBenchmark(
        input_fn=_scalar_inplace_fn,
        op_name="gt_scalar_",
        torch_op=lambda a, b: a.gt_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
