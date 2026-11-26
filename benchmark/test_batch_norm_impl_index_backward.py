import pytest
import torch

import flag_gems

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


def _batch_norm_impl_index_backward_input_fn(shape, dtype, device):
    """Yield argument tuples matching the ``_batch_norm_impl_index_backward``
    schema: ``(impl_index, input, grad_output, weight, running_mean,
    running_var, save_mean, save_var_transform, train, eps, output_mask,
    reservedSpace)``.

    The native backend (``impl_index == 0``) is exercised here because it
    supports every floating dtype, mirroring the real CUDA dispatcher's
    ``!train``-or-``impl_index == 0`` fallback path.
    """
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn_like(inp)
    weight = torch.randn((C,), dtype=dtype, device=device)
    running_mean = torch.zeros((C,), dtype=dtype, device=device)
    running_var = torch.ones((C,), dtype=dtype, device=device)
    # The saved statistics are always stored in the accumulate type (float32),
    # which is exactly what the native forward produces and what the
    # backward expects.
    save_mean = torch.randn((C,), dtype=torch.float32, device=device)
    # The save_var slot holds the inverse standard deviation.
    save_var = torch.abs(torch.randn((C,), dtype=torch.float32, device=device)) + 0.1
    impl_index = 0
    train = True
    eps = 1e-5
    output_mask = [True, True, True]
    reserved_space = torch.empty(0, dtype=torch.uint8, device=device)
    yield (
        impl_index,
        inp,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        train,
        eps,
        output_mask,
        reserved_space,
    )


@pytest.mark.batch_norm_impl_index_backward
def test_batch_norm_impl_index_backward():
    bench = NormBenchmark(
        input_fn=_batch_norm_impl_index_backward_input_fn,
        op_name="batch_norm_impl_index_backward",
        torch_op=torch.ops.aten._batch_norm_impl_index_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._batch_norm_impl_index_backward)
    bench.run()
