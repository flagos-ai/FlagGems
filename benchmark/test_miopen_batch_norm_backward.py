import pytest
import torch

import flag_gems

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    # TODO: add new metric

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


def batchnorm_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = None
    running_var = None
    training = True
    momentum = 0.1
    eps = 1e-5
    cudnn_enabled = True
    yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        running_mean = torch.randn((C,), dtype=dtype, device=device)
        running_var = torch.randn((C,), dtype=dtype, device=device)
        yield inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled


@pytest.mark.miopen_batch_norm_backward
def test_miopen_batch_norm_backward():
    def miopen_batch_norm_backward_input_fn(shape, dtype, device):
        for forward_args in batchnorm_input_fn(shape, dtype, device):
            (
                inp,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                _,
                eps,
                _,
            ) = forward_args

            grad_output = torch.randn_like(inp)
            channels = weight.shape[0] if weight is not None else inp.shape[1]

            if running_mean is None:
                running_mean = torch.zeros(channels, dtype=dtype, device=device)
            if running_var is None:
                running_var = torch.ones(channels, dtype=dtype, device=device)

            save_mean = torch.randn(channels, dtype=torch.float32, device=device)
            # MIOpen names this argument save_var, but it stores saved inverse std.
            save_var = (
                torch.abs(torch.randn(channels, dtype=torch.float32, device=device))
                + 0.1
            )

            yield (
                inp,
                grad_output,
                weight,
                running_mean,
                running_var,
                save_mean,
                save_var,
                eps,
            )

    # Create a reference function that uses native_batch_norm_backward
    # since miopen_batch_norm_backward is not available in PyTorch
    def miopen_batch_norm_backward_reference(
        input,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        epsilon,
    ):
        save_invstd = save_var
        # Use output_mask to get all 3 outputs
        output_mask = [True, weight is not None, True]
        train = True
        return torch.ops.aten.native_batch_norm_backward(
            grad_output,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            train,
            epsilon,
            output_mask,
        )

    bench = NormBenchmark(
        input_fn=miopen_batch_norm_backward_input_fn,
        op_name="miopen_batch_norm_backward",
        torch_op=miopen_batch_norm_backward_reference,
        # MThreads supports only float32 here; NVIDIA uses consts.FLOAT_DTYPES.
        dtypes=[torch.float32]
        if base.vendor_name == "mthreads"
        else consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.miopen_batch_norm_backward)
    bench.run()
