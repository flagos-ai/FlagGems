import pytest
import torch

from . import base


def set_default_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp,


def set_source_tensor_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    source = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"source": source}


def set_source_tensor_storage_offset_input_fn(shape, dtype, device):
    source = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.empty(1, dtype=dtype, device=device)
    size = list(source.size())
    stride = list(source.stride())
    offset = source.storage_offset()
    yield inp, {
        "source": source,
        "storage_offset": offset,
        "size": size,
        "stride": stride,
    }


@pytest.mark.set_
def test_set_default():
    bench = base.GenericBenchmark(
        input_fn=set_default_input_fn,
        op_name="set_",
        torch_op=torch.Tensor.set_,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()


@pytest.mark.set_source_Tensor
def test_set_source_tensor():
    bench = base.GenericBenchmark(
        input_fn=set_source_tensor_input_fn,
        op_name="set_source_Tensor",
        torch_op=torch.Tensor.set_,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()


@pytest.mark.set_source_Tensor_storage_offset
def test_set_source_tensor_storage_offset():
    bench = base.GenericBenchmark(
        input_fn=set_source_tensor_storage_offset_input_fn,
        op_name="set_source_Tensor_storage_offset",
        torch_op=torch.Tensor.set_,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()
