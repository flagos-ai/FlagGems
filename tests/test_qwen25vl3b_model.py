import logging

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.op_1202
def test_softmax_precision():
    torch.manual_seed(42)

    dtype = torch.bfloat16
    with flag_gems.use_gems():
        x = torch.randn(1, 151936, dtype=dtype)
        cpu_result = torch.softmax(x.cpu(), dim=-1)
        cuda_result = torch.softmax(x.cuda(), dim=-1)

    gems_assert_close(cpu_result, cuda_result.cpu(), dtype)

    # gems_assert_close(
    #     res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    # )

    print(f"cpu_result = {cpu_result}")
    print(f"cuda_result = {cuda_result}")

    max_error = (cpu_result - cuda_result.cpu()).abs().max().item()
    print(max_error)


@pytest.mark.op_1203_01
def test_op_1203_01():
    with flag_gems.use_gems():
        x = torch.tensor([1.0, 2.0], dtype=torch.float16, device="cuda")
        x = x.to(torch.bfloat16)
    print(x)


@pytest.mark.op_1203_02
def test_op_1203_02():
    with flag_gems.use_gems():
        gpu_res = torch.tensor([[1, 256, 256]], device="cuda").prod(1)
    cpu_res = torch.tensor([[1, 256, 256]], device="cpu").prod(1)

    gems_assert_close(gpu_res.cpu(), cpu_res.cpu(), gpu_res.dtype)


@pytest.mark.op_1203_03
def test_op_1203_03():
    inputs_embeds = torch.randn(925, 5120, dtype=torch.bfloat16, device="cuda:0")
    is_multimodal = torch.zeros(925, dtype=torch.bool, device="cuda:0")
    is_multimodal[:897] = True
    flattened = torch.randn(897, 5120, dtype=torch.bfloat16, device="cuda:0")

    with flag_gems.use_gems():
        inputs_embeds[is_multimodal] = flattened

    torch.testing.assert_close(inputs_embeds[is_multimodal], flattened)


@pytest.mark.op_1203_04
def test_op_1203_04():
    with flag_gems.use_gems():
        t = torch.ones((1, 0), dtype=torch.int64, device="cuda:0")
        result = torch.ones_like(t)
    print(result)
