
import sys
import os

sys.path.append(os.path.abspath("src"))

import torch

from flag_gems.ops.log10 import log10


def test_small_tensor():

    x = torch.rand(128, device="cuda") + 0.01

    y_triton = log10(x)

    y_torch = torch.log10(x)

    assert torch.allclose(
        y_triton,
        y_torch,
        rtol=1e-4,
        atol=1e-4
    )


def test_large_tensor():

    x = torch.rand(1_000_000, device="cuda") + 0.01

    y_triton = log10(x)

    y_torch = torch.log10(x)

    assert torch.allclose(
        y_triton,
        y_torch,
        rtol=1e-4,
        atol=1e-4
    )


def test_zero_values():

    x = torch.tensor(
        [0.0, 1.0, 10.0],
        device="cuda"
    )

    y_triton = log10(x)

    y_torch = torch.log10(torch.maximum(
        x,
        torch.tensor(1e-12, device="cuda")
    ))

    assert torch.allclose(
        y_triton,
        y_torch,
        rtol=1e-4,
        atol=1e-4
    )


def test_many_shapes():

    shapes = [
        (1,),
        (8,),
        (128,),
        (1024,),
        (4096,),
        (64, 64),
        (128, 128),
        (256, 256),
    ]

    for shape in shapes:

        x = torch.rand(shape, device="cuda") + 0.01

        y_triton = log10(x)

        y_torch = torch.log10(x)

        assert torch.allclose(
            y_triton,
            y_torch,
            rtol=1e-4,
            atol=1e-4
        )


def test_float16():

    x = (
        torch.rand(4096, device="cuda")
        .half()
        + 0.01
    )

    y_triton = log10(x)

    y_torch = torch.log10(x)

    assert torch.allclose(
        y_triton,
        y_torch,
        rtol=1e-2,
        atol=1e-2
    )
