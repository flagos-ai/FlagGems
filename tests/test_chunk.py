import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.chunk
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (4096, 4096), (64, 512, 512)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunks", [2, 3, 7])
@pytest.mark.parametrize("dim", [0, -1])
def test_chunk(shape, dtype, chunks, dim):
    """Test chunk accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.chunk(ref_inp, chunks, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.chunk(inp, chunks, dim=dim)

    assert len(res_out) == len(
        ref_out
    ), f"chunk count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)


@pytest.mark.chunk
@pytest.mark.parametrize("dim,chunks", [(-1, 2), (0, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunk_backward_dispatch(dim, chunks, dtype):
    """Device registration must retain the differentiable decomposition."""
    inp = torch.randn((6, 8), device=flag_gems.device, dtype=dtype, requires_grad=True)
    reference = inp.detach().clone().requires_grad_(True)
    ref_parts = torch.chunk(reference, chunks, dim=dim)
    sum((i + 1) * part.float().sum() for i, part in enumerate(ref_parts)).backward()
    # Exercise the registry: calling flag_gems.chunk directly misses a broken
    # Autograd dispatch entry even though the Python decomposition is correct.
    library = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=library, include=["chunk"])
        parts = torch.chunk(inp, chunks, dim=dim)
        sum((i + 1) * part.float().sum() for i, part in enumerate(parts)).backward()
    finally:
        library._destroy()
        flag_gems.current_work_registrar = previous_registrar
    torch.testing.assert_close(inp.grad, reference.grad)
