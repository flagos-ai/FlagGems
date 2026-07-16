import pytest
import torch
import flag_gems
from . import base, consts

def _ascend_conj_ref(input):
    """Ascend reference implementation using small ops combination."""
    if not input.is_complex():
        return input
    return torch.complex(input.real, -input.imag)

@pytest.mark.conj
def test_conj():
    device = flag_gems.device
    if "npu" in str(device) or "ascend" in str(device).lower():
        # Ascend: use small ops combination as reference
        torch_op = _ascend_conj_ref
    else:
        # NVIDIA: use native torch.conj
        torch_op = torch.conj

    bench = base.UnaryPointwiseBenchmark(
        op_name="conj",
        torch_op=torch_op,
        dtypes=consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conj)
    bench.run()
