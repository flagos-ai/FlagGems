import pytest
import torch

import flag_gems
import gc

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)


@pytest.mark.add
#@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (2048, 1024, 1024), (4096, 1024, 1024), (8192, 1024, 1024)])
#@pytest.mark.parametrize("alpha", [0.001, -0.999, 100.001, -111.999])
#@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1024, 1024, 1024)])
@pytest.mark.parametrize("alpha", [0.001, -0.999, 100.001, -111.999])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_add(shape, alpha, dtype):
    #import pdb; pdb.set_trace()
    inp1_ = torch.randn(shape, dtype=dtype)
    inp1 = inp1_.to(flag_gems.device)
    #pdb.set_trace()
    inp2_ = torch.randn(shape, dtype=dtype)
    inp2 = inp2_.to(flag_gems.device)
    #pdb.set_trace()
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    #pdb.set_trace()
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    print('=====inp1_ ptr   :', hex(inp1_.data_ptr()))
    print('=====ref_inp1 ptr:', hex(ref_inp1.data_ptr()))
    print('=====inp2_ ptr   :', hex(inp2_.data_ptr()))
    print('=====ref_inp2 ptr:', hex(ref_inp2.data_ptr()))
    print('=====ref_out ptr :', hex(ref_out.data_ptr()))
    print('=====inp1 ptr    :', hex(inp1.data_ptr()))
    print('=====inp2 ptr    :', hex(inp2.data_ptr()))
    print('=====res_out ptr :', hex(res_out.data_ptr()))

    #pdb.set_trace()
    gems_assert_close(res_out, ref_out, dtype)
    del inp1_
    del inp2_
    del inp1
    del inp2
    del res_out
    #gc.collect()
    import pdb;pdb.set_trace()
    del ref_out

