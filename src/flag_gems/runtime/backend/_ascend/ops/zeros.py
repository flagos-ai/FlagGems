import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device

@triton.autotune(
configs=[
triton.Config({'BLOCK_SIZE': bs, 'BLOCK_SIZE_SUB': bss})
for bs in [ 20480,32768,40960,65536,81920]
for bss in [4096,40960]
if bss <= bs and bs % bss == 0
],
key=['n_elements']
)
@triton.jit
def zeros_kernel(
output_ptr,
n_elements,
BLOCK_SIZE: tl.constexpr,
BLOCK_SIZE_SUB: tl.constexpr,
):
pid = tle.program_id(axis=0)

for sub_block_start_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
    sub_offset = pid * BLOCK_SIZE + sub_block_start_idx + tl.arange(0, BLOCK_SIZE_SUB)
    mask = sub_offset < n_elements
    tl.store(output_ptr + sub_offset, 0.0, mask=mask)

def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
print("GEMS_ASCEND ZEROS")
if dtype is None:
dtype = torch.get_default_dtype()
if device is None:
device = torch.device(device_.name)

out = torch.empty(size, device=device, dtype=dtype)
N = volume(size)
grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
with torch_device_fn.device(device):
    zeros_kernel[grid_fn](out, N)
return out