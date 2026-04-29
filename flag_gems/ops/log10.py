import triton
import triton.language as tl
from ..utils import libentry

# الـ Kernel الذي يعمل على الـ GPU
@libentry()
@triton.jit
def log10_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # قراءة البيانات
    x = tl.load(in_ptr + offsets, mask=mask)
    
    # العملية الحسابية: log10(x) = log2(x) / log2(10)
    # Triton يدعم log2 مباشرة
    output = tl.log2(x) * 0.30102999566  # (1 / log2(10))
    
    # تخزين النتائج
    tl.store(out_ptr + offsets, output, mask=mask)
