import torch
import torch.nn.functional as F
import flag_gems
from torch.utils.benchmark import Timer

def benchmark_operator(shape, kernel_size, stride, padding):
    input_tensor = torch.randn(shape, dtype=torch.float32, device='cuda')
    
    if stride is None:
        stride = kernel_size
        
    def native():
        return F.avg_pool3d(input_tensor, kernel_size, stride, padding)
    
    def flaggems():
        return flag_gems.avg_pool3d(input_tensor, kernel_size, stride, padding)
    
    for _ in range(10):
        native()
        flaggems()
    
    t1 = Timer("native()", globals={"native": native})
    native_time = t1.blocked_autorange().median
    
    t2 = Timer("flaggems()", globals={"flaggems": flaggems})
    flaggems_time = t2.blocked_autorange().median
    
    speedup = native_time / flaggems_time
    print(f"Shape {shape}: Speedup={speedup:.3f}x")
    return speedup

if __name__ == "__main__":
    benchmark_operator((2, 4, 32, 32, 32), (3, 3, 3), (2, 2, 2), (1, 1, 1))