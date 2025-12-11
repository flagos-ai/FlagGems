import triton
import triton.language as tl
import torch
import torch.profiler

@triton.jit
def bit_reverse_kernel(
    real_in, imag_in,
    real_out, imag_out,
    n
):
    """先进行逆序排列: input[i] -> output[bit_reverse(i)]"""
    tid = tl.program_id(0)
    if tid >= n:
        return
    
    # 计算位数
    temp_n = n
    bit_count = 0
    while temp_n > 1:
        temp_n //= 2
        bit_count += 1
    
    # 计算逆序
    idx = tid
    rev_idx = 0
    temp_idx = idx
    for i in range(bit_count):
        rev_idx = (rev_idx << 1) | (temp_idx & 1)
        temp_idx = temp_idx >> 1
    
    # input[idx] -> output[rev_idx]
    val_real = tl.load(real_in + idx)
    val_imag = tl.load(imag_in + idx)
    tl.store(real_out + rev_idx, val_real)
    tl.store(imag_out + rev_idx, val_imag)

@triton.jit
def fft_stage_kernel(
    real_ptr, imag_ptr,
    n, stage
):
    """迭代 FFT 阶段"""
    pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286  # 圆周率
    tid = tl.program_id(0)
    
    if tid >= n // 2:
        return

    # 计算当前阶段的参数
    block_size = 1 << stage  # 2^stage
    half_block = block_size // 2
    
    # 每个线程处理一个蝶形对
    butterfly_group = tid // half_block  # 属于第几个大块
    pos_in_group = tid % half_block     # 在块前半部分的位置
    
    # 计算蝶形对中两个元素的索引
    first_idx = butterfly_group * block_size + pos_in_group
    second_idx = first_idx + half_block
    
    if second_idx >= n:
        return
    
    # load
    a_real = tl.load(real_ptr + first_idx)
    a_imag = tl.load(imag_ptr + first_idx)
    b_real = tl.load(real_ptr + second_idx)
    b_imag = tl.load(imag_ptr + second_idx)
    
    # 计算复数幅度
    angle = -2.0 * pi * pos_in_group / block_size
    w_real = tl.cos(angle)
    w_imag = tl.sin(angle)
    
    # 使用幅度
    tw_real = b_real * w_real - b_imag * w_imag
    tw_imag = b_real * w_imag + b_imag * w_real
    
    # 蝶形运算
    result_a_real = a_real + tw_real
    result_a_imag = a_imag + tw_imag
    result_b_real = a_real - tw_real
    result_b_imag = a_imag - tw_imag
    
    # store
    tl.store(real_ptr + first_idx, result_a_real)
    tl.store(imag_ptr + first_idx, result_a_imag)
    tl.store(real_ptr + second_idx, result_b_real)
    tl.store(imag_ptr + second_idx, result_b_imag)

def fft_1d(x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    N = x.shape[0]
    assert (N > 0 and (N & (N-1)) == 0)  # 确保N是2的整数幂
    
    x_real = x.real.clone()
    x_imag = x.imag.clone()
    
    # 位逆序
    temp_real = torch.zeros_like(x_real)
    temp_imag = torch.zeros_like(x_imag)
    bit_reverse_kernel[(N,)](x_real, x_imag, temp_real, temp_imag, N)
    
    x_real.copy_(temp_real)
    x_imag.copy_(temp_imag)
    
    log2n = N.bit_length() - 1
    for stage in range(1, log2n + 1):
        fft_stage_kernel[(N // 2,)](x_real, x_imag, N, stage)
    
    output.real.copy_(x_real)
    output.imag.copy_(x_imag)
    return output

# def torch_fft(x):
#     return torch.fft.fft(x)

# def profile_torch_fft():
#     """使用 torch.profiler 分析 torch.fft.fft()"""
#     torch.manual_seed(0)
#     N = 1024  # 使用较大的N以便观察
#     x = torch.arange(N, device='cuda') + torch.arange(N, device='cuda') * 1j
    
#     print("Analyzing torch.fft.fft() with torch.profiler...")
    
#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fft_profile'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#     ) as prof:
#         for i in range(5):  # 运行5次（1次wait + 1次warmup + 3次active）
#             if i < 2:  # 前两次用于warmup
#                 y = torch_fft(x)
#             else:  # 后三次进行分析
#                 y = torch_fft(x)
#             prof.step()  # 标记一个步骤
    
#     # 打印性能分析结果
#     print("\nPerformance Analysis Results:")
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
#     # 打印内存使用情况
#     print("\nMemory Usage Analysis:")
#     print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# def main(N):
#     torch.manual_seed(0)
#     print("-" * 50)
#     print(f"N = {N}")

#     # 调用
#     x = torch.arange(N, device='cuda') + torch.arange(N, device='cuda') * 1j

#     # warm-up
#     for _ in range(5):
#         y_triton = fft_1d(x, N)
#         y_torch = torch_fft(x)
#     torch.cuda.synchronize()
#     # Timekeeping
#     times_triton = []
#     times_torch = []
#     diffs = []
#     for _ in range(10):
#         start_triton = torch.cuda.Event(enable_timing=True)
#         end_triton   = torch.cuda.Event(enable_timing=True)
#         start_triton.record()
#         y_triton = fft_1d(x, N)
#         end_triton.record()
#         torch.cuda.synchronize()
#         times_triton.append(start_triton.elapsed_time(end_triton))

#         start_torch = torch.cuda.Event(enable_timing=True)
#         end_torch   = torch.cuda.Event(enable_timing=True)
#         start_torch.record()
#         y_torch = torch_fft(x)
#         end_torch.record()
#         torch.cuda.synchronize()
#         times_torch.append(start_torch.elapsed_time(end_torch))
        
#         diffs.append(torch.sum(torch.abs(y_triton - y_torch) / torch.abs(y_torch)).item() / N)

    
#     # debug
#     print_output = 0  # 是否打印输出tensor 
#     if print_output:
#         print("input x:")
#         print(x)
#         print("\n\nTriton FFT output y:")
#         print(y_triton)
#         print("\n\nTorch FFT output y:")
#         print(y_torch)

#     time_triton = sum(times_triton) / len(times_triton)
#     time_torch = sum(times_torch) / len(times_torch)
#     av_diff = sum(diffs) / len(diffs)

#     print(f"fft_1d time: {time_triton:.6f} s")
#     print(f"torch_fft  time: {time_torch:.6f} s")
#     print("average diff:", av_diff)
#     print(f"Test {'PASSED' if av_diff < 1e-5 else 'FAILED'}")
#     print("-" * 50)

# if __name__ == "__main__":
#     N_list = [4, 8, 16, 32, 64, 128, 256, 512]
#     # N_list = [4,]
#     for N in N_list:
#         main(N)

#     # profile_torch_fft()