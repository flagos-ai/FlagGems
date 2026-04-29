import torch
import pytest
import triton

def native_pytorch_chunk_gated_delta(q, k, v, beta):
    """
    ይህ የ PyTorch መደበኛ ስሌት (Sequential) ነው። በጣም አዝጋሚ ቢሆንም፣
    የሂሳብ ትክክለኛነቱ (Mathematical Correctness) 100% አስተማማኝ ስለሆነ
    የኛን የ Triton ኮድ ትክክለኛነት የምንፈትሸው ከዚህ ጋር በማወዳደር ነው።
    """
    BATCH, HEADS, SEQ_LEN, DIM = q.shape
    out = torch.zeros_like(q)
    
    state = torch.zeros((BATCH, HEADS, DIM, DIM), device=q.device, dtype=torch.float32)
    
    for i in range(SEQ_LEN):
        q_i = q[:, :, i, :].to(torch.float32)
        k_i = k[:, :, i, :].to(torch.float32)
        v_i = v[:, :, i, :].to(torch.float32)
        beta_i = beta[:, :, i].to(torch.float32)
        
        state = state * beta_i[:, :, None, None] + torch.matmul(k_i.unsqueeze(-1), v_i.unsqueeze(-2))
        
        o_i = torch.matmul(q_i.unsqueeze(-2), state).squeeze(-2)
        out[:, :, i, :] = o_i.to(q.dtype)
        
    return out

BATCH_SIZES = [1, 4]
HEADS_SIZES = [2, 8]
SEQ_LENGTHS = [128, 1024, 4096] # 4096 Large size boundary
DIM_SIZES = [64, 128]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("H", HEADS_SIZES)
@pytest.mark.parametrize("S", SEQ_LENGTHS)
@pytest.mark.parametrize("D", DIM_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_chunk_gated_delta_correctness(B, H, S, D, dtype):
    torch.manual_seed(42)
    device = 'cuda'

    q = torch.randn(B, H, S, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(B, H, S, dtype=dtype, device=device, requires_grad=True) # Decay values [0, 1]
    
    do = torch.randn(B, H, S, D, dtype=dtype, device=device)

    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    beta_ref = beta.clone().detach().requires_grad_(True)
    out_ref = native_pytorch_chunk_gated_delta(q_ref, k_ref, v_ref, beta_ref)
    out_ref.backward(do)

    flagos_module = FlagOS_ChunkGatedDelta()
    out_our = flagos_module(q, k, v, beta)
    out_our.backward(do)
    if dtype == torch.float32:
        atol, rtol = 1.30e-6, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 1.00e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 0.016, 0.016
    else:
        atol, rtol = 1e-3, 1e-3

    torch.testing.assert_close(out_our, out_ref, atol=atol, rtol=rtol, msg="Forward Pass Output Mismatch!")
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol, msg="Backward Pass (dQ) Mismatch!")
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol, msg="Backward Pass (dK) Mismatch!")
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol, msg="Backward Pass (dV) Mismatch!")
    torch.testing.assert_close(beta.grad, beta_ref.grad, atol=atol, rtol=rtol, msg="Backward Pass (dBeta) Mismatch!")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQ_LEN'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['pytorch', 'flagos_triton'],
        line_names=['PyTorch Native (For-Loop)', 'FlagOS Optimized (Triton)'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Execution Time (ms)',
        plot_name='Chunk Gated Delta Rule Extreme Optimization Benchmark',
        args={'BATCH': 2, 'HEADS': 8, 'DIM': 128},
    )
)
def benchmark_chunk_gated_delta(BATCH, HEADS, SEQ_LEN, DIM, provider):
    device = 'cuda'
    dtype = torch.float16

    q = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, dtype=dtype, device=device, requires_grad=True)
    beta = torch.rand(BATCH, HEADS, SEQ_LEN, dtype=dtype, device=device, requires_grad=True)
    do = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, dtype=dtype, device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'pytorch':
        def y_fwd():
            return native_pytorch_chunk_gated_delta(q, k, v, beta)
            
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y_fwd().backward(do, retain_graph=True), quantiles=quantiles
        )

    if provider == 'flagos_triton':
        flagos_module = FlagOS_ChunkGatedDelta()
        def y_fwd_triton():
            return flagos_module(q, k, v, beta)
            
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y_fwd_triton().backward(do, retain_graph=True), quantiles=quantiles
        )

    return ms, min_ms, max_ms

if __name__ == "__main__":
    print("🧪 1. የሙከራ እና ትክክለኛነት (Testing) ሂደቱ ተጀምሯል...")
    test_chunk_gated_delta_correctness(B=2, H=4, S=1024, D=128, dtype=torch.float16)
    print("✅ Functional Correctness Test Passed Perfectly!")
    
    print("\n🚀 2. የማፋጠን (Benchmarking) ሂደቱ ተጀምሯል...")
    benchmark_chunk_gated_delta.run(show_plots=True, print_data=True)
    print("\n🎉 ሁሉም ስራዎች በተሳካ ሁኔታ ተጠናቀዋል! ኮዱ ለ FlagGems GitHub Submission 100% ዝግጁ ነው! 🚀")