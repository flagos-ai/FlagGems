"""
Compare CPU vs CUDA median behavior
"""
import torch

print("="*80)
print("Compare CPU vs CUDA median behavior with duplicate values")
print("="*80)

# Find a case with duplicate median values
for attempt in range(1000):
    torch.manual_seed(attempt)
    data_f32 = torch.randn((1024,), dtype=torch.float32)
    data_bf16 = data_f32.to(torch.bfloat16)

    k = (1024 + 1) // 2  # 513

    # CPU
    data_bf16_cpu = data_bf16.cpu()
    data_f64_cpu = data_f32.to(torch.float64).cpu()

    kth_val_cpu, kth_idx_cpu = torch.kthvalue(data_f64_cpu, k)
    mask_cpu = data_f64_cpu == kth_val_cpu
    positions_cpu = torch.where(mask_cpu)[0]

    if len(positions_cpu) >= 2:
        median_val_cpu, median_idx_cpu = torch.median(data_f64_cpu, dim=0)

        # CUDA
        data_bf16_cuda = data_bf16.cuda()
        data_f64_cuda = data_f32.to(torch.float64).cuda()

        kth_val_cuda, kth_idx_cuda = torch.kthvalue(data_f64_cuda, k)
        median_val_cuda, median_idx_cuda = torch.median(data_f64_cuda, dim=0)

        print(f"\nAttempt {attempt}: Found duplicate median values")
        print(f"Median value {kth_val_cpu.item()} appears at positions (CPU): {positions_cpu.tolist()}")
        print(f"  First: {positions_cpu[0].item()}, Last: {positions_cpu[-1].item()}")

        print(f"\nCPU:")
        print(f"  kthvalue index: {kth_idx_cpu.item()}")
        print(f"  median index: {median_idx_cpu.item()}")

        print(f"\nCUDA:")
        print(f"  kthvalue index: {kth_idx_cuda.item()}")
        print(f"  median index: {median_idx_cuda.item()}")

        print(f"\nCPU kthvalue selects: {('first' if kth_idx_cpu.item() == positions_cpu[0].item() else 'last' if kth_idx_cpu.item() == positions_cpu[-1].item() else 'middle')}")
        print(f"CPU median selects: {('first' if median_idx_cpu.item() == positions_cpu[0].item() else 'last' if median_idx_cpu.item() == positions_cpu[-1].item() else 'middle')}")
        print(f"CUDA kthvalue selects: {('first' if kth_idx_cuda.item() == positions_cpu[0].item() else 'last' if kth_idx_cuda.item() == positions_cpu[-1].item() else 'middle')}")
        print(f"CUDA median selects: {('first' if median_idx_cuda.item() == positions_cpu[0].item() else 'last' if median_idx_cuda.item() == positions_cpu[-1].item() else 'middle')}")

        break

print("\n" + "="*80)
print("Analysis complete")
print("="*80)
