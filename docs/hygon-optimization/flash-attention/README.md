# Hygon `_scaled_dot_product_flash_attention` optimization

The isolated change targets official master and provides guarded Hygon forward paths for the four declared BF16 metadata workloads, preserving output and LSE semantics.

Independent hardware verification on2026-11-12 used master `9a9aecb53103cc83dda27bc7458fc3bfd07f3188` and candidate `efd32566fb2fd258f23d6dea469d39088a929381`. All four cases pass output/LSE correctness and stability. Old/new speedups are **2.6810x,1.6705x,1.6213x,1.7256x**. Native2.8.4/new ratios are **0.9250,1.3108,1.3148,0.9373**, so all four meet native90%.

[Per-case evidence and reproduction](performance-20261112/README.md) includes the exact native version/SO hash, full sample arrays, trace/kernel witnesses, source fingerprints, input assumptions and an offline verifier. These are reconstructed metadata workloads, not original model tensor payload replay. No full profiler archive is added to the repository.
