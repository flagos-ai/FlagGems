# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performance benchmarks for te_general_grouped_gemm operator.

Compares FlagGems Triton implementation against:
1. PyTorch torch.matmul (sequential)
2. TransformerEngine tex.te_general_grouped_gemm (when available)
"""

import argparse
import time
from typing import Callable, List, Tuple

import torch

from flag_gems.fused import te_general_grouped_gemm

# Try to import TransformerEngine
try:
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch.cpp_extensions.gemm import get_cublas_workspace

    HAS_TE = True
except ImportError:
    HAS_TE = False
    tex = None


def benchmark_fn(fn: Callable, warmup: int = 10, repeat: int = 100) -> Tuple[float, float]:
    """Benchmark a function."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    return mean_time, std_time


def create_test_tensors(
    num_gemms: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    transa: bool = True,
    transb: bool = False,
    device: str = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create test tensors for grouped GEMM."""
    A_list = []
    B_list = []

    for _ in range(num_gemms):
        if transa:
            A = torch.randn(K, M, dtype=dtype, device=device) * 0.1
        else:
            A = torch.randn(M, K, dtype=dtype, device=device) * 0.1

        if transb:
            B = torch.randn(N, K, dtype=dtype, device=device) * 0.1
        else:
            B = torch.randn(K, N, dtype=dtype, device=device) * 0.1

        A_list.append(A)
        B_list.append(B)

    return A_list, B_list


def torch_grouped_gemm(A_list: List[torch.Tensor], B_list: List[torch.Tensor], transa: bool, transb: bool):
    """PyTorch sequential grouped GEMM."""
    results = []
    for A, B in zip(A_list, B_list):
        A_mat = A.T if transa else A
        B_mat = B.T if transb else B
        results.append(torch.matmul(A_mat, B_mat))
    return results


def run_benchmark(
    num_gemms: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    transa: bool = True,
    transb: bool = False,
    warmup: int = 10,
    repeat: int = 100,
    verbose: bool = True,
) -> dict:
    """Run benchmark for a specific configuration."""
    A_list, B_list = create_test_tensors(num_gemms, M, N, K, dtype, transa, transb)

    results = {
        "num_gemms": num_gemms,
        "M": M,
        "N": N,
        "K": K,
        "dtype": str(dtype),
        "transa": transa,
        "transb": transb,
    }

    # Benchmark PyTorch sequential
    def torch_fn():
        return torch_grouped_gemm(A_list, B_list, transa, transb)

    torch_time, torch_std = benchmark_fn(torch_fn, warmup, repeat)
    results["torch_time_ms"] = torch_time
    results["torch_std_ms"] = torch_std

    # Benchmark FlagGems
    out_list_gems = [torch.empty(M, N, dtype=dtype, device="cuda") for _ in range(num_gemms)]
    empty_tensor = torch.tensor([], device="cuda")
    empty_tensors = [empty_tensor] * num_gemms
    workspace = [torch.zeros(1, dtype=torch.uint8, device="cuda")]

    def flaggems_fn():
        te_general_grouped_gemm(
            A=A_list,
            transa=transa,
            B=B_list,
            transb=transb,
            D=out_list_gems,
            D_type=dtype,
            m_splits=[],
            bias=empty_tensors,
            bias_type=torch.bfloat16,
            single_output=False,
            pre_gelu_out=empty_tensors,
            grad=False,
            workspace=workspace,
            workspaceSize=1,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

    flaggems_time, flaggems_std = benchmark_fn(flaggems_fn, warmup, repeat)
    results["flaggems_time_ms"] = flaggems_time
    results["flaggems_std_ms"] = flaggems_std
    results["speedup_vs_torch"] = torch_time / flaggems_time

    # Benchmark TransformerEngine if available
    if HAS_TE:
        out_list_te = [torch.empty(M, N, dtype=dtype, device="cuda") for _ in range(num_gemms)]
        workspaces = get_cublas_workspace(0, False, True)
        te_dtype = tex.DType.kFloat16 if dtype == torch.float16 else tex.DType.kBFloat16

        def te_fn():
            tex.te_general_grouped_gemm(
                A_list,
                transa,
                B_list,
                transb,
                out_list_te,
                te_dtype,
                [],
                empty_tensors,
                tex.DType.kBFloat16,
                False,
                empty_tensors,
                False,
                workspaces,
                workspaces[0].shape[0],
                False,
                False,
                0,
            )

        te_time, te_std = benchmark_fn(te_fn, warmup, repeat)
        results["te_time_ms"] = te_time
        results["te_std_ms"] = te_std
        results["speedup_vs_te"] = te_time / flaggems_time

    # Calculate FLOPS
    total_flops = num_gemms * (2 * M * N * K)
    results["tflops_torch"] = total_flops / (torch_time * 1e9)
    results["tflops_flaggems"] = total_flops / (flaggems_time * 1e9)
    if HAS_TE:
        results["tflops_te"] = total_flops / (te_time * 1e9)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Config: {num_gemms} GEMMs, ({M}, {N}, {K}), {dtype}, transa={transa}, transb={transb}")
        print(f"{'='*70}")
        print(f"PyTorch (sequential): {torch_time:.3f} ± {torch_std:.3f} ms ({results['tflops_torch']:.2f} TFLOPS)")
        print(f"FlagGems (Triton):    {flaggems_time:.3f} ± {flaggems_std:.3f} ms ({results['tflops_flaggems']:.2f} TFLOPS)")
        print(f"  Speedup vs PyTorch: {results['speedup_vs_torch']:.2f}x")
        if HAS_TE:
            print(f"TransformerEngine:   {te_time:.3f} ± {te_std:.3f} ms ({results['tflops_te']:.2f} TFLOPS)")
            print(f"  Speedup vs TE:      {results['speedup_vs_te']:.2f}x")

    return results


def run_all_benchmarks(
    warmup: int = 10,
    repeat: int = 100,
    quick: bool = False,
):
    """Run comprehensive benchmarks."""
    if quick:
        num_gemms_list = [4, 8]
        shapes = [
            (256, 256, 128),
            (512, 512, 256),
            (1024, 1024, 512),
        ]
        dtypes = [torch.float16]
    else:
        num_gemms_list = [1, 2, 4, 8, 16]
        shapes = [
            (128, 128, 64),
            (256, 256, 128),
            (512, 512, 256),
            (1024, 1024, 512),
            (2048, 2048, 1024),
            (256, 1024, 512),
            (256, 4096, 1024),
        ]
        dtypes = [torch.float16, torch.bfloat16]

    all_results = []

    print("\n" + "=" * 100)
    print(" FlagGems te_general_grouped_gemm Performance Benchmark")
    print("=" * 100)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"TransformerEngine available: {HAS_TE}")
    print(f"Warmup iterations: {warmup}")
    print(f"Measurement iterations: {repeat}")

    # Use TN layout (default for TE)
    transa, transb = True, False

    for dtype in dtypes:
        for num_gemms in num_gemms_list:
            for M, N, K in shapes:
                try:
                    result = run_benchmark(
                        num_gemms=num_gemms,
                        M=M,
                        N=N,
                        K=K,
                        dtype=dtype,
                        transa=transa,
                        transb=transb,
                        warmup=warmup,
                        repeat=repeat,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Error with config ({num_gemms}, {M}, {N}, {K}, {dtype}): {e}")

    # Print summary table
    print("\n\n" + "=" * 110)
    print(" Summary Table")
    print("=" * 110)
    header = f"{'#GEMMs':>6} {'Shape':>20} {'dtype':>8} {'PyTorch':>10} {'FlagGems':>10} {'Speedup':>8}"
    if HAS_TE:
        header += f" {'TE':>10} {'vs TE':>8}"
    print(header)
    print("-" * 110)

    for r in all_results:
        shape_str = f"({r['M']},{r['N']},{r['K']})"
        dtype_str = "fp16" if "float16" in r["dtype"] else "bf16"
        row = f"{r['num_gemms']:>6} {shape_str:>20} {dtype_str:>8} {r['torch_time_ms']:>10.3f} {r['flaggems_time_ms']:>10.3f} {r['speedup_vs_torch']:>8.2f}x"
        if HAS_TE and "te_time_ms" in r:
            row += f" {r['te_time_ms']:>10.3f} {r['speedup_vs_te']:>8.2f}x"
        print(row)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark te_general_grouped_gemm")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Measurement iterations")
    parser.add_argument("--num-gemms", type=int, help="Number of GEMMs")
    parser.add_argument("--M", type=int, help="M dimension")
    parser.add_argument("--N", type=int, help="N dimension")
    parser.add_argument("--K", type=int, help="K dimension")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if args.num_gemms and args.M and args.N and args.K:
        run_benchmark(
            num_gemms=args.num_gemms,
            M=args.M,
            N=args.N,
            K=args.K,
            dtype=dtype,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    else:
        run_all_benchmarks(
            warmup=args.warmup,
            repeat=args.repeat,
            quick=args.quick,
        )


if __name__ == "__main__":
    main()
