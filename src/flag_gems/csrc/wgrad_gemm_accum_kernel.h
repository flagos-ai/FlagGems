// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Single source of truth for Apex-aligned wgrad GemmEx (fp32 accum).
// Included by:
//   - src/flag_gems/csrc/wgrad_gemm_accum.cpp  (Torch JIT extension + pybind)
//   - cpp/lib/wgrad_gemm_accum.cpp             (official c_operators build)
//
// IMPORTANT: resolve cublasGemmEx via dlsym(RTLD_DEFAULT) so we call the same
// libcublas that created PyTorch's BLAS handle. Linking a second -lcublas and
// mixing handles causes CUBLAS_STATUS_INVALID_VALUE.
//
// Do NOT override cublas math mode here: rely on PyTorch's handle (respects
// torch.backends.cuda.matmul.allow_tf32).

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <torch/torch.h>

namespace flag_gems_wgrad_detail {
namespace {

inline cudaDataType dtype_to_cuda_data_type(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return CUDA_R_32F;
    case at::kHalf:
      return CUDA_R_16F;
    case at::kBFloat16:
      return CUDA_R_16BF;
    default:
      TORCH_CHECK(false, "Unsupported dtype for wgrad_gemm_accum_fp32 GemmEx: ", dtype);
  }
}

// Match the C ABI used by Torch's libcublas:
// computeType is passed as cudaDataType / int (CUDA_R_32F == 0).
using cublasGemmExFn = cublasStatus_t (*)(cublasHandle_t,
                                          cublasOperation_t,
                                          cublasOperation_t,
                                          int,
                                          int,
                                          int,
                                          const void *,
                                          const void *,
                                          cudaDataType,
                                          int,
                                          const void *,
                                          cudaDataType,
                                          int,
                                          const void *,
                                          void *,
                                          cudaDataType,
                                          int,
                                          cudaDataType,
                                          cublasGemmAlgo_t);

inline cublasGemmExFn resolve_gemm_ex() {
  static cublasGemmExFn fn = nullptr;
  if (fn == nullptr) {
    fn = reinterpret_cast<cublasGemmExFn>(dlsym(RTLD_DEFAULT, "cublasGemmEx"));
    TORCH_CHECK(fn != nullptr,
                "dlsym(cublasGemmEx) failed; is Torch CUDA / libcublas loaded?");
  }
  return fn;
}

}  // namespace

inline void wgrad_gemm_accum_fp32_cuda_impl(const at::Tensor &input_2d,
                                            const at::Tensor &grad_output_2d,
                                            at::Tensor &main_grad) {
  TORCH_CHECK(input_2d.is_cuda() && grad_output_2d.is_cuda() && main_grad.is_cuda(),
              "wgrad_gemm_accum_fp32: tensors must be CUDA");
  TORCH_CHECK(main_grad.scalar_type() == at::kFloat,
              "main_grad must be float32 for GemmEx fp32-accum path");
  TORCH_CHECK(input_2d.scalar_type() == grad_output_2d.scalar_type(),
              "input and grad_output dtype must match, got ",
              input_2d.scalar_type(),
              " vs ",
              grad_output_2d.scalar_type());
  TORCH_CHECK(input_2d.dim() == 2 && grad_output_2d.dim() == 2 && main_grad.dim() == 2,
              "expected 2D tensors after collapse");

  at::Tensor input = input_2d.contiguous();
  at::Tensor grad_output = grad_output_2d.contiguous();

  const int64_t hidden_dim = input.size(0);
  const int64_t in_dim = input.size(1);
  const int64_t out_dim = grad_output.size(1);
  TORCH_CHECK(grad_output.size(0) == hidden_dim,
              "input/grad_output row mismatch after collapse");
  TORCH_CHECK(main_grad.size(0) == out_dim && main_grad.size(1) == in_dim,
              "main_grad shape mismatch: expected (",
              out_dim,
              ", ",
              in_dim,
              "), got (",
              main_grad.size(0),
              ", ",
              main_grad.size(1),
              ")");

  if (hidden_dim == 0) {
    return;
  }
  TORCH_CHECK(in_dim > 0 && out_dim > 0,
              "wgrad_gemm_accum_fp32: in_features and out_features must be > 0 "
              "(got in=",
              in_dim,
              ", out=",
              out_dim,
              "); Apex/cublasGemmEx also reject zero M/N");

  const float alpha = 1.0f;
  const float beta = 1.0f;
  const cudaDataType a_type = dtype_to_cuda_data_type(input.scalar_type());

  at::cuda::CUDAGuard device_guard(input.device());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // Fast path 1: contiguous main_grad.
  // layout: main_grad(out,in) += grad_output.T @ input
  // C = alpha * OP_N(A=input) * OP_T(B=grad_output) + beta * C
  if (main_grad.is_contiguous()) {
    cublasStatus_t status = resolve_gemm_ex()(handle,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              static_cast<int>(in_dim),
                                              static_cast<int>(out_dim),
                                              static_cast<int>(hidden_dim),
                                              &alpha,
                                              input.data_ptr(),
                                              a_type,
                                              static_cast<int>(in_dim),
                                              grad_output.data_ptr(),
                                              a_type,
                                              static_cast<int>(out_dim),
                                              &beta,
                                              main_grad.data_ptr(),
                                              CUDA_R_32F,
                                              static_cast<int>(in_dim),
                                              CUDA_R_32F,
                                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cublasGemmEx failed with status ",
                static_cast<int>(status));
    return;
  }

  // Fast path 2: main_grad is a transpose view of a contiguous (in,out) buffer.
  // Equiv: main_grad.T(in,out) += input.T @ grad_output  (write through, no densify).
  at::Tensor main_t = main_grad.transpose(0, 1);
  if (main_t.is_contiguous()) {
    cublasStatus_t status = resolve_gemm_ex()(handle,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              static_cast<int>(out_dim),
                                              static_cast<int>(in_dim),
                                              static_cast<int>(hidden_dim),
                                              &alpha,
                                              grad_output.data_ptr(),
                                              a_type,
                                              static_cast<int>(out_dim),
                                              input.data_ptr(),
                                              a_type,
                                              static_cast<int>(in_dim),
                                              &beta,
                                              main_t.data_ptr(),
                                              CUDA_R_32F,
                                              static_cast<int>(out_dim),
                                              CUDA_R_32F,
                                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cublasGemmEx (transpose-contig main_grad) failed with status ",
                static_cast<int>(status));
    return;
  }

  // Slow path: general non-contiguous — densify, GemmEx, copy_ back.
  at::Tensor weight = main_grad.contiguous();
  cublasStatus_t status = resolve_gemm_ex()(handle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_T,
                                            static_cast<int>(in_dim),
                                            static_cast<int>(out_dim),
                                            static_cast<int>(hidden_dim),
                                            &alpha,
                                            input.data_ptr(),
                                            a_type,
                                            static_cast<int>(in_dim),
                                            grad_output.data_ptr(),
                                            a_type,
                                            static_cast<int>(out_dim),
                                            &beta,
                                            weight.data_ptr(),
                                            CUDA_R_32F,
                                            static_cast<int>(in_dim),
                                            CUDA_R_32F,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "cublasGemmEx failed with status ",
              static_cast<int>(status));
  main_grad.copy_(weight);
}

}  // namespace flag_gems_wgrad_detail
