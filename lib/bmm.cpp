#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

static inline int64_t cdiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

at::Tensor bmm(const at::Tensor& A_in, const at::Tensor& B_in) {
  TORCH_CHECK(A_in.dim() == 3 && B_in.dim() == 3, "both the tensors must be 3-D");
  TORCH_CHECK(A_in.dtype() == B_in.dtype(),
              "expected a and b to have the same dtype, but got: ",
              A_in.dtype(),
              " != ",
              B_in.dtype());

  at::Tensor A = A_in.contiguous();
  at::Tensor B = B_in.contiguous();

  at::IntArrayRef A_sizes = A.sizes();
  at::IntArrayRef B_sizes = B.sizes();

  const int64_t batch = A_sizes[0];
  const int64_t M = A_sizes[1];
  const int64_t N = B_sizes[2];
  const int64_t K = A_sizes[2];

  at::Tensor out = at::empty({batch, M, N}, A.options());

#if defined(FLAGGEMS_USE_IX)
  // On IX (CoreX/iLuvatar) backend, the batched bmm kernel triggers a
  // "divergent base ptr" compiler error due to Triton compiler limitations.
  // Work around by dispatching per-batch 2D mm kernels instead.
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "mm.py"),
                                      "mm_kernel_general");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  const int BLOCK_M = 64;
  const int BLOCK_N = 128;
  const int BLOCK_K = 64;
  const int num_stages = 2;
  const int num_warps = 4;
  const int GROUP_M = 8;

  unsigned int grid_x = static_cast<unsigned int>(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N));

  for (int64_t b = 0; b < batch; ++b) {
    at::Tensor a_slice = A[b];
    at::Tensor b_slice = B[b];
    at::Tensor o_slice = out[b];

    f(/* stream = */ raw_stream,
      /* grid_x = */ grid_x,
      /* grid_y = */ 1u,
      /* grid_z = */ 1u,
      num_warps,
      num_stages,
      a_slice,
      b_slice,
      o_slice,
      (int64_t)M,
      (int64_t)N,
      (int64_t)K,
      a_slice.stride(0),
      a_slice.stride(1),
      b_slice.stride(0),
      b_slice.stride(1),
      o_slice.stride(0),
      o_slice.stride(1),
      /* BLOCK_M = */ BLOCK_M,
      /* BLOCK_N = */ BLOCK_N,
      /* BLOCK_K = */ BLOCK_K,
      /* GROUP_M = */ GROUP_M);
  }
#else
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "bmm.py"),
                                      "bmm_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);
  const int GROUP_M = 8;
  const int TILE_M = 128;
  const int TILE_N = 128;
  const int TILE_K = 32;
  unsigned int grid_x = static_cast<unsigned int>(cdiv(M, TILE_M));
  unsigned int grid_y = static_cast<unsigned int>(cdiv(N, TILE_N));
  bool DIVISIBLE_M = (M % TILE_M == 0);
  bool DIVISIBLE_N = (N % TILE_N == 0);
  bool DIVISIBLE_K = (K % TILE_K == 0);

  f(/* CUstream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ grid_y,
    /* grid_z = */ (unsigned int)batch,
    /* num_warps = */ 4,
    /* num_stages = */ 1,
    A,
    B,
    out,
    (int)M,
    (int)N,
    (int)K,
    A.stride(0),
    A.stride(1),
    A.stride(2),
    B.stride(0),
    B.stride(1),
    B.stride(2),
    out.stride(0),
    out.stride(1),
    out.stride(2),
    TILE_M,
    TILE_N,
    TILE_K,
    GROUP_M,
    DIVISIBLE_M,
    DIVISIBLE_N,
    DIVISIBLE_K);
#endif
  return out;
}

}  // namespace flag_gems
