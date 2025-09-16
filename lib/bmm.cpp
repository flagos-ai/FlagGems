#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor bmm(const at::Tensor& A, const at::Tensor& B) {
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "both the tensors must be 3-D");
  TORCH_CHECK(A.dtype() == B.dtype(),
              "expected a and b to have the same dtype, but got: ",
              A.dtype(),
              " != ",
              B.dtype());
  at::IntArrayRef A_sizes = A.sizes();
  at::IntArrayRef B_sizes = B.sizes();

  at::Tensor A_contig = A.contiguous();
  at::Tensor B_contig = B.contiguous();

  at::Tensor out = at::empty({A_sizes[0], A_sizes[1], B_sizes[2]}, A_contig.options());

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "bmm.py"),
                                      "bmm_kernel");

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  const int GROUP_M = 8;
  const int TILE_M = 128;
  const int TILE_N = 128;
  const int TILE_K = 32;
  const int M = A_sizes[1];
  const int N = B_sizes[2];
  const int K = A_sizes[2];
  unsigned int grid_x = (M + (TILE_M - 1)) / TILE_M;
  unsigned int grid_y = (N + (TILE_N - 1)) / TILE_N;
  bool DIVISIBLE_M = (M % TILE_M == 0);
  bool DIVISIBLE_N = (N % TILE_N == 0);
  bool DIVISIBLE_K = (K % TILE_K == 0);

  f(/* CUstream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ grid_y,
    /* grid_z = */ A_sizes[0],
    /* num_warps = */ 4,
    /* num_stages = */ 1,
    A_contig,
    B_contig,
    out,
    A_sizes[1],
    B_sizes[2],
    A_sizes[2],
    TILE_M,
    TILE_N,
    TILE_K,
    GROUP_M,
    DIVISIBLE_M,
    DIVISIBLE_N,
    DIVISIBLE_K);
  return out;
}

}  // namespace flag_gems
