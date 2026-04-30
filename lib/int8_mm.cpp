#include "flag_gems/backend_utils.h"
#include "flag_gems/device_info.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include <tuple>
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

static inline int cdiv(int x, int y) {
  return (x + y - 1) / y;
}

at::Tensor int8_mm(const at::Tensor &mat1, const at::Tensor &mat2) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "both the tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == at::kChar && mat2.dtype() == at::kChar,
              "expected int8 dtype, got mat1=",
              mat1.dtype(),
              " mat2=",
              mat2.dtype());
  TORCH_CHECK(mat1.size(1) == mat2.size(0), "incompatible dimensions");

  int M = mat1.size(0);
  int K = mat1.size(1);
  int N = mat2.size(1);

  at::Tensor out = at::empty({M, N}, mat1.options().dtype(at::kInt));
  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "int8_mm.py"),
                                      "int8_mm_kernel");

  // refer to the H800
  int BLOCK_M = M <= 1024 ? 32 : 128;
  int BLOCK_N = N <= 1024 ? 32 : 64;
  int BLOCK_K = K < 128 ? 32 : 128;
  int num_stages = BLOCK_M < 128 ? 2 : 1;
  constexpr int num_warps = 4;
  constexpr int GROUP_M = 8;

  unsigned int grid_x = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  f(/* stream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    num_warps,
    num_stages,
    mat1,
    mat2,
    out,
    M,
    N,
    K,
    mat1.stride(0),
    mat1.stride(1),
    mat2.stride(0),
    mat2.stride(1),
    out.stride(0),
    out.stride(1),
    /* BLOCK_M = */ BLOCK_M,
    /* BLOCK_N = */ BLOCK_N,
    /* BLOCK_K = */ BLOCK_K,
    /* GROUP_M = */ GROUP_M,
    /* num_stages = */ num_stages);

  return out;
}

at::Tensor &int8_mm_out(const at::Tensor &mat1, const at::Tensor &mat2, at::Tensor &out) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "both the tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == at::kChar && mat2.dtype() == at::kChar,
              "expected int8 dtype, got mat1=",
              mat1.dtype(),
              " mat2=",
              mat2.dtype());
  TORCH_CHECK(out.dtype() == at::kInt, "out.dtype != int32");
  TORCH_CHECK(mat1.size(1) == mat2.size(0), "mat1 and mat2 incompatible dimensions");
  TORCH_CHECK(out.size(0) == mat1.size(0), "out and mat1 incompatible dimensions");
  TORCH_CHECK(out.size(1) == mat2.size(1), "out and mat2 incompatible dimensions");

  int M = mat1.size(0);
  int K = mat1.size(1);
  int N = mat2.size(1);

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "int8_mm.py"),
                                      "int8_mm_kernel");

  // refer to the H800
  int BLOCK_M = M <= 1024 ? 32 : 128;
  int BLOCK_N = N <= 1024 ? 32 : 64;
  int BLOCK_K = K < 128 ? 32 : 128;
  int num_stages = BLOCK_M < 128 ? 2 : 1;
  constexpr int num_warps = 4;
  constexpr int GROUP_M = 8;

  unsigned int grid_x = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  f(/* stream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    num_warps,
    num_stages,
    mat1,
    mat2,
    out,
    M,
    N,
    K,
    mat1.stride(0),
    mat1.stride(1),
    mat2.stride(0),
    mat2.stride(1),
    out.stride(0),
    out.stride(1),
    /* BLOCK_M = */ BLOCK_M,
    /* BLOCK_N = */ BLOCK_N,
    /* BLOCK_K = */ BLOCK_K,
    /* GROUP_M = */ GROUP_M,
    /* num_stages = */ num_stages);

  return out;
}

}  // namespace flag_gems
