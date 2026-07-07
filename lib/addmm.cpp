#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  struct AddmmLaunchConfig {
    int block_m;
    int block_n;
    int block_k;
    int num_warps;
    int num_stages;
  };

  AddmmLaunchConfig get_addmm_launch_config() {
#if defined(FLAGGEMS_USE_IX)
    // Match an iluvatar tuned config instead of reusing the kunlunxin launch
    // parameters. The previous (2 warps, 5 stages) combination caused bf16
    // accuracy regressions in the C++ addmm path on IX devices.
    return {32, 64, 32, 4, 1};
#else
    return {32, 64, 32, 2, 5};
#endif
  }

}  // namespace

at::Tensor addmm(const at::Tensor& self,
                 const at::Tensor& mat1,
                 const at::Tensor& mat2,
                 const at::Scalar& beta,
                 const at::Scalar& alpha) {
  at::IntArrayRef mat1_sizes = mat1.sizes();
  at::IntArrayRef mat2_sizes = mat2.sizes();
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "Incompatible dimensions");
  TORCH_CHECK(utils::broadcastable_to(self.sizes(), at::IntArrayRef({mat1_sizes[0], mat2_sizes[1]})),
              "Incompatible input shape");
  at::Tensor mat1_c = mat1.contiguous();
  // at::Tensor mat2_c = mat2.contiguous();
  at::Tensor out = at::empty({mat1_sizes[0], mat2_sizes[1]}, mat1.options());
  at::Tensor self_b = self.broadcast_to(out.sizes());
  float alpha_val = alpha.to<float>();
  float beta_val = beta.to<float>();

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "addmm.py"),
                                      "addmm_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);
  const AddmmLaunchConfig config = get_addmm_launch_config();
  unsigned int grid_x = ((mat1_sizes[0] + config.block_m - 1) / config.block_m);
  unsigned int grid_y = ((mat2_sizes[1] + config.block_n - 1) / config.block_n);
  f(/* CUstream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ grid_y,
    /* grid_z = */ 1,
    /* num_warps = */ config.num_warps,
    /* num_stages = */ config.num_stages,
    mat1_c,
    mat2,
    self_b,
    out,
    alpha_val,
    beta_val,
    mat1_sizes[0],
    mat2_sizes[1],
    mat1_sizes[1],
    mat1_c.stride(0),
    mat1_c.stride(1),
    mat2.stride(0),
    mat2.stride(1),
    self_b.stride(0),
    self_b.stride(1),
    out.stride(0),
    out.stride(1),
    /* BLOCK_M = */ config.block_m,
    /* BLOCK_N = */ config.block_n,
    /* BLOCK_K = */ config.block_k);
  return out;
}

at::Tensor& addmm_out(const at::Tensor& self,
                      const at::Tensor& mat1,
                      const at::Tensor& mat2,
                      const at::Scalar& beta,
                      const at::Scalar& alpha,
                      at::Tensor& out) {
  at::IntArrayRef mat1_sizes = mat1.sizes();
  at::IntArrayRef mat2_sizes = mat2.sizes();
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "Incompatible dimensions");
  TORCH_CHECK(utils::broadcastable_to(self.sizes(), at::IntArrayRef({mat1_sizes[0], mat2_sizes[1]})),
              "Incompatible input shape");
  at::Tensor mat1_c = mat1.contiguous();
  // at::Tensor mat2_c = mat2.contiguous();
  at::Tensor self_b = self.broadcast_to(out.sizes());
  float alpha_val = alpha.to<float>();
  float beta_val = beta.to<float>();

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "addmm.py"),
                                      "addmm_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);
  const AddmmLaunchConfig config = get_addmm_launch_config();
  unsigned int grid_x = ((mat1_sizes[0] + config.block_m - 1) / config.block_m);
  unsigned int grid_y = ((mat2_sizes[1] + config.block_n - 1) / config.block_n);
  f(/* CUstream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ grid_y,
    /* grid_z = */ 1,
    /* num_warps = */ config.num_warps,
    /* num_stages = */ config.num_stages,
    mat1_c,
    mat2,
    self_b,
    out,
    alpha_val,
    beta_val,
    mat1_sizes[0],
    mat2_sizes[1],
    mat1_sizes[1],
    mat1_c.stride(0),
    mat1_c.stride(1),
    mat2.stride(0),
    mat2.stride(1),
    self_b.stride(0),
    self_b.stride(1),
    out.stride(0),
    out.stride(1),
    /* BLOCK_M = */ config.block_m,
    /* BLOCK_N = */ config.block_n,
    /* BLOCK_K = */ config.block_k);
  return out;
}

}  // namespace flag_gems
