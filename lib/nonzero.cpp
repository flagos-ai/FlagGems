#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor nonzero(const at::Tensor &inp) {
  int64_t inp_ndim = inp.dim();

  at::Tensor inp_ctg = inp.contiguous();
  int64_t n_elements = inp_ctg.numel();
  at::Tensor inp_view = inp.view(n_elements);

  at::Tensor shape =
      torch::tensor(inp_ctg.sizes(), at::TensorOptions().dtype(torch::kInt32).device(inp_ctg.device()));

  at::Tensor inp_bool = inp_view;
  if (inp_view.dtype() != torch::kBool) {
    inp_bool = (inp_view != 0);
  }

  at::Tensor prefix_sum = at::cumsum(inp_bool, /*dim=*/0);

  int64_t num_nonzeros = n_elements;
  at::Tensor out =
      at::empty({num_nonzeros, inp_ndim}, at::TensorOptions().dtype(torch::kInt64).device(inp_ctg.device()));

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "nonzero.py"),
                                      "nonzero_kernel");

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  int BLOCK_SIZE = 1024;
  unsigned int grid_x = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  f(/* CUstream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    /* num_warps = */ 8,
    /* num_stages = */ 4,
    inp_bool,
    prefix_sum,
    out,
    n_elements,
    shape,
    inp_ndim,
    /* BLOCK_SIZE */ BLOCK_SIZE);

  num_nonzeros = prefix_sum[n_elements - 1].item().to<int64_t>();
  out = out.index({torch::indexing::Slice(0, num_nonzeros)});
  return out;
}

}  // namespace flag_gems
