#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include <tuple>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> topk(
    const at::Tensor &x, int64_t k, int64_t dim, bool largest, bool sorted) {
  TORCH_CHECK(x.dim() >= 1, "input tensor must have at least one dimension");
  TORCH_CHECK(sorted, "currently only support sorted == true");
  dim = dim < 0 ? x.dim() + dim : dim;
  TORCH_CHECK(dim == x.dim() - 1, "currently only support topk in last dimension");
  auto topk_elem_cnt = x.size(dim);
  int64_t batch_size = 1;
  for (int i = 0; i < x.dim() - 1; i++) {
    batch_size *= x.size(i);
  }
  bool descending = largest;
  int64_t chunk_size = (topk_elem_cnt >= 1024) ? 1024 : 256;
  if (chunk_size < k) {
    chunk_size = utils::next_power_of_2(k);
  }
  int64_t chunk_num = (topk_elem_cnt + chunk_size - 1) / chunk_size;
  at::Tensor stage1_out = at::empty(batch_size * chunk_num * k, x.options());
  at::Tensor stage1_out_idx =
      at::empty(batch_size * chunk_num * k, at::TensorOptions().dtype(torch::kLong).device(x.device()));
  int64_t stage2_elem_cnt = chunk_num * k;
  int64_t BLOCK_SIZE = utils::next_power_of_2(stage2_elem_cnt);
  auto out_shape = x.sizes().vec();
  out_shape[out_shape.size() - 1] = k;
  at::Tensor stage2_out = at::empty(out_shape, x.options());
  at::Tensor stage2_out_idx =
      at::empty(out_shape, at::TensorOptions().dtype(torch::kLong).device(x.device()));
  const TritonJITFunction &f1 =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "topk.py"),
                                      "topk_stage1_kernel");
  const TritonJITFunction &f2 =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "topk.py"),
                                      "topk_stage2_kernel");
  c10::DeviceGuard guard(stage1_out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  /*
  def topk_stage1_kernel(y_ptr,
                         index_ptr,
                         x_ptr,
                         k,
                         N: tl.constexpr,
                         CHUNK_SIZE: tl.constexpr,
                         DESCENDING: tl.constexpr):
*/
  f1(raw_stream,
     batch_size,
     chunk_num,
     1,
     /* num_warps */ 8,
     /* num_stages */ 1,
     stage1_out,
     stage1_out_idx,
     x,
     k,
     topk_elem_cnt,
     chunk_size,
     descending);
  /*
  def topk_stage2_kernel(y_ptr,
                           index_ptr,
                           chunk_x,
                           chunk_index,
                           sort_dim: tl.constexpr,
                           k: tl.constexpr,
                           N: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr,
                           DESCENDING: tl.constexpr,):
*/
  f2(raw_stream,
     batch_size,
     1,
     1,
     8,
     1,
     stage2_out,
     stage2_out_idx,
     stage1_out,
     stage1_out_idx,
     dim,
     k,
     stage2_elem_cnt,
     BLOCK_SIZE,
     descending);

  return std::make_tuple(stage2_out, stage2_out_idx);
}
}  // namespace flag_gems
