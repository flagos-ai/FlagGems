#include "c10/cuda/CUDAStream.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor cat(const at::TensorList& tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "torch.cat(): expected a non-empty list of Tensors");
  if (tensors.size() == 1) {
    return tensors[0];
  }
  const auto& first_tensor = tensors[0];
  int64_t ndim = first_tensor.dim();
  TORCH_CHECK(dim >= -ndim && dim < ndim, "cat(): dimension out of range");
  if (dim < 0) {
    dim += ndim;
  }
  const at::IntArrayRef first_shape = first_tensor.sizes();
  for (size_t i = 1; i < tensors.size(); ++i) {
    const auto& current_tensor = tensors[i];
    TORCH_CHECK(current_tensor.dim() == ndim,
                "Tensors must have same number of dimensions: got ",
                ndim,
                " and ",
                current_tensor.dim());
    const at::IntArrayRef current_shape = current_tensor.sizes();
    for (int64_t d = 0; d < ndim; ++d) {
      if (d == dim) continue;
      TORCH_CHECK(current_shape[d] == first_shape[d],
                  "Sizes of tensors must match except in dimension ",
                  dim,
                  ". Expected size ",
                  first_shape[d],
                  " but got size ",
                  current_shape[d],
                  " for tensor number ",
                  i);
    }
  }

  std::vector<int64_t> out_shape_vec = first_shape.vec();
  int64_t cat_dim_size = 0;
  for (const auto& t : tensors) {
    cat_dim_size += t.size(dim);
  }
  out_shape_vec[dim] = cat_dim_size;
  at::Tensor out = at::empty(out_shape_vec, first_tensor.options());

  std::vector<int64_t> storage_offsets;
  int64_t current_storage_offset = 0;
  storage_offsets.push_back(current_storage_offset);
  int64_t out_stride_for_dim = out.stride(dim);
  for (size_t i = 0; i < tensors.size() - 1; ++i) {
    current_storage_offset += tensors[i].size(dim) * out_stride_for_dim;
    storage_offsets.push_back(current_storage_offset);
  }

  const TritonJITFunction& copy_kernel_func =
      TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "cat_copy.py"),
                                      "strided_copy_kernel");
  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& input_tensor = tensors[i];
    if (input_tensor.numel() == 0) continue;

    at::Tensor output_view = at::as_strided(out, input_tensor.sizes(), out.strides(), storage_offsets[i]);

    auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt64);
    at::Tensor in_strides = torch::tensor(input_tensor.strides(), options);
    at::Tensor out_strides = torch::tensor(output_view.strides(), options);
    at::Tensor shapes = torch::tensor(input_tensor.sizes(), options);

    int64_t ndim_val = input_tensor.dim();
    int64_t num_elements = input_tensor.numel();

    constexpr int BLOCK_SIZE = 256;
    constexpr int MAX_DIMS = 8;
    TORCH_CHECK(ndim_val <= MAX_DIMS,
                "Tensor dimension ",
                ndim_val,
                " exceeds the maximum supported by the kernel (",
                MAX_DIMS,
                ")");

    unsigned int grid_x = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    copy_kernel_func(raw_stream,
                     grid_x,
                     1,
                     1,
                     4,
                     2,
                     input_tensor,
                     output_view,
                     in_strides,
                     out_strides,
                     shapes,
                     ndim_val,
                     num_elements,
                     BLOCK_SIZE,
                     MAX_DIMS);
  }
  return out;
}
}  // namespace flag_gems
