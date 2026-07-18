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

#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  bool is_unconstrained_empty(const at::Tensor& tensor) {
    return tensor.dim() == 1 && tensor.size(0) == 0;
  }

  int64_t cat_dim_size_of(const at::Tensor& tensor, int64_t dim) {
    return is_unconstrained_empty(tensor) ? 0 : tensor.size(dim);
  }

  at::ScalarType promote_cat_dtypes(const at::Tensor& ref_tensor, const at::TensorList& tensors) {
    at::ScalarType promoted_dtype = ref_tensor.scalar_type();
    for (const auto& tensor : tensors) {
      if (is_unconstrained_empty(tensor)) {
        continue;
      }
      promoted_dtype = c10::promoteTypes(promoted_dtype, tensor.scalar_type());
    }
    return promoted_dtype;
  }

}  // namespace

at::Tensor cat(const at::TensorList& tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "torch.cat(): expected a non-empty list of Tensors");
  if (tensors.size() == 1) {
    return tensors[0];
  }

  const at::Tensor* ref_tensor = nullptr;
  int64_t non_empty_count = 0;
  const at::Tensor* single_non_empty = nullptr;
  for (const auto& tensor : tensors) {
    if (!is_unconstrained_empty(tensor)) {
      if (ref_tensor == nullptr) {
        ref_tensor = &tensor;
      }
      single_non_empty = &tensor;
      non_empty_count++;
    }
  }

  if (ref_tensor == nullptr) {
    return at::empty({0}, tensors[0].options());
  }

  if (non_empty_count == 1) {
    return *single_non_empty;
  }

  int64_t ndim = ref_tensor->dim();
  TORCH_CHECK(dim >= -ndim && dim < ndim, "cat(): dimension out of range");
  if (dim < 0) {
    dim += ndim;
  }

  const at::IntArrayRef ref_shape = ref_tensor->sizes();
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& current_tensor = tensors[i];
    if (is_unconstrained_empty(current_tensor)) {
      continue;
    }
    TORCH_CHECK(current_tensor.dim() == ndim,
                "Tensors must have same number of dimensions: got ",
                ndim,
                " and ",
                current_tensor.dim());
    const at::IntArrayRef current_shape = current_tensor.sizes();
    for (int64_t d = 0; d < ndim; ++d) {
      if (d == dim) continue;
      TORCH_CHECK(current_shape[d] == ref_shape[d],
                  "Sizes of tensors must match except in dimension ",
                  dim,
                  ". Expected size ",
                  ref_shape[d],
                  " but got size ",
                  current_shape[d],
                  " for tensor number ",
                  i);
    }
  }

  const at::ScalarType out_dtype = promote_cat_dtypes(*ref_tensor, tensors);

  std::vector<int64_t> out_shape_vec = ref_shape.vec();
  int64_t cat_dim_size = 0;
  for (const auto& t : tensors) {
    cat_dim_size += cat_dim_size_of(t, dim);
  }
  out_shape_vec[dim] = cat_dim_size;
  at::Tensor out = at::empty(out_shape_vec, ref_tensor->options().dtype(out_dtype));

  std::vector<int64_t> storage_offsets;
  int64_t current_storage_offset = 0;
  storage_offsets.push_back(current_storage_offset);
  int64_t out_stride_for_dim = out.stride(dim);
  for (size_t i = 0; i < tensors.size() - 1; ++i) {
    current_storage_offset += cat_dim_size_of(tensors[i], dim) * out_stride_for_dim;
    storage_offsets.push_back(current_storage_offset);
  }

  const TritonJITFunction& copy_kernel_func =
      TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "cat_copy.py"),
                                      "strided_copy_kernel");
  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& input_tensor = tensors[i];
    if (input_tensor.numel() == 0) continue;

    at::Tensor src_tensor = input_tensor;
    if (input_tensor.scalar_type() != out_dtype) {
      src_tensor = input_tensor.to(out_dtype);
    }

    at::Tensor output_view = at::as_strided(out, src_tensor.sizes(), out.strides(), storage_offsets[i]);

    auto options = torch::TensorOptions().device(src_tensor.device()).dtype(torch::kInt64);
    at::Tensor in_strides = torch::tensor(src_tensor.strides(), options);
    at::Tensor out_strides = torch::tensor(output_view.strides(), options);
    at::Tensor shapes = torch::tensor(src_tensor.sizes(), options);

    int64_t ndim_val = src_tensor.dim();
    int64_t num_elements = src_tensor.numel();

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
                     src_tensor,
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
