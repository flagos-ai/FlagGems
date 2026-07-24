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

#include <algorithm>
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

  int64_t dim_prod_post = 1;
  for (int64_t d = dim + 1; d < ndim; ++d) {
    dim_prod_post *= out_shape_vec[d];
  }
  int64_t dim_size_out = out_shape_vec[dim];

  const TritonJITFunction& kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "cat.py"),
                                      "cat_copy_func_kernel_4");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  constexpr int BLOCK = 1024;
  constexpr int NUM_WARPS = 4;
  constexpr int NUM_STAGES = 1;

  size_t t_idx = 0;
  int64_t global_dim_offset = 0;

  while (t_idx < tensors.size()) {
    // Collect up to 4 tensors (skip numel==0 tensors within batch collect)
    at::Tensor batch_tensors[4];
    int64_t dim_sizes[4] = {0};
    int64_t dim_offsets[4] = {0};
    int64_t total_elements[4] = {0};

    size_t batch_count = 0;
    int64_t batch_offset = global_dim_offset;
    while (batch_count < 4 && t_idx < tensors.size()) {
      const auto& t = tensors[t_idx];
      t_idx++;

      if (t.numel() == 0) {
        // Update dim_offset for zero-element tensors before skipping
        batch_offset += cat_dim_size_of(t, dim);
        global_dim_offset = batch_offset;
        continue;
      }

      at::Tensor src = t;
      if (src.scalar_type() != out_dtype) {
        src = src.to(out_dtype);
      }
      src = src.contiguous();

      batch_tensors[batch_count] = src;
      dim_sizes[batch_count] = src.size(dim);
      dim_offsets[batch_count] = batch_offset;
      total_elements[batch_count] = src.numel();
      batch_offset += dim_sizes[batch_count];
      batch_count++;
    }
    global_dim_offset = batch_offset;

    if (batch_count == 0) {
      continue;
    }

    // Fill remaining slots with dummy data
    for (size_t j = batch_count; j < 4; ++j) {
      batch_tensors[j] = batch_tensors[0];
    }

    int64_t max_elements = 0;
    for (int j = 0; j < 4; ++j) {
      if (total_elements[j] > max_elements) {
        max_elements = total_elements[j];
      }
    }

    unsigned int grid_x = (max_elements + BLOCK - 1) / BLOCK;
    unsigned int grid_y = batch_count;

    kernel(raw_stream,
           grid_x,
           grid_y,
           1,
           NUM_WARPS,
           NUM_STAGES,
           out,
           batch_tensors[0],
           batch_tensors[1],
           batch_tensors[2],
           batch_tensors[3],
           dim_sizes[0],
           dim_sizes[1],
           dim_sizes[2],
           dim_sizes[3],
           dim_size_out,
           dim_prod_post,
           dim_offsets[0],
           dim_offsets[1],
           dim_offsets[2],
           dim_offsets[3],
           total_elements[0],
           total_elements[1],
           total_elements[2],
           total_elements[3],
           BLOCK);
  }
  return out;
}
}  // namespace flag_gems
