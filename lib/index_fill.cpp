#include <algorithm>

#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {

using namespace triton_jit;

namespace {

int64_t product_after_dim(const at::Tensor& input, int64_t dim) {
  int64_t result = 1;
  for (int64_t i = dim + 1; i < input.dim(); ++i) {
    result *= input.size(i);
  }
  return result;
}

int64_t product_before_dim(const at::Tensor& input, int64_t dim) {
  int64_t result = 1;
  for (int64_t i = 0; i < dim; ++i) {
    result *= input.size(i);
  }
  return result;
}

int64_t normalize_dim(int64_t dim, int64_t ndim) {
  if (dim < 0) dim += ndim;
  return dim;
}

}  // namespace

at::Tensor& index_fill_scalar_(at::Tensor& input,
                              int64_t dim,
                              const at::Tensor& index,
                              const c10::Scalar& value) {
  TORCH_CHECK(input.dim() > 0,
              "index_fill expects self to have at least one dimension");
  dim = normalize_dim(dim, input.dim());
  TORCH_CHECK(dim >= 0 && dim < input.dim(), "index_fill: dim out of range");
  TORCH_CHECK(index.scalar_type() == at::kLong,
              "index_fill_(): Expected dtype int64 for index.");
  TORCH_CHECK(index.dim() <= 1,
              "index_fill_(): Index is supposed to be a vector");
  TORCH_CHECK(index.device() == input.device(),
              "index and input must be on the same device");
  TORCH_CHECK(input.is_contiguous(),
              "C++ index_fill_ launcher only supports contiguous input");
  TORCH_CHECK(backend::isOnDevice(input),
              "input must be on the active backend device");

  const int64_t index_len = index.numel();
  if (input.numel() == 0 || index_len == 0) return input;

  const int64_t dim_size = input.size(dim);
  const int64_t inner_size = product_after_dim(input, dim);
  const int64_t outer_size = product_before_dim(input, dim);
  const int64_t outer_index_len = outer_size * index_len;

  constexpr int64_t BLOCK_SIZE = 512;
  int64_t block_n = 1;
  int64_t block_m = BLOCK_SIZE;
  if (inner_size >= 128) {
    block_n = std::min<int64_t>(128, utils::next_power_of_2(inner_size));
    block_m = 16;
  } else if (inner_size > 1) {
    block_n = std::min<int64_t>(64, utils::next_power_of_2(inner_size));
    if (inner_size > 4) {
      block_m = std::max<int64_t>(1, BLOCK_SIZE / block_n);
    }
  }

  const unsigned int grid_x =
      static_cast<unsigned int>((outer_index_len + block_m - 1) / block_m);
  const unsigned int grid_y =
      static_cast<unsigned int>((inner_size + block_n - 1) / block_n);

  static const TritonJITFunction& kernel = TritonJITFunction::get_instance(
      (utils::get_triton_src_path() / "index_fill.py").string(),
      "index_fill_contiguous_scalar_kernel");

  c10::DeviceGuard guard(input.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  kernel(raw_stream,
         grid_x,
         grid_y,
         1,
         4,
         0,
         input,
         index,
         value,
         outer_index_len,
         index_len,
         dim_size,
         inner_size,
         block_m,
         block_n);

  return input;
}

}  // namespace flag_gems
