#include <ATen/MemoryOverlap.h>
#include <csrc/core/MUSAGuard.h>
#include <csrc/core/MUSAStream.h>
#include <torch/extension.h>
#include <algorithm>
#include <climits>
#include <vector>
extern "C" void launch_scatter(
    unsigned short*, const long long*, const unsigned short*, int, int, int, int, musaStream_t);
extern "C" void launch_scatter_f32(float*, const long long*, const float*, int, int, int, int, musaStream_t);
extern "C" void launch_row_clone_scatter_f16(unsigned short*,
                                             const unsigned short*,
                                             const long long*,
                                             const unsigned short*,
                                             int,
                                             int,
                                             int,
                                             bool,
                                             musaStream_t);
extern "C" void launch_row_clone_scatter_bf16(unsigned short*,
                                              const unsigned short*,
                                              const long long*,
                                              const unsigned short*,
                                              int,
                                              int,
                                              int,
                                              bool,
                                              musaStream_t);
extern "C" void launch_row_clone_scatter_f32(
    float*, const float*, const long long*, const float*, int, int, int, bool, musaStream_t);
struct Offset3 {
  long long s0, s1, s2;
};
extern "C" void launch_fullrank_u16(unsigned short*,
                                    const long long*,
                                    const long long*,
                                    const long long*,
                                    const unsigned short*,
                                    long long,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    Offset3,
                                    Offset3,
                                    Offset3,
                                    Offset3,
                                    Offset3,
                                    musaStream_t);
extern "C" void launch_fullrank_f32(float*,
                                    const long long*,
                                    const long long*,
                                    const long long*,
                                    const float*,
                                    long long,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    Offset3,
                                    Offset3,
                                    Offset3,
                                    Offset3,
                                    Offset3,
                                    musaStream_t);
extern "C" void launch_fullrank_atomic_u16(unsigned short*,
                                           const long long*,
                                           const long long*,
                                           const long long*,
                                           const unsigned short*,
                                           long long,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           Offset3,
                                           Offset3,
                                           Offset3,
                                           Offset3,
                                           Offset3,
                                           int,
                                           musaStream_t);
extern "C" void launch_fullrank_atomic_f32(float*,
                                           const long long*,
                                           const long long*,
                                           const long long*,
                                           const float*,
                                           long long,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           Offset3,
                                           Offset3,
                                           Offset3,
                                           Offset3,
                                           Offset3,
                                           musaStream_t);
extern "C" void launch_atomic_f32(float*, const long long*, const float*, long long, long long, musaStream_t);
extern "C" void launch_atomic_f16(
    unsigned short*, const long long*, const unsigned short*, long long, long long, musaStream_t);
extern "C" void launch_bool_mask_prepare(const bool*, int*, int*, long long, int, musaStream_t);
extern "C" void launch_bool_mask_f16(unsigned short*,
                                     const bool*,
                                     const unsigned short*,
                                     const int*,
                                     long long,
                                     long long,
                                     int,
                                     bool,
                                     musaStream_t);
extern "C" void launch_bool_mask_bf16(unsigned short*,
                                      const bool*,
                                      const unsigned short*,
                                      const int*,
                                      long long,
                                      long long,
                                      int,
                                      bool,
                                      musaStream_t);
extern "C" void launch_bool_mask_f32(
    float*, const bool*, const float*, const int*, long long, long long, int, bool, musaStream_t);

static Offset3 make_offset3(const torch::Tensor& tensor, int rank) {
  Offset3 result {0, 0, 0};
  if (rank > 0) result.s0 = tensor.stride(0);
  if (rank > 1) result.s1 = tensor.stride(1);
  if (rank > 2) result.s2 = tensor.stride(2);
  return result;
}

torch::Tensor native_index_put_(torch::Tensor input, torch::Tensor indices, torch::Tensor values) {
  TORCH_CHECK(input.is_musa() && indices.is_musa() && values.is_musa());
  TORCH_CHECK(input.device() == indices.device() && input.device() == values.device());
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16 ||
              input.scalar_type() == torch::kFloat32);
  TORCH_CHECK(values.scalar_type() == input.scalar_type());
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous());
  TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == torch::kLong && indices.is_contiguous());
  TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && indices.numel() > 0);
  TORCH_CHECK(input.size(0) <= INT_MAX && input.size(1) <= INT_MAX && indices.numel() <= 65535);
  TORCH_CHECK(at::get_overlap_status(input, values) == at::MemOverlapStatus::No,
              "native index_put does not support overlapping values and input");
  TORCH_CHECK(at::get_overlap_status(input, indices) == at::MemOverlapStatus::No,
              "native index_put does not support overlapping indices and input");
  TORCH_CHECK(values.is_contiguous() &&
              ((values.dim() == 2 && values.size(0) == indices.numel() && values.size(1) == input.size(1)) ||
               (values.dim() == 1 && values.size(0) == input.size(1))));
  const c10::musa::MUSAGuard device_guard(input.device());
  auto s = c10::musa::getCurrentMUSAStream(input.device().index()).stream();
  if (input.scalar_type() == torch::kFloat32)
    launch_scatter_f32((float*)input.data_ptr(),
                       (const long long*)indices.data_ptr(),
                       (const float*)values.data_ptr(),
                       indices.numel(),
                       input.size(0),
                       input.size(1),
                       values.dim() == 1,
                       s);
  else
    launch_scatter((unsigned short*)input.data_ptr(),
                   (const long long*)indices.data_ptr(),
                   (const unsigned short*)values.data_ptr(),
                   indices.numel(),
                   input.size(0),
                   input.size(1),
                   values.dim() == 1,
                   s);
  TORCH_CHECK(musaGetLastError() == musaSuccess, "native index_put kernel launch failed");
  return input;
}

pybind11::object try_native_index_put_row_clone_(torch::Tensor input,
                                                 torch::Tensor indices,
                                                 torch::Tensor values) {
  if (!input.is_musa() || !indices.is_musa() || !values.is_musa() || input.device() != indices.device() ||
      input.device() != values.device() || input.dim() != 2 || !input.is_contiguous() || indices.dim() < 1 ||
      indices.dim() > 3 || indices.scalar_type() != torch::kLong || !indices.is_contiguous() ||
      indices.numel() <= 0 || indices.numel() > 256 || input.size(0) <= 0 || input.size(0) > 65535 ||
      input.size(1) <= 0 || input.size(1) > INT_MAX || input.numel() > 4096 ||
      input.scalar_type() != values.scalar_type() || !values.is_contiguous() || values.requires_grad() ||
      input.requires_grad() ||
      (input.scalar_type() != torch::kFloat16 && input.scalar_type() != torch::kBFloat16 &&
       input.scalar_type() != torch::kFloat32)) {
    return pybind11::none();
  }
  bool broadcast_values = values.dim() == 1;
  if (!((broadcast_values && values.size(0) == input.size(1)) ||
        (values.dim() == 2 && values.size(0) == indices.numel() && values.size(1) == input.size(1)))) {
    return pybind11::none();
  }
  const c10::musa::MUSAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  auto stream = c10::musa::getCurrentMUSAStream(input.device().index()).stream();
  auto index_ptr = reinterpret_cast<const long long*>(indices.data_ptr());
  if (input.scalar_type() == torch::kFloat32) {
    launch_row_clone_scatter_f32(reinterpret_cast<float*>(output.data_ptr()),
                                 reinterpret_cast<const float*>(input.data_ptr()),
                                 index_ptr,
                                 reinterpret_cast<const float*>(values.data_ptr()),
                                 indices.numel(),
                                 input.size(0),
                                 input.size(1),
                                 broadcast_values,
                                 stream);
  } else if (input.scalar_type() == torch::kFloat16) {
    launch_row_clone_scatter_f16(reinterpret_cast<unsigned short*>(output.data_ptr()),
                                 reinterpret_cast<const unsigned short*>(input.data_ptr()),
                                 index_ptr,
                                 reinterpret_cast<const unsigned short*>(values.data_ptr()),
                                 indices.numel(),
                                 input.size(0),
                                 input.size(1),
                                 broadcast_values,
                                 stream);
  } else {
    launch_row_clone_scatter_bf16(reinterpret_cast<unsigned short*>(output.data_ptr()),
                                  reinterpret_cast<const unsigned short*>(input.data_ptr()),
                                  index_ptr,
                                  reinterpret_cast<const unsigned short*>(values.data_ptr()),
                                  indices.numel(),
                                  input.size(0),
                                  input.size(1),
                                  broadcast_values,
                                  stream);
  }
  TORCH_CHECK(musaGetLastError() == musaSuccess, "native fused row Clone+Scatter kernel launch failed");
  return pybind11::cast(output);
}

pybind11::object try_native_index_put_fullrank_(torch::Tensor input,
                                                std::vector<torch::Tensor> indices,
                                                torch::Tensor values,
                                                bool inplace,
                                                bool accumulate) {
  if (!input.is_musa() || !values.is_musa() || input.device() != values.device() || input.dim() < 1 ||
      input.dim() > 3 || !input.is_contiguous() || indices.size() != static_cast<size_t>(input.dim()) ||
      (input.scalar_type() != torch::kFloat16 && input.scalar_type() != torch::kBFloat16 &&
       input.scalar_type() != torch::kFloat32) ||
      values.scalar_type() != input.scalar_type() || input.requires_grad() || values.requires_grad() ||
      !values.is_contiguous()) {
    return pybind11::none();
  }
  if (accumulate && input.scalar_type() == torch::kBFloat16) {
    return pybind11::none();
  }
  if (inplace && at::get_overlap_status(input, values) != at::MemOverlapStatus::No) {
    return pybind11::none();
  }

  int broadcast_rank = 0;
  for (const auto& index : indices) {
    if (!index.is_musa() || index.device() != input.device() || index.scalar_type() != torch::kLong ||
        index.dim() < 1 || index.dim() > 3 || !index.is_contiguous()) {
      return pybind11::none();
    }
    if (inplace && at::get_overlap_status(input, index) != at::MemOverlapStatus::No) {
      return pybind11::none();
    }
    broadcast_rank = std::max(broadcast_rank, static_cast<int>(index.dim()));
  }
  if (broadcast_rank < 1 || broadcast_rank > 3) return pybind11::none();

  std::vector<int64_t> broadcast_shape(broadcast_rank, 1);
  for (const auto& index : indices) {
    int shift = broadcast_rank - static_cast<int>(index.dim());
    for (int d = 0; d < broadcast_rank; ++d) {
      int64_t size = d < shift ? 1 : index.size(d - shift);
      if (size != 1 && broadcast_shape[d] != 1 && broadcast_shape[d] != size) {
        return pybind11::none();
      }
      broadcast_shape[d] = std::max(broadcast_shape[d], size);
    }
  }
  if (values.dim() > broadcast_rank) return pybind11::none();
  int value_shift = broadcast_rank - static_cast<int>(values.dim());
  for (int d = 0; d < broadcast_rank; ++d) {
    int64_t size = d < value_shift ? 1 : values.size(d - value_shift);
    if (size != 1 && size != broadcast_shape[d]) return pybind11::none();
  }
  int64_t count = 1;
  for (int64_t size : broadcast_shape) {
    if (size <= 0 || size > INT_MAX || count > (65535LL * 256) / size) {
      return pybind11::none();
    }
    count *= size;
  }
  for (int d = 0; d < input.dim(); ++d) {
    if (input.size(d) <= 0 || input.size(d) > INT_MAX) return pybind11::none();
  }

  const c10::musa::MUSAGuard device_guard(input.device());
  // For out-of-place updates, preserve possible input/value/index aliases by
  // cloning the source before the scatter on the same current MUSA stream.
  auto output = inplace ? input : input.clone();

  std::vector<torch::Tensor> expanded_indices;
  expanded_indices.reserve(indices.size());
  for (const auto& index : indices) {
    expanded_indices.push_back(index.expand(broadcast_shape));
  }
  auto expanded_values = values.expand(broadcast_shape);
  const long long* index_ptrs[3] = {nullptr, nullptr, nullptr};
  Offset3 index_strides[3] = {
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0}
  };
  for (size_t i = 0; i < expanded_indices.size(); ++i) {
    index_ptrs[i] = reinterpret_cast<const long long*>(expanded_indices[i].data_ptr());
    index_strides[i] = make_offset3(expanded_indices[i], broadcast_rank);
  }
  Offset3 value_strides = make_offset3(expanded_values, broadcast_rank);
  Offset3 input_strides {0, 0, 0};
  for (int d = 0; d < output.dim(); ++d) {
    if (d == 0) input_strides.s0 = output.stride(d);
    if (d == 1) input_strides.s1 = output.stride(d);
    if (d == 2) input_strides.s2 = output.stride(d);
  }
  int shape0 = broadcast_shape[0];
  int shape1 = broadcast_rank > 1 ? broadcast_shape[1] : 1;
  int shape2 = broadcast_rank > 2 ? broadcast_shape[2] : 1;
  int dim0 = output.size(0);
  int dim1 = output.dim() > 1 ? output.size(1) : 1;
  int dim2 = output.dim() > 2 ? output.size(2) : 1;
  auto stream = c10::musa::getCurrentMUSAStream(output.device().index()).stream();
  if (accumulate && output.scalar_type() == torch::kFloat32) {
    launch_fullrank_atomic_f32(reinterpret_cast<float*>(output.data_ptr()),
                               index_ptrs[0],
                               index_ptrs[1],
                               index_ptrs[2],
                               reinterpret_cast<const float*>(expanded_values.data_ptr()),
                               count,
                               broadcast_rank,
                               shape0,
                               shape1,
                               shape2,
                               input.dim(),
                               dim0,
                               dim1,
                               dim2,
                               index_strides[0],
                               index_strides[1],
                               index_strides[2],
                               value_strides,
                               input_strides,
                               stream);
  } else if (accumulate && output.scalar_type() == torch::kFloat16) {
    launch_fullrank_atomic_u16(reinterpret_cast<unsigned short*>(output.data_ptr()),
                               index_ptrs[0],
                               index_ptrs[1],
                               index_ptrs[2],
                               reinterpret_cast<const unsigned short*>(expanded_values.data_ptr()),
                               count,
                               broadcast_rank,
                               shape0,
                               shape1,
                               shape2,
                               input.dim(),
                               dim0,
                               dim1,
                               dim2,
                               index_strides[0],
                               index_strides[1],
                               index_strides[2],
                               value_strides,
                               input_strides,
                               0,
                               stream);
  } else if (output.scalar_type() == torch::kFloat32) {
    launch_fullrank_f32(reinterpret_cast<float*>(output.data_ptr()),
                        index_ptrs[0],
                        index_ptrs[1],
                        index_ptrs[2],
                        reinterpret_cast<const float*>(expanded_values.data_ptr()),
                        count,
                        broadcast_rank,
                        shape0,
                        shape1,
                        shape2,
                        input.dim(),
                        dim0,
                        dim1,
                        dim2,
                        index_strides[0],
                        index_strides[1],
                        index_strides[2],
                        value_strides,
                        input_strides,
                        stream);
  } else {
    launch_fullrank_u16(reinterpret_cast<unsigned short*>(output.data_ptr()),
                        index_ptrs[0],
                        index_ptrs[1],
                        index_ptrs[2],
                        reinterpret_cast<const unsigned short*>(expanded_values.data_ptr()),
                        count,
                        broadcast_rank,
                        shape0,
                        shape1,
                        shape2,
                        input.dim(),
                        dim0,
                        dim1,
                        dim2,
                        index_strides[0],
                        index_strides[1],
                        index_strides[2],
                        value_strides,
                        input_strides,
                        stream);
  }
  TORCH_CHECK(musaGetLastError() == musaSuccess, "native full-rank index_put kernel launch failed");
  return pybind11::cast(output);
}

pybind11::object try_native_index_put_accumulate_(torch::Tensor input,
                                                  torch::Tensor indices,
                                                  torch::Tensor values,
                                                  bool inplace) {
  if (!input.is_musa() || !indices.is_musa() || !values.is_musa() || input.device() != indices.device() ||
      input.device() != values.device() || input.dim() != 1 || !input.is_contiguous() || input.size(0) <= 0 ||
      (input.scalar_type() != torch::kFloat16 && input.scalar_type() != torch::kFloat32) ||
      values.scalar_type() != input.scalar_type() || indices.scalar_type() != torch::kLong ||
      indices.dim() != 1 || !indices.is_contiguous() || !values.is_contiguous() || values.dim() != 1 ||
      values.numel() != indices.numel() || indices.numel() <= 0 || indices.numel() > 65535LL * 256 ||
      input.requires_grad() || values.requires_grad()) {
    return pybind11::none();
  }
  if (inplace && (at::get_overlap_status(input, values) != at::MemOverlapStatus::No ||
                  at::get_overlap_status(input, indices) != at::MemOverlapStatus::No)) {
    return pybind11::none();
  }

  const c10::musa::MUSAGuard device_guard(input.device());
  auto output = inplace ? input : input.clone();
  auto stream = c10::musa::getCurrentMUSAStream(output.device().index()).stream();
  if (output.scalar_type() == torch::kFloat32) {
    launch_atomic_f32(reinterpret_cast<float*>(output.data_ptr()),
                      reinterpret_cast<const long long*>(indices.data_ptr()),
                      reinterpret_cast<const float*>(values.data_ptr()),
                      indices.numel(),
                      input.numel(),
                      stream);
  } else {
    launch_atomic_f16(reinterpret_cast<unsigned short*>(output.data_ptr()),
                      reinterpret_cast<const long long*>(indices.data_ptr()),
                      reinterpret_cast<const unsigned short*>(values.data_ptr()),
                      indices.numel(),
                      input.numel(),
                      stream);
  }
  TORCH_CHECK(musaGetLastError() == musaSuccess, "native atomic index_put kernel launch failed");
  return pybind11::cast(output);
}

pybind11::object try_native_index_put_bool_(
    torch::Tensor input, torch::Tensor mask, torch::Tensor values, bool inplace, bool accumulate) {
  if (!input.is_musa() || !mask.is_musa() || !values.is_musa() || input.device() != mask.device() ||
      input.device() != values.device() || input.dim() < 1 || input.dim() > 3 || !input.is_contiguous() ||
      !mask.is_contiguous() || mask.scalar_type() != torch::kBool || mask.dim() != input.dim() ||
      (input.scalar_type() != torch::kFloat16 && input.scalar_type() != torch::kBFloat16 &&
       input.scalar_type() != torch::kFloat32) ||
      values.scalar_type() != input.scalar_type() || values.dim() != 1 || !values.is_contiguous() ||
      input.requires_grad() || values.requires_grad() || input.numel() <= 0 ||
      input.numel() > (accumulate ? 262144 : 4096) || values.numel() > input.numel()) {
    return pybind11::none();
  }
  for (int64_t d = 0; d < input.dim(); ++d) {
    if (mask.size(d) != input.size(d)) return pybind11::none();
  }
  if (inplace && (at::get_overlap_status(input, mask) != at::MemOverlapStatus::No ||
                  at::get_overlap_status(input, values) != at::MemOverlapStatus::No)) {
    return pybind11::none();
  }

  const c10::musa::MUSAGuard device_guard(input.device());
  auto output = inplace ? input : input.clone();
  const int blocks = static_cast<int>((input.numel() + 255) / 256);
  auto scratch_options = input.options().dtype(torch::kInt32);
  auto block_counts = torch::empty({blocks}, scratch_options);
  // The final element stores the total True count, used to validate the
  // dynamic boolean-index Values length before any in-place write occurs.
  auto block_offsets = torch::empty({blocks + 1}, scratch_options);
  auto stream = c10::musa::getCurrentMUSAStream(input.device().index()).stream();
  launch_bool_mask_prepare(reinterpret_cast<const bool*>(mask.data_ptr()),
                           reinterpret_cast<int*>(block_counts.data_ptr()),
                           reinterpret_cast<int*>(block_offsets.data_ptr()),
                           input.numel(),
                           blocks,
                           stream);
  TORCH_CHECK(musaGetLastError() == musaSuccess, "native boolean index_put validation launch failed");
  const int64_t selected = block_offsets.select(0, blocks).item<int>();
  TORCH_CHECK(selected == values.numel(),
              "shape mismatch: value tensor length must match the number of true mask entries");
  // Scratch Tensor references remain live through the launch. Their storage
  // is released on the same current stream when this function returns.
  if (input.scalar_type() == torch::kFloat32) {
    launch_bool_mask_f32(reinterpret_cast<float*>(output.data_ptr()),
                         reinterpret_cast<const bool*>(mask.data_ptr()),
                         reinterpret_cast<const float*>(values.data_ptr()),
                         reinterpret_cast<const int*>(block_offsets.data_ptr()),
                         input.numel(),
                         values.numel(),
                         blocks,
                         accumulate,
                         stream);
  } else if (input.scalar_type() == torch::kFloat16) {
    launch_bool_mask_f16(reinterpret_cast<unsigned short*>(output.data_ptr()),
                         reinterpret_cast<const bool*>(mask.data_ptr()),
                         reinterpret_cast<const unsigned short*>(values.data_ptr()),
                         reinterpret_cast<const int*>(block_offsets.data_ptr()),
                         input.numel(),
                         values.numel(),
                         blocks,
                         accumulate,
                         stream);
  } else {
    launch_bool_mask_bf16(reinterpret_cast<unsigned short*>(output.data_ptr()),
                          reinterpret_cast<const bool*>(mask.data_ptr()),
                          reinterpret_cast<const unsigned short*>(values.data_ptr()),
                          reinterpret_cast<const int*>(block_offsets.data_ptr()),
                          input.numel(),
                          values.numel(),
                          blocks,
                          accumulate,
                          stream);
  }
  TORCH_CHECK(musaGetLastError() == musaSuccess, "native boolean index_put scatter launch failed");
  return pybind11::cast(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("native_index_put_", &native_index_put_);
  m.def("try_native_index_put_row_clone_", &try_native_index_put_row_clone_);
  m.def("try_native_index_put_fullrank_",
        &try_native_index_put_fullrank_,
        pybind11::arg("input"),
        pybind11::arg("indices"),
        pybind11::arg("values"),
        pybind11::arg("inplace"),
        pybind11::arg("accumulate") = false);
  m.def("try_native_index_put_accumulate_", &try_native_index_put_accumulate_);
  m.def("try_native_index_put_bool_", &try_native_index_put_bool_);
}
