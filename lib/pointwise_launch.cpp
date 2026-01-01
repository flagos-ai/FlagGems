#include "flag_gems/pointwise_launch.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "flag_gems/shape_utils.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {

using namespace triton_jit;

namespace {

  std::vector<int64_t> compute_stride_order(c10::IntArrayRef strides) {
    std::vector<int64_t> order(strides.size());
    std::iota(order.begin(), order.end(), int64_t {0});
    std::sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
      auto lhs_abs = std::llabs(strides[lhs]);
      auto rhs_abs = std::llabs(strides[rhs]);
      return lhs_abs < rhs_abs;
    });
    return order;
  }

  struct LaunchConfig {
    unsigned int grid_x = 1;
    unsigned int grid_y = 1;
    unsigned int grid_z = 1;
    int num_warps = 4;
    int num_stages = 1;
    int64_t tiles_per_cta = 1;
    bool one_tile_per_cta = true;
    std::vector<int64_t> tile_sizes;
  };

  LaunchConfig make_launch_config(c10::IntArrayRef shape) {
    TORCH_CHECK(!shape.empty(), "Expected non-empty shape for launch config");

    auto tile_sizes = flag_gems::shape_utils::heuristics_for_tile_size(shape);
    const int64_t tile_product =
        std::accumulate(tile_sizes.begin(), tile_sizes.end(), int64_t {1}, std::multiplies<int64_t>());

    int64_t num_tiles = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      num_tiles *= flag_gems::utils::cdiv(shape[i], tile_sizes[i]);
    }

    TORCH_CHECK(num_tiles > 0, "Invalid tile configuration: zero tiles");
    TORCH_CHECK(num_tiles <= std::numeric_limits<unsigned int>::max(), "Too many tiles for CUDA grid");
    const unsigned int grid_x = static_cast<unsigned int>(num_tiles);

    const int num_warps = flag_gems::shape_utils::heuristics_for_num_warps_tile(tile_product);
    const int64_t tiles_per_cta =
        std::max<int64_t>(int64_t {1}, flag_gems::utils::cdiv(num_tiles, static_cast<int64_t>(grid_x)));

    LaunchConfig config;
    config.grid_x = grid_x;
    config.num_warps = num_warps;
    config.tiles_per_cta = tiles_per_cta;
    config.one_tile_per_cta = tiles_per_cta == 1;
    config.tile_sizes = std::move(tile_sizes);
    return config;
  }

  std::string make_kernel_name(std::string_view prefix, int64_t rank) {
    std::string name(prefix);
    name.append("_rank_");
    name.append(std::to_string(rank));
    return name;
  }

}  // namespace

bool launch_pointwise_unary_rank0_to2(const at::Tensor &src,
                                      at::Tensor &dst,
                                      const int64_t rank,
                                      CUstream raw_stream,
                                      std::string_view triton_module_path,
                                      std::string_view kernel_name_prefix) {
  TORCH_CHECK(src.device() == dst.device(),
              "launch_pointwise_unary_rank0_to2 expects tensors on same device");
  TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "Input/output dtype mismatch");
  TORCH_CHECK(src.sizes() == dst.sizes(), "Input/output shape mismatch");

  if (rank < 0 || rank > 2) {
    return false;
  }

  const std::string kernel_name = make_kernel_name(kernel_name_prefix, rank);
  const triton_jit::TritonJITFunction &kernel =
      triton_jit::TritonJITFunction::get_instance(triton_module_path, kernel_name);

  if (rank == 0) {
    kernel(raw_stream, 1, 1, 1, 4, 1, src, dst);
    return true;
  }

  LaunchConfig config = make_launch_config(dst.sizes());
  auto in_stride_order = compute_stride_order(src.strides());
  auto out_stride_order = compute_stride_order(dst.strides());

  if (rank == 1) {
    kernel(raw_stream,
           config.grid_x,
           config.grid_y,
           config.grid_z,
           config.num_warps,
           config.num_stages,
           src,
           dst,
           src.stride(0),
           in_stride_order[0],
           dst.stride(0),
           out_stride_order[0],
           dst.size(0),
           dst.numel(),
           config.tiles_per_cta,
           config.tile_sizes[0],
           config.one_tile_per_cta);
    return true;
  }

  if (rank == 2) {
    kernel(raw_stream,
           config.grid_x,
           config.grid_y,
           config.grid_z,
           config.num_warps,
           config.num_stages,
           src,
           dst,
           src.stride(0),
           src.stride(1),
           in_stride_order[0],
           in_stride_order[1],
           dst.stride(0),
           dst.stride(1),
           out_stride_order[0],
           out_stride_order[1],
           dst.size(0),
           dst.size(1),
           dst.numel(),
           config.tiles_per_cta,
           config.tile_sizes[0],
           config.tile_sizes[1],
           config.one_tile_per_cta);
    return true;
  }

  return false;
}

}  // namespace flag_gems
