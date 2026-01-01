#include "flag_gems/shape_utils.h"

#include <algorithm>
#include <numeric>

#include "flag_gems/utils.h"

namespace flag_gems::shape_utils {
namespace {
  constexpr int64_t kMaxTileSize = 512;
}  // namespace

// Mirrors flag_gems/utils/shape_utils.py::heuristics_for_tile_size
std::vector<int64_t> heuristics_for_tile_size(c10::IntArrayRef shape) {
  const size_t ndim = shape.size();
  std::vector<int64_t> tile_sizes(ndim);
  int64_t max_tile_size = kMaxTileSize;

  for (size_t i = 0; i < ndim; ++i) {
    const size_t axis = ndim - 1 - i;
    const int64_t size = shape[axis];
    int64_t tile_size = std::min<int64_t>(max_tile_size, flag_gems::utils::next_power_of_2(size));
    tile_size = std::max<int64_t>(int64_t {1}, tile_size);
    tile_sizes[axis] = tile_size;
    max_tile_size = std::max<int64_t>(int64_t {1}, max_tile_size / tile_size);
  }
  return tile_sizes;
}

// Mirrors flag_gems/utils/shape_utils.py::heuristics_for_num_warps
int heuristics_for_num_warps_tile(int64_t tile_size) {
  if (tile_size < 2048) {
    return 4;
  }
  if (tile_size < 4096) {
    return 8;
  }
  return 16;
}

}  // namespace flag_gems::shape_utils
