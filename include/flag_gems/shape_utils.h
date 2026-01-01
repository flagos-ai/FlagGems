#pragma once

#include <c10/util/ArrayRef.h>
#include <cstdint>
#include <vector>

namespace flag_gems::shape_utils {

/**
 * Mirrors flag_gems/utils/shape_utils.py::heuristics_for_tile_size.
 * Chooses per-axis tile sizes (power-of-two capped at 512 total product)
 * starting from the innermost dimension.
 */
std::vector<int64_t> heuristics_for_tile_size(c10::IntArrayRef shape);

/**
 * Mirrors flag_gems/utils/shape_utils.py::heuristics_for_num_warps.
 * Returns the number of warps to launch for a given tile size.
 */
int heuristics_for_num_warps_tile(int64_t tile_size);

}  // namespace flag_gems::shape_utils
