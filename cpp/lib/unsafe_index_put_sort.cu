/**
 * CUB radix sort wrapper for unsafe_index_put's sort-based accumulate path.
 *
 * CUB headers contain device intrinsics (threadIdx, __shfl_sync, ...) and
 * therefore must be compiled by nvcc; this translation unit isolates them
 * from the plain-C++ operator implementation in unsafe_index_put.cpp.
 */
#include "flag_gems/operators.h"

#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
#include <cub/device/device_radix_sort.cuh>

#include <cstdint>

namespace flag_gems {

void radix_sort_pairs_i64(const int64_t* keys_in,
                          int64_t* keys_out,
                          const int64_t* values_in,
                          int64_t* values_out,
                          int64_t num_items,
                          int begin_bit,
                          int end_bit,
                          void* temp_storage,
                          size_t& temp_storage_bytes,
                          uintptr_t stream) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (temp_storage == nullptr) {
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    temp_storage_bytes,
                                    keys_in,
                                    keys_out,
                                    values_in,
                                    values_out,
                                    num_items,
                                    begin_bit,
                                    end_bit,
                                    s);
    return;
  }
  cub::DeviceRadixSort::SortPairs(temp_storage,
                                  temp_storage_bytes,
                                  keys_in,
                                  keys_out,
                                  values_in,
                                  values_out,
                                  num_items,
                                  begin_bit,
                                  end_bit,
                                  s);
}

}  // namespace flag_gems
#endif  // FLAGGEMS_USE_CUDA || FLAGGEMS_USE_IX
