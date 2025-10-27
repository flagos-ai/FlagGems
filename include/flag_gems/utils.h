#pragma once
#include <ATen/Generator.h>
#include <c10/util/Optional.h>
#include <dlfcn.h>  // dladdr
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include "torch/torch.h"

namespace flag_gems::utils {

std::filesystem::path get_path_of_this_library();
std::filesystem::path get_triton_src_path();
std::filesystem::path get_flag_gems_src_path();
int64_t next_power_of_2(int64_t n);
bool broadcastable_to(at::IntArrayRef s1, at::IntArrayRef s2);
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(const at::Tensor &tensor,
                                                                      int reduction_axis);
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(
    const at::Tensor &tensor, at::OptionalIntArrayRef reduction_axes_opt);
std::tuple<int64_t, int64_t, int64_t> parse_reduction_axes(const at::Tensor &tensor, int reduction_axis);
int cdiv(int a, int b);
}  // namespace flag_gems::utils

// Public declaration for reuse (implementation in lib/exponential_.cpp)
namespace flag_gems {
std::pair<uint64_t, uint64_t> philox_backend_seed_offset(int64_t increment,
                                                         c10::optional<at::Generator> generator_opt);
}
