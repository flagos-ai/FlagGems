#include "flag_gems/utils.h"

namespace flag_gems::utils {

std::filesystem::path get_path_of_this_library() {
  // This function gives the library path of this library as runtime, similar to the $ORIGIN
  // that is used for run path (RPATH), but unfortunately, for custom dependencies (instead of linking)
  // there is no build system generator to take care of this.
  static const std::filesystem::path cached_path = []() {
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void *>(&get_path_of_this_library), &dl_info) && dl_info.dli_fname) {
      return std::filesystem::canonical(dl_info.dli_fname);  // Ensure absolute, resolved path
    } else {
      throw std::runtime_error("cannot get the path of libjit_utils.so");
    }
  }();
  return cached_path;
}

std::filesystem::path get_triton_src_path() {
  const static std::filesystem::path triton_src_dir = []() {
    std::filesystem::path installed_script_path =
        get_path_of_this_library().parent_path().parent_path() / "share" / "flag_gems" / "triton_src";
    if (std::filesystem::exists(installed_script_path)) {
      return installed_script_path;
    } else {
      std::filesystem::path source_script_path =
          std::filesystem::path(__FILE__).parent_path().parent_path() / "triton_src";
      return source_script_path;
    }
  }();
  return triton_src_dir;
}

std::filesystem::path get_flag_gems_src_path() {
  const static std::filesystem::path flag_gems_src_dir = []() {
    const char *flag_gems_dir = std::getenv("FLAGGEMS_SOURCE_DIR");
    if (!flag_gems_dir) {
      throw std::runtime_error("Environment variable FLAGGEMS_SOURCE_DIR not set");
    }
    return std::filesystem::path(flag_gems_dir);
  }();
  return flag_gems_src_dir;
}

int64_t next_power_of_2(int64_t n) {
  if (n <= 1) return 1;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

bool broadcastable_to(at::IntArrayRef s1, at::IntArrayRef s2) {
  size_t r1 = s1.size();
  if (r1 == 0) {
    return true;
  }

  size_t r2 = s2.size();
  if (r2 == 0) {
    return false;
  }

  if (r1 > r2) {
    return false;
  }

  size_t d = r2 - r1;
  for (size_t i = 0; i < r1; ++i) {
    if (s1[i] == 1 || s1[i] == s2[d + i]) {
      continue;
    }
    return false;
  }

  return true;
}

std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(
    const at::Tensor &tensor, at::OptionalIntArrayRef reduction_axes_opt) {
  int64_t dim = tensor.dim();
  c10::DimVector reduction_axes;

  if (reduction_axes_opt.has_value()) {
    reduction_axes = reduction_axes_opt.value().vec();
  }

  std::unordered_set<int64_t> reduction_set(reduction_axes.begin(), reduction_axes.end());

  c10::DimVector left_axes, right_axes;
  int64_t non_reduction_size = 1, reduction_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (reduction_set.count(i)) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    }
  }

  // Concatenate left and right axes to form the new permutation order
  c10::DimVector permute_order = left_axes;
  permute_order.insert(permute_order.end(), right_axes.begin(), right_axes.end());

  return {tensor.permute(permute_order), non_reduction_size, reduction_size};
}
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(const at::Tensor &tensor,
                                                                      int reduction_axis) {
  int64_t dim = tensor.dim();
  c10::DimVector left_axes, right_axes;
  int64_t non_reduction_size = 1, reduction_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (i == reduction_axis) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    }
  }
  c10::DimVector permute_order = left_axes;
  permute_order.insert(permute_order.end(), right_axes.begin(), right_axes.end());

  return {tensor.permute(permute_order), non_reduction_size, reduction_size};
}

std::tuple<int64_t, int64_t, int64_t> parse_reduction_axes(const at::Tensor &tensor, int reduction_axis) {
  int64_t dim = tensor.dim();
  c10::DimVector left_axes, right_axes, remain_axes;
  int64_t non_reduction_size = 1;
  int64_t reduction_size = 1;
  int64_t remain_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (i < reduction_axis) {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    } else if (i == reduction_axis) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      remain_axes.push_back(i);
      remain_size *= tensor.size(i);
    }
  }
  return {non_reduction_size, reduction_size, remain_size};
}
int cdiv(int a, int b) {
  return (a + b - 1) / b;
}
}  // namespace flag_gems::utils
