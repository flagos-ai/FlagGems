#include "flag_gems/utils.h"

namespace flag_gems::utils {

std::filesystem::path get_path_of_this_library() {
  // This function gives the library path of this library as runtime, similar to the $ORIGIN
  // that is used for run path (RPATH), but unfortunately, for custom dependencies (instead of linking)
  // there is no build system generator to take care of this.
  static const std::filesystem::path cached_path = []() {
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void*>(&get_path_of_this_library), &dl_info) && dl_info.dli_fname) {
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
    const char* flag_gems_dir = std::getenv("FLAGGEMS_SOURCE_DIR");
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
}  // namespace flag_gems::utils
