// Phase 2 - Task C probe: verify whether libtriton_jit's
//   `fmt::format("{}", item)` for constexpr args produces output that matches
//   Python's `str()` output. If they diverge, the C++ kernel cache key will
//   never match what Python uses → silent cache miss.
//
// We are exercising the exact call site from
// libtriton_jit/include/triton_jit/triton_jit_function.h:177 :
//     signature.push_back(fmt::format("{}", item));
//
// The fmt version we link is 10.2.1 (FMT_VERSION 100201), same one FlagGems
// already pulled.

#include <fmt/format.h>
#include <cstdint>
#include <iostream>
#include <string>

template <typename T>
static std::string emit(const T& item) {
  return fmt::format("{}", item);
}

int main() {
  // bool — the headline risk: Python str(True) == "True"
  std::cout << "bool true       -> \"" << emit<bool>(true) << "\"\n";
  std::cout << "bool false      -> \"" << emit<bool>(false) << "\"\n";

  // int variants — should be unambiguous, but confirm anyway
  std::cout << "int 0           -> \"" << emit<int>(0) << "\"\n";
  std::cout << "int 1           -> \"" << emit<int>(1) << "\"\n";
  std::cout << "int -7          -> \"" << emit<int>(-7) << "\"\n";
  std::cout << "int64_t 128     -> \"" << emit<int64_t>(128) << "\"\n";
  std::cout << "uint64_t big    -> \"" << emit<uint64_t>(1ull << 40) << "\"\n";

  // float / double — Triton constexpr can in principle be float; check fmt's
  // default ("{}") behaviour vs Python str().
  std::cout << "float 1.5       -> \"" << emit<float>(1.5f) << "\"\n";
  std::cout << "double 1.5      -> \"" << emit<double>(1.5) << "\"\n";
  std::cout << "double 1.25e-3  -> \"" << emit<double>(1.25e-3) << "\"\n";

  // const char* — sanity, used for dtype names
  std::cout << "cstr \"fp16\"     -> \"" << emit<const char*>("fp16") << "\"\n";

  return 0;
}
