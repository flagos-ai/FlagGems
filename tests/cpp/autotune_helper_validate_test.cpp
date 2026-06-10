// Phase 4.2 runtime test: exercise flag_gems::AutotunedCall::validate()
// against the synthetic kernel (`autotune_bridge_synth_kernel.py`, whose
// LibTuner `key=["M", "N"]`).
//
// Three cases:
//   1. expected_key_names = {"M", "N"}      -> validate() succeeds
//   2. expected_key_names = {"M", "K"}      -> validate() throws (name)
//   3. expected_key_names = {"M"}           -> validate() throws (arity)
//
// Embeds a Python interpreter via pybind11::scoped_interpreter so no
// existing TritonJITFunctionImpl is needed to bring Python up.

#include "utils/autotune_helper.h"

#include <pybind11/embed.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static const char* kKernelPath =
    "/workspace/FlagGems/tests/cpp/autotune_bridge_synth_kernel.py";

static void case_happy_path() {
  flag_gems::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});
  ac.validate();  // must not throw
  ac.validate();  // idempotent; second call is a no-op
}

static void case_name_mismatch() {
  flag_gems::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "K"});
  try {
    ac.validate();
  } catch (const std::runtime_error& e) {
    std::string what = e.what();
    assert(what.find("name mismatch") != std::string::npos);
    assert(what.find("position 1") != std::string::npos);  // 0-indexed; "N" vs "K"
    assert(what.find("\"K\"") != std::string::npos);
    assert(what.find("\"N\"") != std::string::npos);
    return;
  }
  throw std::runtime_error("case_name_mismatch: expected validate() to throw");
}

static void case_arity_mismatch() {
  flag_gems::AutotunedCall ac(kKernelPath, "synth_kernel", {"M"});
  try {
    ac.validate();
  } catch (const std::runtime_error& e) {
    std::string what = e.what();
    assert(what.find("arity mismatch") != std::string::npos);
    assert(what.find("declared 1") != std::string::npos);
    assert(what.find("has 2") != std::string::npos);
    return;
  }
  throw std::runtime_error("case_arity_mismatch: expected validate() to throw");
}

int main() {
  py::scoped_interpreter guard{};

  std::cout << "[1/3] happy path\n";
  case_happy_path();
  std::cout << "[2/3] name mismatch -> throws\n";
  case_name_mismatch();
  std::cout << "[3/3] arity mismatch -> throws\n";
  case_arity_mismatch();
  std::cout << "autotune_helper_validate_test: all 3 passed\n";
  return 0;
}
