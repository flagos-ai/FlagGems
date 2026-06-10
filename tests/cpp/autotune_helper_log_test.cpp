// Phase 4.5 runtime test: LIBTRITON_JIT_LOG_AUTOTUNE=1 observability.
//
// Four cases, all single-process (env flipped between cases via
// setenv/unsetenv; helper re-reads on each dispatch so the flips take
// effect immediately):
//   1. env unset    -> miss path runs, NO log line emitted
//   2. env=1        -> miss path runs, ONE log line with expected fields
//   3. env=1, hit   -> NO additional log (cache hit short-circuits
//                       dispatch entirely)
//   4. env=1, new K -> ONE additional log line, with the new key's values
//
// stderr is redirected into a stringstream for assertion; restored between
// cases so cerr output before/after the captured window goes to the real
// stderr.

#include "utils/autotune_helper.h"

#include <pybind11/embed.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace py = pybind11;
namespace fg = flag_gems;

static const char* kKernelPath = "/workspace/FlagGems/tests/cpp/autotune_bridge_synth_kernel.py";

static auto dummy_grid_fn = [](const triton_jit::Config&) -> std::tuple<unsigned, unsigned, unsigned> {
  return {1u, 1u, 1u};
};

static const triton_jit::Config& lookup_for(fg::AutotunedCall& ac, int64_t M, int64_t N) {
  return ac.lookup(fg::TuneKey {M, N}, dummy_grid_fn, int64_t {0}, int64_t {0}, M, N);
}

static size_t count_substr(const std::string& hay, const std::string& needle) {
  size_t n = 0, pos = 0;
  while ((pos = hay.find(needle, pos)) != std::string::npos) {
    ++n;
    pos += 1;
  }
  return n;
}

// Capture cerr for the duration of `body`, returning the captured text.
template <typename F>
static std::string capture_cerr(F&& body) {
  std::stringstream buf;
  std::streambuf* old = std::cerr.rdbuf(buf.rdbuf());
  body();
  std::cerr.rdbuf(old);
  return buf.str();
}

static void case_env_unset_no_log() {
  unsetenv("LIBTRITON_JIT_LOG_AUTOTUNE");
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});

  std::string out = capture_cerr([&]() { lookup_for(ac, 256, 128); });
  assert(out.empty());
}

static void case_env_on_miss_logs_once() {
  setenv("LIBTRITON_JIT_LOG_AUTOTUNE", "1", 1);
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});

  std::string out = capture_cerr([&]() { lookup_for(ac, 256, 128); });

  // Exactly one log line with all the expected fields.
  assert(count_substr(out, "[autotune] miss") == 1);
  assert(out.find("synth_kernel") != std::string::npos);
  assert(out.find("key=(M=256,N=128)") != std::string::npos);
  assert(out.find("num_warps=4") != std::string::npos);
  assert(out.find("num_stages=2") != std::string::npos);
  assert(out.find("TILE_M=128") != std::string::npos);
  assert(out.find("TILE_N=64") != std::string::npos);
  assert(out.find("GROUP_M=8") != std::string::npos);
  // bool formatted as "true"/"false", not "1"/"0"
  assert(out.find("DIVISIBLE_M=true") != std::string::npos);
  assert(out.find("DIVISIBLE_N=true") != std::string::npos);
  assert(out.find("IS_FP64=false") != std::string::npos);
}

static void case_hit_does_not_log() {
  setenv("LIBTRITON_JIT_LOG_AUTOTUNE", "1", 1);
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});

  // Prime the cache (log line discarded — outside the capture).
  lookup_for(ac, 256, 128);
  assert(ac.cache_size() == 1);

  std::string out = capture_cerr([&]() {
    lookup_for(ac, 256, 128);  // hit
  });
  assert(out.empty());
}

static void case_new_key_emits_second_log() {
  setenv("LIBTRITON_JIT_LOG_AUTOTUNE", "1", 1);
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});

  lookup_for(ac, 256, 128);  // first miss (uncaptured)

  std::string out = capture_cerr([&]() {
    lookup_for(ac, 200, 128);  // second miss, new key
  });
  assert(count_substr(out, "[autotune] miss") == 1);
  assert(out.find("key=(M=200,N=128)") != std::string::npos);
  // Heuristic flipped: 200 % 128 != 0 -> DIVISIBLE_M=false
  assert(out.find("DIVISIBLE_M=false") != std::string::npos);
}

int main() {
  py::scoped_interpreter guard {};

  std::cout << "[1/4] env unset -> no log\n";
  case_env_unset_no_log();
  std::cout << "[2/4] env=1 + miss -> one log line, all fields present\n";
  case_env_on_miss_logs_once();
  std::cout << "[3/4] env=1 + hit -> no log (cache short-circuits dispatch)\n";
  case_hit_does_not_log();
  std::cout << "[4/4] env=1 + new key -> additional log, heuristic flipped\n";
  case_new_key_emits_second_log();
  std::cout << "autotune_helper_log_test: all 4 passed\n";
  return 0;
}
