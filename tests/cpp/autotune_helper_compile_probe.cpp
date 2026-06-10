// Phase 4.2 compile probe: forces parsing of flag_gems::AutotunedCall's
// non-template members. No instantiation tricks needed (the class itself
// is not a template), so a single TU including the header and constructing
// it gets full type-check coverage. Compile-only (`-c`), no link / no run.

#include "utils/autotune_helper.h"

#include <string>
#include <vector>

void force_parse() {
  flag_gems::AutotunedCall ac(
      "/dev/null",
      "dummy",
      std::vector<std::string>{"M", "N"});
  // Accessors compile-check
  (void)ac.kernel_path();
  (void)ac.function_name();
  (void)ac.expected_key_names();
  (void)ac.cache_size();

  // Hash functor compile-check
  flag_gems::TuneKey k = {1, 2, 3};
  flag_gems::TuneKeyHash hasher;
  (void)hasher(k);

  // Force instantiation of lookup_or_compute with a lambda.
  const auto& cfg = ac.lookup_or_compute(k, [](const flag_gems::TuneKey&) {
    return triton_jit::Config{4, 1, {}};
  });
  (void)cfg;

  // Force instantiation of lookup<Args...> with int args (no tensor; pybind11
  // tensor caster requires torch headers we don't pull at compile probe).
  // Body of dispatch_to_bridge_ is instantiated too, exercising the full
  // GIL / py::make_tuple / dict-unpack chain at the type level.
  // NOT actually called -- if (false) keeps it dead code post-O0.
  if (false) {
    auto grid = [](const triton_jit::Config&)
        -> std::tuple<unsigned, unsigned, unsigned> { return {1u, 1u, 1u}; };
    const auto& cfg2 = ac.lookup(k, grid, 0, 0, 256, 128);
    (void)cfg2;
  }
}
