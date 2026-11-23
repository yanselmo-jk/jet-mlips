// jet: library definition + config registry. Kernels register their own ops
// through TORCH_LIBRARY_FRAGMENT(jet_mlip, m) in their .cu files.
#include <torch/library.h>
#include <torch/torch.h>
#include <string>
#include <vector>

#include "configs_gen.h"

namespace jet_mlip {

int64_t num_configs() { return JET_NUM_CONFIGS; }

std::string config_key(int64_t id) {
  TORCH_CHECK(id >= 0 && id < JET_NUM_CONFIGS, "jet: bad config id ", id);
  return JET_CONFIG_KEYS[id];
}

// id of a config key in this build (ids differ between builds). A missing key is printed before
// the throw: torch 2.7 segfaults while formatting any exception raised inside a frozen module
// (InterpreterStateImpl::callstack), which is what deployed models are.
int64_t config_index(const std::string& key) {
  for (int64_t i = 0; i < JET_NUM_CONFIGS; i++) {
    if (key == JET_CONFIG_KEYS[i]) return i;
  }
  const std::string msg = "jet: config not compiled in this build: " + key +
      " (add it to jet_mlip/configs.py, e.g. python -m jet_mlip.shapes <family> <model> --apply, and rebuild)";
  TORCH_WARN(msg);
  TORCH_CHECK(false, msg);
}

}  // namespace jet_mlip

TORCH_LIBRARY(jet_mlip, m) {
  m.def("num_configs() -> int", &jet_mlip::num_configs);
  m.def("config_key(int id) -> str", &jet_mlip::config_key);
  // CONSERVATIVE: never constant-folded, so a deployed (torch.jit.freeze) model resolves the
  // key against the jet build that loads it, not the one that deployed it
  m.def(torch::schema("config_index(str key) -> int", c10::AliasAnalysisKind::CONSERVATIVE),
        &jet_mlip::config_index);
}
