#include "nanobind/nanobind.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(SignalsmithStretch, m) {
    m.doc() = "Initial Commit";
}
