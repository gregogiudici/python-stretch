#include "nanobind/nanobind.h"
#include "stretch/signalsmith-stretch.h"

namespace nb = nanobind;

using namespace nb::literals;

// Assuming Sample is 'float' for simplicity
using Sample = float;

NB_MODULE(SignalsmithStretch, m) {
    nb::class_<signalsmith::stretch::SignalsmithStretch<Sample>>(m, "Stretch")
    .def(nb::init<>())  // Default constructor
    .def(nb::init<long>(), "seed"_a)  // Constructor with seed

    .def("blockSamples", &signalsmith::stretch::SignalsmithStretch<Sample>::blockSamples)
    .def("intervalSamples", &signalsmith::stretch::SignalsmithStretch<Sample>::intervalSamples)
    .def("inputLatency", &signalsmith::stretch::SignalsmithStretch<Sample>::inputLatency)
    .def("outputLatency", &signalsmith::stretch::SignalsmithStretch<Sample>::outputLatency)
    .def("reset", &signalsmith::stretch::SignalsmithStretch<Sample>::reset)
    .def("presetDefault", &signalsmith::stretch::SignalsmithStretch<Sample>::presetDefault)
    .def("presetCheaper", &signalsmith::stretch::SignalsmithStretch<Sample>::presetCheaper)
    .def("configure", &signalsmith::stretch::SignalsmithStretch<Sample>::configure)
    ;
}
