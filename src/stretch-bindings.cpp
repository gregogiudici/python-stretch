#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>
#include "stretch/signalsmith-stretch.h"

namespace nb = nanobind;

using namespace nb::literals;


// Buffer class for reading audio, with offset reading
// (Buffer class is based on Wav class used in https://github.com/Signalsmith-Audio/signalsmith-stretch/blob/example-cmd/cmd/util/wav.h)
template<typename Sample=float>
class Buffer{
    private:
        Sample** buffer_;
        size_t offset_ = 0;
        size_t ch_size_;
    public:
        Buffer(Sample** buffer, size_t size) : buffer_(buffer), ch_size_(size), offset_(0) {}

        template<bool isConst>
	    class ChannelReader {
		    using CSample = typename std::conditional<isConst, const Sample, Sample>::type;
		    CSample *data_;
		    int offset_;
	    public:
		    ChannelReader(CSample *samples, int offset) : data_(samples), offset_(offset) {}
		
		    CSample & operator [](int i) {
			return data_[i + offset_];
		    }
        };

        ChannelReader<false> operator[](int channel) {
            return ChannelReader<false>(buffer_[channel], offset_);
        }

        ChannelReader<true> operator[](int channel) const {
            return ChannelReader<true>(buffer_[channel], offset_);
        }

        void reset() {
            offset_ = 0;
        }

        size_t size() const {
            return ch_size_;
        }

        size_t getOffset() const {
            return offset_;
        }


        void setOffset(int offset) {
            offset_ = offset;
        }
};


template<typename Sample=float>
struct Stretch{
    private:
        signalsmith::stretch::SignalsmithStretch<Sample> stretch_;
        Sample timeFactor_;
        Sample sampleRate_;
        Sample exactLength_ = false;
    public:
        Stretch() : stretch_() {}
        Stretch(long seed) : stretch_(seed) {}

        // === Getters === 
        int blockSamples() const {
            return stretch_.blockSamples();
        }
        int intervalSamples() const {
            return stretch_.intervalSamples();
        }
        int inputLatency() const {
            return stretch_.inputLatency();
        }
        int outputLatency() const {
            return stretch_.outputLatency();
        }

        // === Reset the stretch processor ===
        void reset() {
            stretch_.reset();
        }

        // === Preset configuration ==
        void preset(int nChannels, Sample sampleRate, bool cheaper = false, bool exactLength = false) {
            if (cheaper) {
                stretch_.presetCheaper(nChannels, sampleRate);
            } else {
                stretch_.presetDefault(nChannels, sampleRate);
            }
            sampleRate_ = sampleRate;
            exactLength_ = exactLength;
        }
        // === Manual configuration ===
        void configure(int nChannels, int blockSamples, int intervalSamples) {
            stretch_.configure(nChannels, blockSamples, intervalSamples);
        }

        // Set transpose factor
        void setTransposeFactor(Sample multiplier, Sample tonalityLimit=0) {
            stretch_.setTransposeFactor(multiplier, tonalityLimit);
        }
        void setTransposeSemitones(Sample semitones, Sample tonalityLimit=0) {
            stretch_.setTransposeSemitones(semitones, tonalityLimit);
        }
        void setFreqMap(std::function<Sample(Sample)> inputToOutput) {
            stretch_.setFreqMap(inputToOutput);
        }

        // === Processing ===
        nb::ndarray<nb::numpy, nb::ndim<2>> process(nb::ndarray<nb::numpy, nb::ndim<2>> audio_input){
            auto input = audio_input.view();
            auto data = input.data();

            if (input.shape(0) > 2) {
                throw std::runtime_error("Only mono or stereo audio is supported. The input should have shape (1,samples) or (2,samples).");
            }

            // Padding for latency
            size_t paddedInputSamples = in.shape(1) + stretch_.inputLatency();
	        int tailSamples = exactLength_ ? stretch_.outputLatency() : (stretch_.outputLatency() + stretch_.inputLatency()); // if we don't need exact length, add a bit more output to catch any wobbles past the end
            int outputSamples = std::round(input.shape(1) * _timeFactor);
            size_t paddedOutputSamples = outputSamples + tailSamples;

            // Create input buffer
            Sample** inputBuffer = new Sample*[input.shape(0)];
            for (size_t i = 0; i < input.shape(0); i++) {
                inputBuffer[i] = new Sample[paddedInputSamples];
                std::fill(inputBuffer[i], inputBuffer[i] + paddedInputSamples, 0);
            }

            // Copy input data to input buffer[channel]
            for (size_t i = 0; i < input.shape(0); i++) {
                std::copy(data + i*input.shape(1), data + (i+1)*input.shape(1), inputBuffer[i]);
            }

            // Create output buffer
            Sample** outputBuffer = new Sample*[in.shape(0)];
            for (size_t i = 0; i < input.shape(0); i++) {
                outputBuffer[i] = new Sample[paddedOutputSamples];
                std::fill(outputBuffer[i], outputBuffer[i] + paddedOutputSamples, 0);
            }

            // Wrap input/output buffer
            Buffer<Sample> inBuffer(inputBuffer, paddedInputSamples);
            Buffer<Sample> outBuffer(outputBuffer, outputSamples);

            // Set offset of inBuffer. This is the latency of the stretch processor
            inBuffer.setOffset(stretch_.inputLatency());

            // PROCESSING
            stretch_.process(inBuffer, input.shape(1), outBuffer, OutputSamples);

            // Read the last bit of output without providing any further input
            outBuffer.setOffset(outputSamples);
            stretch_.flush(outputBuffer, tailSamples);
            outBuffer.setOffset(0);
        }
};

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
