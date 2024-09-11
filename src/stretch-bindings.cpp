#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>
#include "stretch/signalsmith-stretch.h"

namespace nb = nanobind;

using namespace nb::literals;


// Buffer class for reading audio, with offset reading
// (Buffer class is based on "Wav" class used in https://github.com/Signalsmith-Audio/signalsmith-stretch/blob/example-cmd/cmd/util/wav.h)
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
    public:
        Stretch() : stretch_() {}
        Stretch(long seed) : stretch_(seed) {}

        // === Configuration ===
        bool exactLength_ = true;
        Sample sampleRate_;
        Sample timeFactor_ = 1;

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
        nb::ndarray<nb::numpy, Sample,nb::ndim<2>> process(nb::ndarray<nb::numpy, Sample,nb::ndim<2>> audio_input){
            auto data = audio_input.data();

            if (audio_input.shape(0) > 2) {
                throw std::runtime_error("Only mono or stereo audio is supported. The input should have shape (1,samples) or (2,samples).");
            }

            // Padding for latency
            size_t paddedInputSamples = audio_input.shape(1) + stretch_.inputLatency();
	        int tailSamples = exactLength_ ? stretch_.outputLatency() : (stretch_.outputLatency() + stretch_.inputLatency()); // if we don't need exact length, add a bit more output to catch any wobbles past the end
            int outputSamples = std::round(audio_input.shape(1) * timeFactor_);
            size_t paddedOutputSamples = outputSamples + tailSamples;

            // Create input buffer
            Sample** inputBuffer = new Sample*[audio_input.shape(0)];
            for (size_t i = 0; i < audio_input.shape(0); i++) {
                inputBuffer[i] = new Sample[paddedInputSamples];
                std::fill(inputBuffer[i], inputBuffer[i] + paddedInputSamples, 0);
            }

            // Copy input data to input buffer[channel]
            for (size_t i = 0; i < audio_input.shape(0); i++) {
                std::copy(data + i*audio_input.shape(1), data + (i+1)*audio_input.shape(1), inputBuffer[i]);
            }

            // Create output buffer
            Sample** outputBuffer = new Sample*[audio_input.shape(0)];
            for (size_t i = 0; i < audio_input.shape(0); i++) {
                outputBuffer[i] = new Sample[paddedOutputSamples];
                std::fill(outputBuffer[i], outputBuffer[i] + paddedOutputSamples, 0);
            }

            // Wrap input/output buffer
            Buffer<Sample> inBuffer(inputBuffer, paddedInputSamples);
            Buffer<Sample> outBuffer(outputBuffer, outputSamples);

            // Set offset of inBuffer. This is the latency of the stretch processor
            inBuffer.setOffset(stretch_.inputLatency());

            // PROCESSING
            stretch_.process(inBuffer, audio_input.shape(1), outBuffer, outputSamples);

            // Read the last bit of output without providing any further input
            outBuffer.setOffset(outputSamples);
            stretch_.flush(outputBuffer, tailSamples);
            outBuffer.setOffset(0);

            // Allocate output data buffer and shape outside the conditional
            Sample* outData = nullptr;
            size_t outShape[2];

            // Decide the size of `outData` and `outShape` based on `exactLength_`
            if (exactLength_) {
                outData = new Sample[audio_input.shape(0) * outputSamples];

                for (size_t i = 0; i < audio_input.shape(0); ++i) {
                    std::copy(outputBuffer[i], outputBuffer[i] + outputSamples, outData + i * outputSamples);
                }

                outShape[0] = audio_input.shape(0);
                outShape[1] = outputSamples;
                
            } else {
                outData = new Sample[audio_input.shape(0) * paddedOutputSamples];

                for (size_t i = 0; i < audio_input.shape(0); ++i) {
                    std::copy(outputBuffer[i], outputBuffer[i] + paddedOutputSamples, outData + i * paddedOutputSamples);
                }

                outShape[0] = audio_input.shape(0);
                outShape[1] = paddedOutputSamples;
            }

            // Delete 'outData' when the 'owner' capsule expires
            nb::capsule owner(outData, [](void *p) noexcept {
                delete[] static_cast<Sample*>(p);  // Properly cast the pointer before deletion
            });

            // Create the output ndarray
            return nb::ndarray<nb::numpy, Sample, nb::ndim<2>>(outData, 2, outShape, owner);
            }      
};

// Assuming Sample is 'float' for simplicity
using Sample = float;

NB_MODULE(SignalsmithStretch, m) {
    nb::class_<Stretch<Sample>>(m, "Stretch")
        .def(nb::init<>())
        .def(nb::init<long>(), "seed"_a)
        // Getters
        .def("blockSamples", &Stretch<Sample>::blockSamples)
        .def("intervalSamples", &Stretch<Sample>::intervalSamples)
        .def("inputLatency", &Stretch<Sample>::inputLatency)
        .def("outputLatency", &Stretch<Sample>::outputLatency)
        // Access to timeFactor_, sampleRate_, exactLength_
        .def_rw("timeFactor", &Stretch<Sample>::timeFactor_)
        .def_rw("sampleRate", &Stretch<Sample>::sampleRate_)
        .def_rw("exactLength", &Stretch<Sample>::exactLength_)
        // Settings
        .def("reset", &Stretch<Sample>::reset)
        .def("preset", &Stretch<Sample>::preset,
            "nChannels"_a, "sampleRate"_a, "cheaper"_a=false, "exactLength"_a=false)
        .def("configure", &Stretch<Sample>::configure,
            "nChannels"_a, "blockSamples"_a, "intervalSamples"_a)
        .def("setTransposeFactor", &Stretch<Sample>::setTransposeFactor,
            "multiplier"_a, "tonalityLimit"_a=0)
        .def("setTransposeSemitones", &Stretch<Sample>::setTransposeSemitones,
            "semitones"_a, "tonalityLimit"_a=0)
        .def("setFreqMap", &Stretch<Sample>::setFreqMap,
            "inputToOutput"_a)
        // Processing   
        .def("process", &Stretch<Sample>::process,
            "audio_input"_a);
}
