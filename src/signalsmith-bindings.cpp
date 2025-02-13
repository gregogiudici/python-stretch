#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>
#include "stretch/signalsmith-stretch.h"

namespace nb = nanobind;

using namespace nb::literals;


// Buffer class for reading audio (with offset reading)
// (Buffer class is based on "Wav" class in https://github.com/Signalsmith-Audio/signalsmith-stretch/blob/main/cmd/util/wav.h)
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
        Sample sampleRate_;
        Sample timeFactor_ = 1.f;
        Sample freqMultiplier_ = 1.f;
        Sample freqSemitones_ = 0.f;
    public:
        Stretch() : stretch_() {}
        Stretch(long seed) : stretch_(seed) {}

        // === Access to private members ===
        void set_sr(Sample value) { sampleRate_ = value; }
        Sample sampleRate() const { return sampleRate_;}
        void set_tf(Sample value) { timeFactor_ = value; }
        Sample timeFactor() const { return timeFactor_; }
        void set_fm(Sample value) { freqMultiplier_ = value; }
        Sample freqMultiplier() const { return freqMultiplier_; }
        void set_fs(Sample value) { freqSemitones_ = value; }
        Sample freqSemitones() const { return freqSemitones_; } 

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
        void preset(int nChannels, Sample sampleRate, bool cheaper = false) {
            if (cheaper) {
                stretch_.presetCheaper(nChannels, sampleRate);
            } else {
                stretch_.presetDefault(nChannels, sampleRate);
            }
            sampleRate_ = sampleRate;
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
        // void setFreqMap(std::function<Sample(Sample)> inputToOutput) {
        //     stretch_.setFreqMap(inputToOutput);
        // }

        void setTimeFactor(Sample timeFactor) {
            timeFactor_ = timeFactor;
        }

        // ==================== TO BE REMOVED ====================
        // Simple stretch function
        void simple_stretch_(const float* inputSignal, size_t inputSize, float* outputSignal, size_t outputSize) {
            // Compress by linear interpolation
            for (size_t i = 0; i < outputSize; ++i) {
                // Compute the input index using multiplication (works for compression)
                float inputIndex = (timeFactor_ > 0.f ) ? i / timeFactor_ : i * timeFactor_;
                size_t idx1 = static_cast<size_t>(inputIndex);

                // Ensure we don't go out of bounds
                size_t idx2 = (idx1 + 1 < inputSize) ? idx1 + 1 : idx1;

                // Linear interpolation between inputSignal[idx1] and inputSignal[idx2]
                float fraction = inputIndex - idx1;
                float interpolatedValue = (1.0f - fraction) * inputSignal[idx1] + fraction * inputSignal[idx2];

                // Assign the interpolated value to the output signal
                outputSignal[i] = interpolatedValue;
            }
        }
        // ====================

        // === Processing ===
        nb::ndarray<nb::numpy, float, nb::ndim<2>> process(nb::ndarray<nb::numpy, float, nb::ndim<2>> audio_input) {
            auto inData = audio_input.data();

            size_t numChannels = audio_input.shape(0);
            size_t inputLength  = audio_input.shape(1);
            
            // Padding for latency
            size_t paddedInputLength = inputLength  + stretch_.inputLatency();
            int tailSamples = stretch_.outputLatency();
            size_t outputLength  = std::round(inputLength / timeFactor_);
            size_t paddedOutputLength = outputLength  + tailSamples;

            // Allocate and initialize buffers
            float** inputChannels = new float*[numChannels];
            float** outputChannels = new float*[numChannels];
            
            for (size_t i = 0; i < numChannels; ++i) {
                inputChannels[i] = new float[paddedInputLength]();
                outputChannels[i] = new float[paddedOutputLength]();
            }

            // Copy from inData to inputChannels
            for (size_t i = 0; i < numChannels; ++i) {
                std::copy(inData + i*inputLength  , inData + (i+1)*inputLength  , inputChannels[i]);
            }

            // Wrap input/output channel-buffer with Buffer class (for offset reading/writing)
            Buffer<float> inBuffer(inputChannels, paddedInputLength);
            Buffer<float> outBuffer(outputChannels, outputLength);

            // Seek to the beginning of the input buffer
            stretch_.seek(inBuffer, stretch_.inputLatency(), timeFactor_);

            // Set offset of inBuffer
            inBuffer.setOffset(stretch_.inputLatency());

            // PROCESSING
            stretch_.process(inBuffer, inputLength, outBuffer, outputLength);

            // Read the last bit of output without providing any further input
            outBuffer.setOffset(outputLength);
            stretch_.flush(outBuffer, tailSamples);
            // outBuffer.setOffset(tailSamples);

            // Prepare output data
            size_t outShape[2] = {numChannels, outputLength };
            float* outData = new float[numChannels * outShape[1]];

            // Copy from outputChannels to outData
            for (size_t i = 0; i < numChannels; ++i) {
                std::copy(outputChannels[i] + tailSamples, outputChannels[i] + paddedOutputLength , outData + i * outputLength );
            }

            // REMEMBER: Reset the stretch processor or we will get an error: free() invalid pointer
            stretch_.reset();

            // Clean up
            for (size_t i = 0; i < numChannels; ++i) {
                delete[] inputChannels[i];
                delete[] outputChannels[i];
            }
            delete[] inputChannels;
            delete[] outputChannels;

            // Delete 'outData' when the 'owner' capsule expires
            nb::capsule owner(outData, [](void *p) noexcept {
                delete[] static_cast<float*>(p);
            });

            // Create the output ndarray
            return nb::ndarray<nb::numpy, float, nb::ndim<2>>(outData, 2, outShape, owner);
        }
};

// Assuming Sample is 'float' for simplicity
using Sample = float;

NB_MODULE(Signalsmith, m) {
    m.doc() = "Python binding of the Signalsmith Stretch library, providing time-stretching and pitch-shifting capabilities.";
    
    nb::class_<Stretch<Sample>>(m, "Stretch", "Class for Stretch processor.")
        .def(nb::init<>(), "Default constructor.")
        .def(nb::init<long>(), "seed"_a, "Constructor with seed for deterministic behavior.")
        
        // Attribute getters
        .def("blockSamples", &Stretch<Sample>::blockSamples, "Get the block size used in processing.")
        .def("intervalSamples", &Stretch<Sample>::intervalSamples, "Get the interval size for overlapping.")
        .def("inputLatency", &Stretch<Sample>::inputLatency, "Get the input latency of the processor in samples.")
        .def("outputLatency", &Stretch<Sample>::outputLatency, "Get the output latency of the processor in samples.")
        
        // Access to timeFactor_ and sampleRate_
        .def_prop_rw("sampleRate", 
            [](Stretch<Sample> &t) { return t.sampleRate(); },
            [](Stretch<Sample> &t, Sample value) { t.set_sr(value); },
            "Sample rate of the processor in Hz.")
        .def_prop_rw("timeFactor", 
            [](Stretch<Sample> &t) { return t.timeFactor(); },
            [](Stretch<Sample> &t, Sample value) { t.set_tf(value); },
            "Time-stretching factor. A value >1 speeds up the signal, <1 slows it down.")
        
        // Settings
        .def("reset", &Stretch<Sample>::reset, "Reset the processor to its initial state.")
        .def("preset", &Stretch<Sample>::preset,
            "nChannels"_a, "sampleRate"_a, "cheaper"_a=false,
            "Configure the Stretch processor with a preset.\n\n"
            "Parameters:\n"
            "----------\n"
            "- nChannels (int): Number of audio channels.\n"
            "- sampleRate (float): Sample rate in Hz.\n"
            "- cheaper (bool, optional): If True, uses a lower-quality but more efficient configuration (default: False).")
        .def("configure", &Stretch<Sample>::configure,
            "nChannels"_a, "blockSamples"_a, "intervalSamples"_a,
            "Manually configure the stretch processor.\n\n"
            "Parameters:\n"
            "----------\n"
            "- nChannels (int): Number of audio channels.\n"
            "- blockSamples (int): Block size for processing.\n"
            "- intervalSamples (int): Interval size for overlapping.")
        
        .def("setTransposeFactor", &Stretch<Sample>::setTransposeFactor,
            "multiplier"_a, "tonalityLimit"_a=0,
            "Set the transposition factor for pitch shifting.\n\n"
            "Parameters:\n"
            "----------\n"
            "- multiplier (float): Pitch shift multiplier (e.g., 2.0 for an octave up).\n"
            "- tonalityLimit (float, optional): Restriction on tonal adjustments (default: 0).")
        .def("setTransposeSemitones", &Stretch<Sample>::setTransposeSemitones,
            "semitones"_a, "tonalityLimit"_a=0,
            "Set the pitch shift in semitones.\n\n"
            "Parameters:\n"
            "----------\n"
            "- semitones (float): Number of semitones to shift (e.g., +12 for an octave up).\n"
            "- tonalityLimit (float, optional): Restriction on tonal adjustments (default: 0).")
        .def("setTimeFactor", &Stretch<Sample>::setTimeFactor,
            "timeFactor"_a,
            "Set the time-stretching factor.\n\n"
            "Parameters:\n"
            "----------\n"
            "- timeFactor (float): Factor by which time is stretched or compressed (e.g., 0.5 slows down by half, 2.0 doubles speed).")

        // PROCESSING   
        .def("process", &Stretch<Sample>::process,
            "audio_input"_a,
            "Process an input audio buffer and return the stretched or pitch-shifted output.\n\n"
            "Parameters:\n"
            "----------\n"
            "- audio_input (numpy.ndarray): Input audio buffer to be processed.\n\n"
            "Returns:\n"
            "----------\n"
            "- numpy.ndarray: Stretched or pitch-shifted output audio buffer.")
        ;
        // .def("setFreqMap", &Stretch<Sample>::setFreqMap,
        //     "inputToOutput"_a) // TODO: implement custom frequency mapping
}
