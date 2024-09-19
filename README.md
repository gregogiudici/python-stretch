# python-stretch: pitch shifting and time stretching
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/gregogiudici/python-stretch/blob/main/LICENSE)
[![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)](https://pypi.org/project/python-stretch)
[![Pip Action Status][actions-pip-badge]][actions-pip-link]
[![Pip Action Status][actions-wheels-badge]][actions-wheels-link]
<!-- [![PyPI - Wheel](https://img.shields.io/pypi/wheel/python-stretch)](https://pypi.org/project/python-stretch) -->
<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/python-stretch)](https://pypi.org/project/python-stretch) -->


[actions-pip-link]:        https://github.com/gregogiudici/python-stretch/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/gregogiudici/python-stretch/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/gregogiudici/python-stretch/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/gregogiudici/python-stretch/workflows/Wheels/badge.svg

A simple Python Wrapper of the Signalsmith Stretch C++ library for pitch and time stretching.

## Features

- **Time stretching**
- **Pitch shifting**
- **Multichannel support**: Works with mono and multichannel audio files.
- **Seamless integration**: Works natively with NumPy arrays for compatibility with libraries such as `librosa` and many others used in audio processing pipelines.


## Installation

`python-stretch` is available via PyPI (via [Platform Wheels](https://packaging.python.org/guides/distributing-packages-using-setuptools/#platform-wheels)):
```
pip install python-stretch
```
Alternatevly, you can easly build it from source:
```
# Clone from github
git clone --recurse-submodules https://github.com/gregogiudici/python-stretch.git
# Install
pip install ./python-stretch
```

# Examples
## Quick Start
```
import numpy as np
import librosa
import python_stretch as ps

# Load an audio example from librosa (e.g., 'trumpet', 'brahms',...)
audio, sr = librosa.load(librosa.ex('trumpet'), sr=None)

# Assure that "audio" is a 2d array
if (audio.shape == 1):
    audio = audio[np.newaxis, :]

# Create a Stretch object
s = ps.Signalsmith.Stretch()
# Configure using a preset
s.preset(audio.shape(0), sr) # numChannels, sampleRate
# Shift pitch 1 octave up
s.setTransposeSemitones(12)
# Double audio duration
s.timeFactor = 0.5

# Process
audio_processed = s.process(audio)

# Save and listen
import soundfile as sf
sf.write("audio_original.wav", np.squeeze(audio), sr)
sf.write("audio_processed.wav", np.squeeze(audio_processed), sr)
```

# Acknowledgements
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch): `python-stretch` is built on top of the Signalsmith Stretch C++ library, which provides the core algorithms for time stretching and pitch shifting.
- [nanobind](https://github.com/wjakob/nanobind): This project utilizes nanobind for easily binding the C++ code to Python.
