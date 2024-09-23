import numpy as np
import timeit
import python_stretch as pystretch
import librosa

def test_1():
    # Load the audio file
    y,sr = librosa.load('examples/les_bridge_fing01__00000.wav', sr=None)
    y = y[np.newaxis,:]

    # Create the pystretch object
    ps = pystretch.Signalsmith.Stretch()
    ps.preset(1,sr)
    ps.setTransposeSemitones(12)

    # Time the stretch function
    t = timeit.timeit(lambda: ps.process(y), number=10)
    print('Test 1 (transpose): %f' % t)

def test_2():

    def test_stretch():
        ps = pystretch.Signalsmith.Stretch()
        ps.preset(1,sr)
        ps.timeFactor = 2.0

        # Time the stretch function
        ps.process(y)
    
    # Load the audio file
    y, sr = librosa.load('examples/les_bridge_fing01__00000.wav', sr=None)
    y = y[np.newaxis,:]

    # Time the stretch function
    t = timeit.timeit(test_stretch, number=10)
    print('Test 2 (stretch): %f' % t)

if __name__ == '__main__':
    test_1()
    test_2()