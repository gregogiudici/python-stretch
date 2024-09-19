import numpy as np
import timeit
import pystretch
import librosa

def test_1():
    # Load the audio file
    y,sr = librosa.load('examples/les_bridge_fing01__00000.wav', sr=None)
    y = y[np.newaxis,:]

    # Create the pystretch object
    ps = pystretch.Signalsmith.Stretch()
    ps.preset(1,sr)

    # Time the stretch function
    t = timeit.timeit(lambda: ps.process(y), number=2)
    print('Stretch time: %f' % t)

def test_2():

    def test_stretch():
        ps = pystretch.Signalsmith.Stretch()
        ps.preset(1,sr)

        # Time the stretch function
        ps.process(y)
    
    # Load the audio file
    y, sr = librosa.load('examples/les_bridge_fing01__00000.wav', sr=None)
    y = y[np.newaxis,:]

    # Time the stretch function
    t = timeit.timeit(test_stretch, number=10)
    print('Stretch time: %f' % t)

if __name__ == '__main__':
    test_1()