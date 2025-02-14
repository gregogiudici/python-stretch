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

def test_multichannel():
    y,sr = librosa.load('examples/les_bridge_fing01__00000.wav', sr=None, mono=False)
    print('Original file',y.shape)
    y = y[np.newaxis,:]
    print('Original MONO file',y.shape)
    
    ps = pystretch.Signalsmith.Stretch()
    ps.preset(y.shape[0],sr)
    
    # Process
    y_1 = ps.process(y)
    print('Stretched MONO',y_1.shape)
    
    # Copy the first channel to a second channel
    y = np.concatenate((y,y),axis=0)
    print('Original MONO file',y.shape)
    
    # Process
    ps.preset(y.shape[0],sr)
    ps.setTransposeSemitones(12)
    y_2 = ps.process(y)
    
    print('Stretched STEREO',y_2.shape)
    
    # Copy the first two channels to other two channels
    y = np.concatenate((y,y),axis=0)
    print('Original Multichannel', y.shape)
    
    ps.preset(y.shape[0],sr)
    y_3 = ps.process(y)
    print('Stretched Multichannel',y_3.shape)
    

if __name__ == '__main__':
    # test_1()
    # test_2()
    test_multichannel()