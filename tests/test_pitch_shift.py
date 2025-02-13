import python_stretch as m
import numpy as np

def test_mono():
    x1 = np.random.rand(1, 44100)
    
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(12)
    
    y1 = ps.process(x1) 
    del ps
    assert y1.shape == (1, 44100)
    
def test_stereo():
    x1 = np.random.rand(2, 44100)
    
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(12)
    
    y1 = ps.process(x1) 
    del ps
    assert y1.shape == (2, 44100)
    
def test_multichannel():
    nChannels = np.random.randint(3, 10)
    x1 = np.random.rand(nChannels, 44100)
    
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(12)
    
    y1 = ps.process(x1)
    del ps
    
    assert y1.shape == (nChannels, 44100)
    