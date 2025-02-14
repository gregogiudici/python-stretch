import python_stretch as m
import numpy as np

def test_mono():
    x1 = np.random.normal(0, 0.1, size=(1, 44100)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(12)
    
    y1 = ps.process(x1) 
    del ps
    assert y1.shape == (1, 44100)
    
def test_stereo():
    x1 = np.random.normal(0, 0.1, size=(2, 44100)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(12)
    
    y1 = ps.process(x1) 
    del ps
    assert y1.shape == (2, 44100)
    
def test_multichannel():
    n_channels = np.random.randint(3, 10)
    x1 = np.random.normal(0, 0.1, size=(n_channels, 44100)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(12)
    
    y1 = ps.process(x1)
    del ps
    
    assert y1.shape == (n_channels, 44100)
    