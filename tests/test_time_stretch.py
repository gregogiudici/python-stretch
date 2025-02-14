import python_stretch as m
import numpy as np

def test_mono_double_length():
    x1 = np.random.normal(0, 0.1, size=(1, 44100)).astype(np.float32)
    x2 = np.random.normal(0, 0.1, size=(1, 22050)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTimeFactor(0.5)
    
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    
    assert y1.shape == (1, 88200)
    assert y2.shape == (1, 44100)

def test_mono_half_length():
    x1 = np.random.normal(0, 0.1, size=(1, 44100)).astype(np.float32)
    x2 = np.random.normal(0, 0.1, size=(1, 22050)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTimeFactor(2.)
    
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    
    assert y1.shape == (1, 22050)
    assert y2.shape == (1, 11025)
    
def test_stereo_double_length():
    x1 = np.random.normal(0, 0.1, size=(2, 44100)).astype(np.float32)
    x2 = np.random.normal(0, 0.1, size=(2, 22050)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTimeFactor(0.5)
    
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    
    assert y1.shape == (2, 88200)
    assert y2.shape == (2, 44100)
    
def test_stereo_half_length():
    x1 = np.random.rand(2,44100)
    x2 = np.random.rand(2,22050)
    
    ps = m.Signalsmith.Stretch()
    ps.setTimeFactor(2.)
    
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    
    assert y1.shape == (2, 22050)
    assert y2.shape == (2, 11025)
    
def test_multichannel_double_length():
    n_channels = np.random.randint(3, 10)
    x1 = np.random.normal(0, 0.1, size=(n_channels, 44100)).astype(np.float32)
    x2 = np.random.normal(0, 0.1, size=(n_channels, 22050)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTimeFactor(0.5)
    
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    
    assert y1.shape == (n_channels, 88200)
    assert y2.shape == (n_channels, 44100)
    
def test_multichannel_half_length():
    n_channels = np.random.randint(3, 10)
    x1 = np.random.normal(0, 0.1, size=(n_channels, 44100)).astype(np.float32)
    x2 = np.random.normal(0, 0.1, size=(n_channels, 22050)).astype(np.float32)
    
    ps = m.Signalsmith.Stretch()
    ps.setTimeFactor(2.)
    
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    
    assert y1.shape == (n_channels, 22050)
    assert y2.shape == (n_channels, 11025)