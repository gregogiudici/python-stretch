import pystretch as m
import numpy as np

def test_double_length():
    x1 = np.random.rand(1,44100)
    x2 = np.random.rand(1,22050)
    ps = m.SignalsmithStretch.Stretch()
    ps.timeFactor = 0.5
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    assert y1.shape == (1, 88200)
    assert y2.shape == (1, 44100)

def test_half_length():
    x1 = np.random.rand(1,44100)
    x2 = np.random.rand(1,22050)
    ps = m.SignalsmithStretch.Stretch()
    ps.timeFactor = 2.
    y1 = ps.process(x1) 
    y2 = ps.process(x2)
    del ps
    assert y1.shape == (1, 22050)
    assert y2.shape == (1, 11025)