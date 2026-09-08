import python_stretch as m
import numpy as np


def test_non_contiguous_stereo_input_matches_contiguous():
    # This is the layout librosa.load(..., mono=False) commonly hands you:
    # a (samples, channels) array transposed to (channels, samples), which
    # is a view, not a copy, so it is not C-contiguous. See issue #3.
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.1, size=(44100, 2)).astype(np.float32)
    audio_noncontig = audio.T
    assert not audio_noncontig.flags["C_CONTIGUOUS"]

    audio_contig = np.ascontiguousarray(audio_noncontig)
    assert np.array_equal(audio_noncontig, audio_contig)

    ps_contig = m.Signalsmith.Stretch(seed=0)
    ps_contig.preset(2, 44100.0)
    ps_contig.setTimeFactor(1.0)
    out_contig = ps_contig.process(audio_contig)

    ps_noncontig = m.Signalsmith.Stretch(seed=0)
    ps_noncontig.preset(2, 44100.0)
    ps_noncontig.setTimeFactor(1.0)
    out_noncontig = ps_noncontig.process(audio_noncontig)

    assert np.array_equal(out_contig, out_noncontig)


def test_non_contiguous_stereo_from_slicing_matches_contiguous():
    # A second, unrelated way to end up non-contiguous: taking every other
    # sample. Strides differ from the transpose case above, so this checks
    # the fix isn't specific to one stride pattern.
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.1, size=(2, 88200)).astype(np.float32)
    audio_noncontig = audio[:, ::2]
    assert not audio_noncontig.flags["C_CONTIGUOUS"]

    audio_contig = np.ascontiguousarray(audio_noncontig)

    ps_contig = m.Signalsmith.Stretch(seed=0)
    ps_contig.preset(2, 44100.0)
    ps_contig.setTimeFactor(1.0)
    out_contig = ps_contig.process(audio_contig)

    ps_noncontig = m.Signalsmith.Stretch(seed=0)
    ps_noncontig.preset(2, 44100.0)
    ps_noncontig.setTimeFactor(1.0)
    out_noncontig = ps_noncontig.process(audio_noncontig)

    assert np.array_equal(out_contig, out_noncontig)


def test_mono_1d_input_is_promoted_to_a_single_channel():
    # process() previously required a 2-D array and raised a TypeError for
    # a 1-D one. The maintainer's proposed fix promotes a 1-D input to 2-D,
    # so this is new behaviour, not a case that already worked -- it is
    # exercised here so the promotion path doesn't silently regress later.
    rng = np.random.default_rng(2)
    audio = rng.normal(0, 0.1, size=(44100,)).astype(np.float32)
    assert audio.ndim == 1

    ps = m.Signalsmith.Stretch(seed=0)
    ps.preset(1, 44100.0)
    ps.setTimeFactor(1.0)
    out = ps.process(audio)

    assert out.shape == (1, 44100)
