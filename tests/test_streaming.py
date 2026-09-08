import hashlib

import numpy as np
import python_stretch as m


SR = 44100.0


def _aligned_residual(a, b, max_lag=64):
    """Align two 1-D signals within +/- max_lag samples (by lowest
    residual energy), then return the residual energy of (a - b) at that
    alignment, normalized by b's energy.

    0 means identical. A max-sample-to-sample-step metric can score a
    signal that has been corrupted every other block the same as a clean
    one (both can be dominated by one loud transient elsewhere in the
    signal) -- this residual-energy metric doesn't have that blind spot,
    because it compares the whole signal, not just its single worst jump.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b)) - max_lag
    best_lag, best_score = 0, None
    for lag in range(-max_lag, max_lag + 1):
        seg_a = a[lag:lag + n] if lag >= 0 else a[:n]
        seg_b = b[:n] if lag >= 0 else b[-lag:-lag + n]
        score = np.sum((seg_a - seg_b) ** 2)
        if best_score is None or score < best_score:
            best_score, best_lag = score, lag
    seg_a = a[best_lag:best_lag + n] if best_lag >= 0 else a[:n]
    seg_b = b[:n] if best_lag >= 0 else b[-best_lag:-best_lag + n]
    residual_energy = np.sum((seg_a - seg_b) ** 2)
    reference_energy = np.sum(seg_b ** 2)
    return residual_energy / reference_energy if reference_energy > 0 else float("inf")


# process()'s behaviour must not change by one sample. These digests were
# captured by running the SAME configurations below against the
# unmodified build (before processBlock()/seek()/flush() were added,
# process() itself untouched). If process() ever produces different
# output for these inputs, one of these will fail -- if that's ever a
# deliberate change, these digests are the thing to regenerate, and only
# that.
EXPECTED_DIGESTS = {
    "mono_tf1_notranspose": ("33fcf21ef375323523575dc391cdf10cdcbacc5e2f5986ef214f3d6658138240", (1, 22050)),
    "mono_tf0_75_notranspose": ("17f0fcc51bc4d5018c0b683f7fe979723efaf8bfc6258ae2667da7e372518d30", (1, 29400)),
    "mono_tf1_transpose12": ("0f02302923cc64b6e02f35f492a573b7729dedaca4d6faa01e7f8788fe3c37d8", (1, 22050)),
    "stereo_tf1_notranspose": ("163eba351a24455dcac3eb3c5ad32055f50b860c7387eace9052f738e79dbcdd", (2, 22050)),
    "stereo_tf1_5_notranspose": ("6e0c94926b212cab6b2d055f0e7359939404986d93098af58aa58204ac07f63f", (2, 14700)),
    "stereo_tf1_transpose_minus7": ("e65634cf4889030ceb291cbd42cac28701f8263be9f4b1918f6d94295d2d9768", (2, 22050)),
}

# (channels, n_samples, time_factor, transpose_semitones, seed), matched
# to the names above by position.
_CONFIGS = [
    ("mono_tf1_notranspose", 1, 22050, 1.0, 0.0, 0),
    ("mono_tf0_75_notranspose", 1, 22050, 0.75, 0.0, 1),
    ("mono_tf1_transpose12", 1, 22050, 1.0, 12.0, 2),
    ("stereo_tf1_notranspose", 2, 22050, 1.0, 0.0, 3),
    ("stereo_tf1_5_notranspose", 2, 22050, 1.5, 0.0, 4),
    ("stereo_tf1_transpose_minus7", 2, 22050, 1.0, -7.0, 5),
]


def test_process_output_is_unchanged_backwards_compatible():
    # process() isn't touched by this change at all -- this test pins its
    # existing behaviour rather than exercising anything new, so there's
    # no red phase to show for it.
    for name, channels, n, tf, semitones, seed in _CONFIGS:
        rng = np.random.default_rng(seed)
        audio = rng.normal(0, 0.1, size=(channels, n)).astype(np.float32)
        s = m.Signalsmith.Stretch(seed=0)
        s.preset(channels, SR)
        s.setTimeFactor(tf)
        if semitones != 0.0:
            s.setTransposeSemitones(semitones)
        out = s.process(audio)
        expected_digest, expected_shape = EXPECTED_DIGESTS[name]
        assert out.shape == expected_shape, name
        assert hashlib.sha256(out.tobytes()).hexdigest() == expected_digest, name


def test_streaming_blocks_match_a_one_shot_reference():
    rng = np.random.default_rng(10)
    total = rng.normal(0, 0.1, size=(1, 44100)).astype(np.float32)
    chunk = 4410

    reference = m.Signalsmith.Stretch(seed=0)
    reference.preset(1, SR)
    ref_out = reference.processBlock(total, total.shape[1])

    streamed = m.Signalsmith.Stretch(seed=0)
    streamed.preset(1, SR)
    parts = [
        streamed.processBlock(total[:, i:i + chunk], chunk)
        for i in range(0, total.shape[1], chunk)
    ]
    streamed_out = np.concatenate(parts, axis=1)

    # In this case they're not just close, they're bit-for-bit identical --
    # the residual/energy check below is what a real (non-integer) block
    # size or time-stretch ratio would need, kept here for the negative
    # control that follows.
    assert np.array_equal(ref_out, streamed_out)
    clean_score = _aligned_residual(streamed_out[0], ref_out[0])
    assert clean_score < 1e-9

    # Negative control: corrupt every other block by inverting its sign,
    # and check the metric actually reacts. A metric that can't tell a
    # corrupted stream from a clean one isn't testing anything.
    corrupted = m.Signalsmith.Stretch(seed=0)
    corrupted.preset(1, SR)
    parts = []
    for idx, i in enumerate(range(0, total.shape[1], chunk)):
        block_out = corrupted.processBlock(total[:, i:i + chunk], chunk)
        parts.append(-block_out if idx % 2 else block_out)
    corrupted_out = np.concatenate(parts, axis=1)

    corrupted_score = _aligned_residual(corrupted_out[0], ref_out[0])
    assert corrupted_score > 1.0
    assert corrupted_score > 1000 * max(clean_score, 1e-12)


def test_persistent_object_differs_from_a_fresh_object_per_block():
    # process() resets the processor on every call, so a persistent
    # object driven in blocks and a fresh object built per block give
    # bit-identical output through process() -- that's the defect this
    # change exists to fix (see tests/test_pitch_shift.py and
    # tests/test_time_stretch.py, which never drive process() in blocks
    # at all, since there was no point: every call was already a fresh
    # start). processBlock() doesn't reset, so state should carry over,
    # and a persistent object should behave differently from a fresh one
    # per block.
    rng = np.random.default_rng(11)
    total = rng.normal(0, 0.1, size=(1, 44100)).astype(np.float32)
    chunk = 4410

    persistent = m.Signalsmith.Stretch(seed=0)
    persistent.preset(1, SR)
    parts = [
        persistent.processBlock(total[:, i:i + chunk], chunk)
        for i in range(0, total.shape[1], chunk)
    ]
    out_persistent = np.concatenate(parts, axis=1)

    parts = []
    for i in range(0, total.shape[1], chunk):
        fresh = m.Signalsmith.Stretch(seed=0)
        fresh.preset(1, SR)
        parts.append(fresh.processBlock(total[:, i:i + chunk], chunk))
    out_fresh = np.concatenate(parts, axis=1)

    assert not np.array_equal(out_persistent, out_fresh)


def test_streaming_is_deterministic_across_repeated_runs():
    def run():
        rng = np.random.default_rng(12)
        total = rng.normal(0, 0.1, size=(2, 22050)).astype(np.float32)
        s = m.Signalsmith.Stretch(seed=0)
        s.preset(2, SR)
        chunk = 2205
        parts = [
            s.processBlock(total[:, i:i + chunk], chunk)
            for i in range(0, total.shape[1], chunk)
        ]
        return np.concatenate(parts, axis=1)

    assert np.array_equal(run(), run())


def test_time_factor_change_between_blocks_takes_effect_and_keeps_pitch():
    freq = 440.0
    n = 44100
    t = np.arange(n) / SR
    tone = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(1, -1)

    s = m.Signalsmith.Stretch(seed=0)
    s.preset(1, SR)

    chunk = 4410
    half = n // 2
    parts = [
        s.processBlock(tone[:, i:i + chunk], chunk)
        for i in range(0, half, chunk)
    ]
    # Second half: ask for double-length output per block, i.e. half speed.
    parts += [
        s.processBlock(tone[:, i:i + chunk], chunk * 2)
        for i in range(half, n, chunk)
    ]
    out = np.concatenate(parts, axis=1)

    assert out.shape[1] == half + (n - half) * 2

    def dominant_freq(x):
        window = np.hanning(len(x))
        spectrum = np.abs(np.fft.rfft(x * window))
        freqs = np.fft.rfftfreq(len(x), d=1 / SR)
        return freqs[np.argmax(spectrum)]

    # 44.1kHz / 22050 samples gives ~2Hz FFT bins; 5Hz is a generous
    # margin either side of the true 440Hz tone.
    assert abs(dominant_freq(out[0, :half]) - freq) < 5.0
    assert abs(dominant_freq(out[0, half:]) - freq) < 5.0


def test_many_blocks_do_not_crash():
    rng = np.random.default_rng(13)
    s = m.Signalsmith.Stretch(seed=0)
    s.preset(1, SR)
    chunk = 512
    for _ in range(20000):
        block = rng.normal(0, 0.1, size=(1, chunk)).astype(np.float32)
        out = s.processBlock(block, chunk)
        assert out.shape == (1, chunk)


def test_seek_and_flush_do_not_crash_and_return_correct_shapes():
    rng = np.random.default_rng(14)
    total = rng.normal(0, 0.1, size=(2, 22050)).astype(np.float32)

    s = m.Signalsmith.Stretch(seed=0)
    s.preset(2, SR)
    s.seek(total[:, :4410], 1.0)
    out = s.processBlock(total[:, 4410:8820], 4410)
    assert out.shape == (2, 4410)
    tail = s.flush(s.outputLatency())
    assert tail.shape == (2, s.outputLatency())


def test_processblock_output_samples_defaults_to_timefactor():
    # Same default as process(): output length = round(input length /
    # timeFactor), used whenever output_samples isn't given.
    s = m.Signalsmith.Stretch(seed=0)
    s.preset(1, SR)
    s.setTimeFactor(2.0)
    block = np.zeros((1, 1000), dtype=np.float32)

    out_explicit_none = s.processBlock(block, None)
    assert out_explicit_none.shape == (1, 500)

    out_omitted = s.processBlock(block)
    assert out_omitted.shape == (1, 500)


def test_processblock_before_configure_raises_a_clear_error():
    s = m.Signalsmith.Stretch(seed=0)
    audio = np.zeros((1, 100), dtype=np.float32)
    try:
        s.processBlock(audio, 100)
    except RuntimeError as e:
        assert "preset" in str(e) or "configure" in str(e)
    else:
        raise AssertionError("expected a RuntimeError before preset()/configure()")
