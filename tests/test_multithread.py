import numpy as np
import python_stretch as m
from concurrent.futures import ThreadPoolExecutor

NUM_THREADS = 8
SAMPLE_RATE = 44100
SEMITONES = 3
TIME_FACTOR = 1.25


def _make_stretch():
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(SEMITONES)
    ps.setTimeFactor(TIME_FACTOR)
    return ps


def _process_one(audio):
    ps = _make_stretch()
    return ps.process(audio)


def test_single_thread_determinism():
    """Same input → bit-identical output on repeated serial calls."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, SAMPLE_RATE * 4)).astype(np.float32)

    out_a = _process_one(x)
    out_b = _process_one(x)

    assert np.array_equal(out_a, out_b), "Serial outputs differ across repeated calls"


def test_parallel_consistency():
    """N independent Stretch instances in a thread pool → match serial reference outputs."""
    rng = np.random.default_rng(42)
    inputs = [rng.standard_normal((2, SAMPLE_RATE * 4)).astype(np.float32) for _ in range(NUM_THREADS)]

    # Serial reference: one Stretch per input, all on main thread
    serial_outputs = [_process_one(x) for x in inputs]

    # Parallel: one Stretch per thread (each thread owns its instance — the safe pattern)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        parallel_outputs = list(executor.map(_process_one, inputs))

    for i, (serial, parallel) in enumerate(zip(serial_outputs, parallel_outputs)):
        assert np.array_equal(serial, parallel), (
            f"Thread {i}: parallel output differs from serial reference"
        )


def test_cross_run_stability():
    """Same parallel batch repeated twice → identical results across runs."""
    rng = np.random.default_rng(42)
    inputs = [rng.standard_normal((2, SAMPLE_RATE * 4)).astype(np.float32) for _ in range(NUM_THREADS)]

    def run_parallel():
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            return list(executor.map(_process_one, inputs))

    outputs_a = run_parallel()
    outputs_b = run_parallel()

    for i, (a, b) in enumerate(zip(outputs_a, outputs_b)):
        assert np.array_equal(a, b), f"Thread {i}: outputs differ across parallel runs"
