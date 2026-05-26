"""Benchmark serial vs parallel Stretch.process() to measure GIL-release speedup.

No audio files required. Run with:
    python examples/benchmark_multithread.py [--threads N] [--duration S] [--repeats N]

Expected results with the GIL-release patch (PR #4):
  - Serial (8x sequential) : ~400 ms
  - Parallel (8 threads)   :  ~55 ms
  - Speedup                :  ~7x

Without the patch, parallel speedup will be ~1x (threads serialize on the GIL).
"""

import argparse
import timeit
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import python_stretch as m


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=8, help="Number of parallel threads (default: 8)")
    p.add_argument("--duration", type=float, default=4.0, help="Audio duration in seconds (default: 4.0)")
    p.add_argument("--repeats", type=int, default=5, help="Number of timeit repeats (default: 5)")
    p.add_argument("--semitones", type=float, default=3.0, help="Pitch shift in semitones (default: 3)")
    p.add_argument("--time-factor", type=float, default=1.25, help="Time stretch factor (default: 1.25)")
    return p.parse_args()


def make_stretch(semitones, time_factor):
    ps = m.Signalsmith.Stretch()
    ps.setTransposeSemitones(semitones)
    ps.setTimeFactor(time_factor)
    return ps


def main():
    args = parse_args()

    sample_rate = 44100
    n_samples = int(sample_rate * args.duration)
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((2, n_samples)).astype(np.float32)

    print(f"python-stretch benchmark — GIL release speedup")
    print(f"  Audio   : {args.duration:.1f}s stereo @ {sample_rate} Hz")
    print(f"  Config  : +{args.semitones} semitones, {args.time_factor}x time")
    print(f"  Threads : {args.threads}")
    print(f"  Repeats : {args.repeats}")
    print()

    def run_serial():
        for _ in range(args.threads):
            ps = make_stretch(args.semitones, args.time_factor)
            ps.process(audio)

    def worker(_):
        ps = make_stretch(args.semitones, args.time_factor)
        ps.process(audio)

    def run_parallel():
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            list(executor.map(worker, range(args.threads)))

    t_serial = timeit.timeit(run_serial, number=args.repeats) / args.repeats * 1000
    t_parallel = timeit.timeit(run_parallel, number=args.repeats) / args.repeats * 1000
    speedup = t_serial / t_parallel

    col = 30
    print(f"{'':>{col}}  {'ms / run':>10}")
    print(f"{'Serial ({} x sequential)'.format(args.threads):>{col}}  {t_serial:>10.1f}")
    print(f"{'Parallel ({} threads)'.format(args.threads):>{col}}  {t_parallel:>10.1f}")
    print(f"{'Speedup':>{col}}  {speedup:>10.2f}x")
    print()

    if speedup >= args.threads * 0.5:
        print(f"✓  GIL released during Stretch.process() — {speedup:.1f}x parallel scaling confirmed.")
    elif speedup >= 2:
        print(f"~  Partial speedup ({speedup:.1f}x). GIL may be released but other bottlenecks present.")
    else:
        print(f"✗  No meaningful speedup ({speedup:.1f}x). This build likely does not include the GIL release patch.")


if __name__ == "__main__":
    main()
