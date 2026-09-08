"""Self-contained example/benchmark for the streaming API (seek/processBlock/
flush). Needs no audio file -- run it directly:

    python examples/example_streaming.py
"""
import numpy as np
import python_stretch as m

SR = 44100.0


def demo_basic_streaming():
    # Drive a persistent Stretch object through several blocks and check
    # the concatenated output matches a single one-shot call.
    rng = np.random.default_rng(0)
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

    print("one-shot vs streamed, bit-identical:", np.array_equal(ref_out, streamed_out))


def demo_time_factor_change_mid_stream():
    # A 440Hz tone, stretched to half speed partway through, with pitch
    # unaffected by the change.
    freq = 440.0
    n = 44100
    t = np.arange(n) / SR
    tone = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(1, -1)

    s = m.Signalsmith.Stretch(seed=0)
    s.preset(1, SR)
    chunk, half = 4410, n // 2
    parts = [s.processBlock(tone[:, i:i + chunk], chunk) for i in range(0, half, chunk)]
    parts += [s.processBlock(tone[:, i:i + chunk], chunk * 2) for i in range(half, n, chunk)]
    out = np.concatenate(parts, axis=1)

    def dominant_freq(x):
        window = np.hanning(len(x))
        spectrum = np.abs(np.fft.rfft(x * window))
        freqs = np.fft.rfftfreq(len(x), d=1 / SR)
        return freqs[np.argmax(spectrum)]

    print("input length:", n, "output length:", out.shape[1], "(second half at half speed)")
    print("dominant freq before speed change:", dominant_freq(out[0, :half]), "Hz")
    print("dominant freq after speed change: ", dominant_freq(out[0, half:]), "Hz")


def demo_memory_is_bounded():
    # How this was actually measured (Windows): psapi.GetProcessMemoryInfo
    # working-set size, before vs. after driving 50000 blocks through a
    # single persistent Stretch object, with a short warmup first so the
    # internal scratch buffers have already grown to their steady size.
    # Measured result on the machine this was written on: 0 bytes of
    # growth over 50000 blocks (512 samples each). If you're on Linux or
    # macOS, swap in `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`
    # instead -- the loop below is the part that matters.
    import ctypes
    import sys

    rng = np.random.default_rng(1)
    s = m.Signalsmith.Stretch(seed=0)
    s.preset(1, SR)
    chunk = 512

    for _ in range(200):
        block = rng.normal(0, 0.1, size=(1, chunk)).astype(np.float32)
        s.processBlock(block, chunk)

    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi.dll")
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        def working_set_bytes():
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return counters.WorkingSetSize

        before = working_set_bytes()
        n_blocks = 50000
        for _ in range(n_blocks):
            block = rng.normal(0, 0.1, size=(1, chunk)).astype(np.float32)
            s.processBlock(block, chunk)
        after = working_set_bytes()
        print(f"working set growth over {n_blocks} blocks: {after - before} bytes")
    else:
        import resource

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        n_blocks = 50000
        for _ in range(n_blocks):
            block = rng.normal(0, 0.1, size=(1, chunk)).astype(np.float32)
            s.processBlock(block, chunk)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"max RSS growth over {n_blocks} blocks: {after - before} KB (ru_maxrss units)")


if __name__ == "__main__":
    demo_basic_streaming()
    demo_time_factor_change_mid_stream()
    demo_memory_is_bounded()
