import time
import numpy as np


class PerfSample:
    def __init__(self, perf_tracker):
        self.perf_tracker = perf_tracker
        self.start_time = None

    def __enter__(self):
        self.start_time = time.monotonic_ns()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        duration = time.monotonic_ns() - self.start_time
        self.perf_tracker.samples.append(duration)


class PerfMonitor:
    """PerfMonitor is a context manager that tracks the performance of a block of code by taking several samples.

    Example:
        tracker = PerfMonitor("MyOperation")
        for _ in range(10):
            with tracker.sample():
                // Do something

        print(tracker.summary_str())
    """

    def __init__(self, name):
        self.name = name
        self.samples = []

    def sample(self) -> PerfSample:
        """Returns a context manager that will record the duration of the block it wraps."""
        return PerfSample(self)

    def summary_str(self) -> str:
        """Returns a string summarizing the performance of the tracked operation."""
        if not self.samples:
            return f"{self.name} performance: N=0"

        durations_ns = np.array(self.samples)

        return (
            f"{self.name} performance: N={len(durations_ns)} | "
            + f"Min={self._format_duration(np.min(durations_ns))} | "
            + f"Max={self._format_duration(np.max(durations_ns))} | "
            + f"Median={self._format_duration(np.median(durations_ns))} | "
            + f"P90={self._format_duration(np.percentile(durations_ns, 90))}"
        )

    def _format_duration(self, duration_ns: int) -> str:
        units = [
            ("ns", 1),
            ("μs", 1000),
            ("ms", 1000_000),
            ("s", 1000_000_000),
            ("min", 60 * 1000_000_000),
        ]

        for unit, divisor in reversed(units):
            if duration_ns >= divisor:
                return f"{duration_ns/divisor:.2f} {unit}"

        return f"{duration_ns:.2f} ns"
