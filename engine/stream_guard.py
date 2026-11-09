"""engine/stream_guard.py - hard stops for a streamed scoring run: bytes read over the network, read rate, sampled memory.

  budget = ByteBudget(max_bytes, min_rate_bytes_s=None, grace_s=60.0)
  fetch  = budget.wrap(fetch)          # counts the bytes of every range read; raises BudgetExceeded past max_bytes
  budget.check()                       # call between layers: also enforces the minimum average rate after the grace period
  mem = MemoryGuard(max_anon_mb)       # samples RssAnon in a thread; mem.check() raises MemoryExceeded after a breach
The caller must call check() at least once per layer: a stop is a raised exception, not a log line.
"""
import re
import threading
import time
from typing import Callable


class BudgetExceeded(RuntimeError):
    pass


class MemoryExceeded(RuntimeError):
    pass


class ByteBudget:
    def __init__(self, max_bytes: int, min_rate_bytes_s: float = None, grace_s: float = 60.0, clock=time.monotonic):
        self.max_bytes, self.min_rate, self.grace = max_bytes, min_rate_bytes_s, grace_s
        self.bytes, self.calls, self._clock, self._t0 = 0, 0, clock, clock()
        self._lock = threading.Lock()

    def reserve(self, n: int) -> None:
        """Charge n bytes BEFORE an attempt that may download up to n (use as hf_fetch's meter: every retry is charged again)."""
        with self._lock:
            if self.bytes + n > self.max_bytes:
                raise BudgetExceeded(f"read of {n} bytes would pass the cap of {self.max_bytes} (already {self.bytes})")
            self.bytes += n
            self.calls += 1

    def wrap(self, fetch: Callable) -> Callable:
        """For a fetch that does not meter itself (local files, tests). Not for hf_fetch: pass meter=budget.reserve to it."""
        def counted(f, start, end):
            self.reserve(end - start + 1)
            return fetch(f, start, end)
        return counted

    def check(self) -> None:
        dt = self._clock() - self._t0
        if self.min_rate and dt > self.grace and self.bytes / dt < self.min_rate:
            raise BudgetExceeded(f"average rate {self.bytes / dt:.0f} B/s is under the minimum {self.min_rate:.0f} B/s after {dt:.0f} s")


class MemoryGuard:
    def __init__(self, max_anon_mb: float, period_s: float = 0.2, reader=None):
        self.max, self.peak, self.breached = max_anon_mb, 0.0, False
        self._read = reader or self._rss_anon_mb
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(period_s,), daemon=True)
        self._t.start()

    @staticmethod
    def _rss_anon_mb() -> float:
        return int(re.search(r"RssAnon:\s+(\d+)", open("/proc/self/status").read()).group(1)) / 1024

    def _run(self, period):
        while not self._stop.is_set():
            v = self._read()
            self.peak = max(self.peak, v)
            if v > self.max:
                self.breached = True
            time.sleep(period)

    def check(self) -> None:
        if self.breached:
            raise MemoryExceeded(f"sampled anonymous memory passed {self.max} MB (peak {self.peak:.0f} MB)")

    def close(self) -> None:
        self._stop.set()
