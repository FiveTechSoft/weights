import time
import pytest
from engine.stream_guard import BudgetExceeded, ByteBudget, MemoryExceeded, MemoryGuard


def test_counts_bytes_and_stops_before_the_read_that_passes_the_cap():
    calls = []
    b = ByteBudget(100)
    f = b.wrap(lambda name, s, e: calls.append((s, e)) or b"x")
    f("a", 0, 59)
    assert b.bytes == 60 and b.calls == 1
    with pytest.raises(BudgetExceeded):
        f("a", 0, 49)
    assert len(calls) == 1 and b.bytes == 60  # the refused read neither happened nor counted


def test_minimum_rate_only_after_grace():
    t = [0.0]
    b = ByteBudget(10**9, min_rate_bytes_s=100, grace_s=10, clock=lambda: t[0])
    b.wrap(lambda *a: b"")("a", 0, 99)  # 100 bytes
    t[0] = 5
    b.check()  # inside the grace period
    t[0] = 20  # 5 B/s
    with pytest.raises(BudgetExceeded):
        b.check()


def test_memory_guard_raises_after_breach():
    g = MemoryGuard(100, period_s=0.01, reader=lambda: 150.0)
    time.sleep(0.1)
    with pytest.raises(MemoryExceeded):
        g.check()
    g.close()


def test_memory_guard_quiet_under_cap():
    g = MemoryGuard(100, period_s=0.01, reader=lambda: 50.0)
    time.sleep(0.05)
    g.check()
    assert g.peak == 50.0
    g.close()
