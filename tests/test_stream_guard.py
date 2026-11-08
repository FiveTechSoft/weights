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


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status, self._b, self.headers, self.read_calls = status, body, headers or {}, []

    def read(self, n=-1):
        self.read_calls.append(n)
        return self._b if n < 0 else self._b[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patched(resps):
    import urllib.request
    it = iter(resps)
    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=None: next(it)
    return orig


def test_hf_fetch_charges_every_retry_and_drops_a_reply_that_ignores_range(monkeypatch):
    import urllib.request
    from engine import weight_diff as wd
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)
    whole = _Resp(200, b"x" * 1000)  # ignores Range: 1000 bytes for a 10-byte request, Content-Length absent
    good = _Resp(206, b"y" * 10)
    it = iter([whole, good])
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: next(it))
    b = ByteBudget(1000)
    out = wd.hf_fetch("o/r", "rev", meter=b.reserve)("f.safetensors", 100, 109)
    assert out == b"y" * 10
    assert b.bytes == 20 and b.calls == 2  # the failed attempt is charged too
    assert whole.read_calls == []  # its body was never read


def test_hf_fetch_reads_at_most_the_requested_length_plus_one(monkeypatch):
    import urllib.request
    from engine import weight_diff as wd
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)
    big = _Resp(206, b"z" * 1000)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: big)
    with pytest.raises(IOError):
        wd.hf_fetch("o/r", "rev", retries=1, meter=ByteBudget(10**6).reserve)("f", 0, 9)
    assert big.read_calls == [11]


def test_cap_refusal_is_not_retried_and_nothing_is_downloaded(monkeypatch):
    import urllib.request
    from engine import weight_diff as wd
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not download"))
    with pytest.raises(BudgetExceeded):
        wd.hf_fetch("o/r", "rev", meter=ByteBudget(5).reserve)("f", 0, 9)


def test_memory_abort_reaches_the_caller_and_the_spill_files_go_with_the_temp_dir():
    import os
    import tempfile
    torch = pytest.importorskip("torch")
    from engine import stream_score as ss
    from tests.test_stream_score import _tiny_cfg
    cfg = _tiny_cfg()
    g = MemoryGuard(100, period_s=0.005, reader=lambda: 150.0)
    time.sleep(0.05)

    def load(L, layer):
        g.check()

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(MemoryExceeded):
            ss.stream_nll(cfg, [[1, 2, 3, 4]], lambda ids: torch.zeros(len(ids), cfg.hidden_size), load,
                          torch.ones(cfg.hidden_size), lambda: iter(()), [1], spill_dir=d)
        assert os.listdir(d) == ["h0.pt"]  # written before the abort...
    assert not os.path.exists(d)  # ...and removed with the caller's temp dir
    g.close()
    assert g._stop.is_set()
