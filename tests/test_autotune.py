import time

import heliax as hx


def test_autotune_selects_fastest_candidate():
    def fast():
        time.sleep(0.0001)

    def slow():
        time.sleep(0.002)

    winner, timings = hx.select_fastest({"fast": fast, "slow": slow}, warmup=0, repeats=2)
    assert winner == "fast"
    assert set(timings) == {"fast", "slow"}
    report = hx.autotune_report({"fast": fast, "slow": slow}, warmup=0, repeats=2)
    assert report["winner"] == "fast"
    assert hx.time_callable(fast, warmup=0, repeats=1) > 0
