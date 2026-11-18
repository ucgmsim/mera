import threading

import numpy as np
import pytest

from mera import _lmm


def test_invalid_input_raises_value_error():
    codes = [np.array([0, 1, 1], dtype=np.int64)]
    with pytest.raises(ValueError):
        _lmm.CrossedModel(codes, np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError):
        _lmm.CrossedModel(
            [np.array([0, -1, 1], dtype=np.int64)], np.array([1.0, 2.0, 3.0])
        )
    with pytest.raises(ValueError):
        _lmm.CrossedModel(codes, np.array([1.0, 2.0, 3.0]), np.ones((2, 1)))
    model = _lmm.CrossedModel(codes, np.array([1.0, 2.0, 4.0]))
    with pytest.raises(ValueError):
        model.deviance([-1.0])
    with pytest.raises(ValueError):
        model.deviance([np.nan])
    with pytest.raises(ValueError):
        model.deviance([1.0, 1.0])


def test_collinear_fixed_effects_raise_runtime_error():
    codes = [np.array([0, 0, 1, 1], dtype=np.int64)]
    # an all-zero covariate makes X'V^-1X singular
    x = np.column_stack([np.ones(4), np.zeros(4)])
    with pytest.raises(RuntimeError, match="collinear"):
        _lmm.CrossedModel(codes, np.arange(4.0), x).deviance([1.0])


def test_other_threads_run_during_a_calculation() -> None:
    """The engine releases Python's lock (the GIL) while it calculates, so fits in
    separate threads use separate cores.

    The evidence here is that a second thread can reach a model while a calculation on it
    is in progress, which is refused with an error. If the lock were held, the second
    thread could never get there: the calls would take turns, and none would be refused.
    """
    rng = np.random.default_rng(0)
    n_events, n_sites, records_per_event = 1500, 750, 20
    events = np.repeat(np.arange(n_events, dtype=np.int64), records_per_event)
    sites = rng.integers(0, n_sites, events.size, dtype=np.int64)
    model = _lmm.CrossedModel([events, sites], rng.normal(size=events.size))
    start_together = threading.Barrier(2)
    refused = []

    def evaluate() -> None:
        start_together.wait()
        for _ in range(10):
            try:
                model.deviance([1.0, 1.0])
            except RuntimeError as error:
                refused.append(str(error))

    threads = [threading.Thread(target=evaluate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert refused and all("borrowed" in message for message in refused)
