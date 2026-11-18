import threading

import numpy as np
import pytest

from mera import _lmm


def test_invalid_input_raises_value_error():
    codes = [np.array([0, 1, 1], dtype=np.int64)]
    no_x = np.empty((3, 0))  # no fixed effects
    y = np.array([1.0, 2.0, 4.0])
    with pytest.raises(ValueError):
        _lmm.CrossedModel(codes, [2], np.array([1.0, np.nan, 3.0]), no_x)
    with pytest.raises(ValueError):
        _lmm.CrossedModel(codes, [2], y, np.ones((2, 1)))
    model = _lmm.CrossedModel(codes, [2], y, no_x)
    with pytest.raises(ValueError):
        model.deviance([-1.0])
    with pytest.raises(ValueError):
        model.deviance([np.nan])
    with pytest.raises(ValueError):
        model.deviance([1.0, 1.0])


def test_codes_must_be_levels_or_minus_one():
    """A code is the number of a level, from 0 to one less than the number of levels of
    its factor, or -1 for an observation that has no level in that factor."""
    no_x = np.empty((3, 0))
    y = np.array([1.0, 2.0, 4.0])

    def model(codes: list[int], n_levels: int) -> _lmm.CrossedModel:
        return _lmm.CrossedModel([np.array(codes, dtype=np.int64)], [n_levels], y, no_x)

    assert np.isfinite(model([0, 1, -1], 2).deviance([1.0]))
    with pytest.raises(ValueError, match="-1"):
        model([0, 1, -2], 2)
    with pytest.raises(ValueError, match="not one of its 2 levels"):
        model([0, 1, 2], 2)
    # A far larger code must be an error too, not an attempt to allocate that many levels.
    with pytest.raises(ValueError, match="not one of its 2 levels"):
        model([0, 1, 10**12], 2)
    with pytest.raises(ValueError):
        _lmm.CrossedModel([np.array([0, 1, 1], dtype=np.int64)], [2, 2], y, no_x)


def test_collinear_fixed_effects_raise_runtime_error():
    codes = [np.array([0, 0, 1, 1], dtype=np.int64)]
    # an all-zero covariate makes X'V^-1X singular
    x = np.column_stack([np.ones(4), np.zeros(4)])
    with pytest.raises(RuntimeError, match="collinear"):
        _lmm.CrossedModel(codes, [2], np.arange(4.0), x).deviance([1.0])


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
    y = rng.normal(size=events.size)
    model = _lmm.CrossedModel(
        [events, sites], [n_events, n_sites], y, np.empty((y.size, 0))
    )
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
