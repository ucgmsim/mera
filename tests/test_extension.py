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
