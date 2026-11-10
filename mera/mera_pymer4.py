"""Deprecated: import ``run_mera`` and ``MeraResults`` from :mod:`mera` instead.

Kept so that ``from mera.mera_pymer4 import run_mera`` keeps working; mera no longer uses
pymer4.
"""

import warnings

from .analysis import MeraResults, run_mera

warnings.warn(
    "mera.mera_pymer4 is deprecated; import run_mera and MeraResults from mera instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MeraResults", "run_mera"]
