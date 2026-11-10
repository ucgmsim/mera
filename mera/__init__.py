"""Mixed-effects regression analysis (MERA) of ground-motion residuals.

Everything that most users need can be imported from here:

- :func:`run_mera` fits the model to each intensity measure of a table of residuals;
- :class:`MeraResults` holds the tables that it returns, and saves and loads them;
- :func:`mask_too_few_records` makes a mask that leaves out the records of events and
  sites with too few records.

The modules behind them:

- ``analysis.py``: ``run_mera`` and ``MeraResults``, the bookkeeping around the fits;
- ``lmm.py``: ``fit_lmm``, the statistics of one fit;
- ``_lmm``: the numerical engine, written in Rust (``crates/mera_lmm``) and described
  for Python in ``_lmm.pyi``;
- ``utils.py``: ``mask_too_few_records``.
"""

from .analysis import MeraResults, run_mera
from .utils import mask_too_few_records

__all__ = ["MeraResults", "mask_too_few_records", "run_mera"]
