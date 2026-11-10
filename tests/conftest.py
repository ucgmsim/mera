import pandas as pd
import pytest
from example_data import Lme4Fit, load_example_residuals, load_lme4_fits


@pytest.fixture(scope="session")
def example_residuals() -> pd.DataFrame:
    """ln(obs / sim) for every IM of the example data, with prefixed event and site ids."""
    return load_example_residuals()


@pytest.fixture(scope="session")
def lme4_fits() -> dict[tuple[str, bool, bool], Lme4Fit]:
    """lme4's fits for example_data.LME4_FITS."""
    return load_lme4_fits()
