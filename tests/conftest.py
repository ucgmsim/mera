import pandas as pd
import pytest
from example_data import Lme4Fit, load_example_residuals, load_lme4_reference


@pytest.fixture(scope="session")
def example_residuals() -> pd.DataFrame:
    """ln(obs / sim) for every IM of the example data, with prefixed event and site ids."""
    return load_example_residuals()


@pytest.fixture(scope="session")
def lme4_fits() -> dict[tuple[str, bool, bool], Lme4Fit]:
    """lme4's fits for example_data.LME4_FITS, keyed like it."""
    return load_lme4_reference("", ["data", "intercept", "site_term"])


@pytest.fixture(scope="session")
def lme4_covariate_fits() -> dict[tuple[str, bool], Lme4Fit]:
    """lme4's fits for example_data.COVARIATE_FITS, keyed like it."""
    return load_lme4_reference("covariate_", ["im", "reml"])


@pytest.fixture(scope="session")
def lme4_class_fits() -> dict[tuple[str, bool], Lme4Fit]:
    """lme4's fits for example_data.CLASS_FITS, keyed like it."""
    return load_lme4_reference("class_", ["im", "reml"])
