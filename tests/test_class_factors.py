"""A separate standard deviation for each class of a grouping factor: one factor per
class, in which the observations of the other classes have no level."""

import numpy as np
import pandas as pd
import pytest
from example_data import (
    CLASS_FITS,
    SITE_CLASSES,
    Lme4Fit,
    add_site_classes,
    assert_matches_dense,
    assert_matches_lme4,
    class_inputs,
)

from mera.example import maths_walkthrough as walkthrough
from mera.lmm import fit_lmm, split_by_class


def class_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A response, the event and site of each record, the class of its site, and
    indicator covariates for the classes: 40 events, 24 sites in three classes."""
    y, events, sites = walkthrough.make_example_data(n_sites=24, sites_per_event=10)
    y, site_class, covariates = add_site_classes(y, sites)
    return y, events, sites, site_class, covariates


def test_split_by_class_gives_one_factor_per_class() -> None:
    sites = np.array(["s1", "s2", "s3", "s1", "s4"])
    classes = np.array(["rock", "soil", "rock", "rock", "soil"])
    rock, soil = split_by_class(sites, classes)  # in the sorted order of the classes
    # tolist() shows the entries without a level as None
    assert rock.tolist() == ["s1", None, "s3", "s1", None]
    assert soil.tolist() == [None, "s2", None, None, "s4"]


@pytest.mark.parametrize("reml", [True])
@pytest.mark.parametrize("intercept", [True, False])
def test_a_standard_deviation_per_class_matches_the_dense_implementation(
    intercept: bool, reml: bool
) -> None:
    y, events, sites, site_class, covariates = class_data()
    groups = [events, *split_by_class(sites, site_class)]
    x = np.column_stack([np.ones((y.size, int(intercept))), covariates])
    z, level_factor, levels = walkthrough.indicator_matrix(groups)
    model = walkthrough.Model(y, x, z, level_factor)
    dense = walkthrough.estimates(walkthrough.find_theta(model), model)

    fit = fit_lmm(y, groups, intercept=intercept, covariates=covariates, reml=reml)

    assert fit.theta.shape == (1 + len(SITE_CLASSES),)
    # each class's factor has that class's sites as its levels, and no others
    for k, name in enumerate(SITE_CLASSES):
        np.testing.assert_array_equal(
            fit.levels[1 + k], np.unique(sites[site_class == name])
        )
    assert_matches_dense(fit, dense, levels, y)


@pytest.mark.parametrize("im, reml", CLASS_FITS)
def test_a_standard_deviation_per_class_matches_lme4(
    example_residuals: pd.DataFrame,
    lme4_class_fits: dict[tuple[str, bool], Lme4Fit],
    im: str,
    reml: bool,
) -> None:
    """lme4's ``y ~ 1 + low + mid + (1|event) + (0 + high|site) + (0 + low|site) +
    (0 + mid|site)``, by REML and by maximum likelihood, on the example data."""
    y, groups, covariates = class_inputs(example_residuals, im)
    fit = fit_lmm(y, groups, reml=reml, covariates=covariates)
    assert fit.converged
    assert_matches_lme4(fit, lme4_class_fits[im, reml], groups, covariates)


def test_an_observation_gets_no_term_from_a_factor_it_is_not_in() -> None:
    y, events, sites, site_class, covariates = class_data()
    fit = fit_lmm(
        y, [events, *split_by_class(sites, site_class)], covariates=covariates
    )
    x = np.column_stack([np.ones(y.size), covariates])
    expected = x @ fit.beta + fit.ranef[0][np.searchsorted(fit.levels[0], events)]
    for k, name in enumerate(SITE_CLASSES):
        in_class = site_class == name
        position = np.searchsorted(fit.levels[1 + k], sites[in_class])
        expected[in_class] += fit.ranef[1 + k][position]
    np.testing.assert_allclose(fit.fitted, expected)


def test_a_factor_that_every_observation_is_in_is_an_ordinary_factor() -> None:
    """A masked array with nothing masked is the same as a plain array."""
    y, events, sites, _, _ = class_data()
    plain = fit_lmm(y, [events, sites])
    masked = fit_lmm(y, [events, np.ma.masked_array(sites, mask=False)])
    np.testing.assert_array_equal(masked.theta, plain.theta)
    np.testing.assert_array_equal(masked.fitted, plain.fitted)


def test_a_shared_dummy_level_agrees_under_reml_but_not_ml() -> None:
    """Before a factor could leave observations out, the other classes' records were
    sent to one shared dummy level of each class's factor. With an offset per class
    among the fixed effects, that leaves the REML criterion, theta and beta as they
    should be, but not the ML ones, nor the standard errors of beta."""
    y, events, sites, site_class, covariates = class_data()
    exact_groups = [events, *split_by_class(sites, site_class)]
    dummy_groups = [events] + [
        np.where(site_class == name, sites, "other classes") for name in SITE_CLASSES
    ]
    exact = fit_lmm(y, exact_groups, covariates=covariates)
    dummy = fit_lmm(y, dummy_groups, covariates=covariates)
    assert dummy.deviance == pytest.approx(exact.deviance, abs=1e-7)
    np.testing.assert_allclose(dummy.theta, exact.theta, atol=1e-5)
    np.testing.assert_allclose(dummy.beta, exact.beta, atol=1e-6)
    # every standard error is too large: by a third to three times here
    assert np.all(dummy.beta_se > 1.2 * exact.beta_se)

    exact_ml = fit_lmm(y, exact_groups, covariates=covariates, reml=False)
    dummy_ml = fit_lmm(y, dummy_groups, covariates=covariates, reml=False)
    assert dummy_ml.deviance > exact_ml.deviance + 1.0


def test_rejects_invalid_class_factors() -> None:
    y, events, sites, site_class, _ = class_data()
    with pytest.raises(ValueError, match="one class per label"):
        split_by_class(sites, site_class[:-1])
    with pytest.raises(ValueError, match="missing"):
        split_by_class(sites, np.where(np.arange(y.size) == 3, None, site_class))
    # a factor that no observation is in
    nobody = np.ma.masked_array(sites, mask=True)
    with pytest.raises(ValueError, match="more than one level"):
        fit_lmm(y, [events, nobody])
    # a class with a single site: one level cannot have a standard deviation
    one_site = np.ma.masked_array(sites, mask=sites != sites[0])
    with pytest.raises(ValueError, match="more than one level"):
        fit_lmm(y, [events, one_site])
    # a missing label where the observation is in the factor is still an error ...
    with_missing = np.ma.masked_array(sites.astype(object), mask=site_class != "high")
    with_missing[np.flatnonzero(site_class == "high")[0]] = None
    with pytest.raises(ValueError, match="missing"):
        fit_lmm(y, [events, with_missing])
    # ... but not where it is left out of the factor
    left_out = np.ma.masked_array(sites.astype(object), mask=site_class != "high")
    left_out.data[np.flatnonzero(site_class != "high")[0]] = None
    assert fit_lmm(y, [events, left_out]).converged
