"""Tests for the alternative link functions of the Bayesian fit."""

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest

from eci import build_bayesian_model
from eci.links import LINKS, link_mean

ETA = np.linspace(-25, 25, 501)
STUKEL_ALPHAS = [(0.0, 0.0), (0.6, -0.6), (-1.2, 0.9), (1e-8, -1e-8)]


def _cases():
    for link in LINKS:
        for alphas in STUKEL_ALPHAS if link == "stukel" else [(None, None)]:
            yield link, alphas


@pytest.mark.parametrize("link,alphas", list(_cases()))
def test_link_is_a_rising_curve_in_the_unit_interval(link, alphas):
    mu = link_mean(link, ETA, *alphas)
    assert np.isfinite(mu).all()
    assert (mu >= 0).all() and (mu <= 1).all()
    assert (np.diff(mu) >= 0).all()
    assert mu[0] < 0.1 and mu[-1] > 0.9


@pytest.mark.parametrize("link,alphas", list(_cases()))
def test_pytensor_matches_numpy_with_finite_gradients(link, alphas):
    eta, upper, lower = pt.dvector(), pt.dscalar(), pt.dscalar()
    mu = link_mean(link, eta, upper, lower, xp=pt)
    wrt = [eta, upper, lower] if link == "stukel" else [eta]
    f = pytensor.function([eta, upper, lower], [mu, *pt.grad(mu.sum(), wrt)],
                          on_unused_input="ignore")
    out = f(ETA, alphas[0] or 0.0, alphas[1] or 0.0)
    np.testing.assert_allclose(out[0], link_mean(link, ETA, *alphas), atol=1e-12)
    assert all(np.isfinite(o).all() for o in out)


def test_stukel_at_zero_is_the_logistic():
    np.testing.assert_allclose(
        link_mean("stukel", ETA, 0.0, 0.0), link_mean("logit", ETA), atol=1e-12
    )


def test_stukel_alpha_sign_sets_each_tail_separately():
    logit = link_mean("logit", ETA)
    thin_upper = link_mean("stukel", ETA, 0.5, 0.0)
    fat_lower = link_mean("stukel", ETA, 0.0, -0.5)
    above, below = ETA > 1, ETA < -1
    assert (thin_upper[above] > logit[above]).all()
    np.testing.assert_allclose(thin_upper[below], logit[below])
    assert (fat_lower[below] > logit[below]).all()
    np.testing.assert_allclose(fat_lower[above], logit[above])


def test_unknown_link_raises():
    with pytest.raises(ValueError, match="unknown link"):
        link_mean("identity", ETA)


def _toy_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {"model_id": f"m{m}", "Model": f"Model {m}", "benchmark_id": f"b{b}",
         "benchmark": f"Bench {b}", "performance": float(rng.uniform(0.05, 0.95))}
        for m in range(6) for b in range(4)
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("link", LINKS)
def test_model_builds_with_finite_log_probability(link):
    model = build_bayesian_model(_toy_data(), link=link)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert ("link_alpha" in model.named_vars) == (link == "stukel")


def test_model_rejects_unknown_link():
    with pytest.raises(ValueError, match="unknown link"):
        build_bayesian_model(_toy_data(), link="identity")
