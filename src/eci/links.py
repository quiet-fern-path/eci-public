"""
Link functions for the Bayesian ECI fit.

The model's mean score is ``mu = F(eta)`` with
``eta = slope * (capability - difficulty)``. ``F`` is the inverse link: the
logistic curve by default, or one of the alternatives here. Every ``F`` rises
from 0 to 1 in ``eta``, so capability keeps its direction under all of them.

- ``logit``: ``1 / (1 + exp(-eta))``, the default.
- ``probit``: the standard normal CDF; thinner tails than the logistic.
- ``cauchit``: the standard Cauchy CDF; much heavier tails.
- ``cloglog``: ``1 - exp(-exp(eta))``; approaches 1 fast and 0 slowly.
- ``loglog``: ``exp(-exp(-eta))``; the mirror image of cloglog.
- ``stukel``: Stukel's (1988) two-tail generalisation of the logistic,
  ``sigmoid(h(eta))``, with one learned shape parameter per tail shared by
  all benchmarks: ``link_alpha[upper]`` bends the curve above 0.5 and
  ``link_alpha[lower]`` the curve below it. Zero gives the logistic; positive
  values thin that tail (it saturates sooner), negative values fatten it.

Only the symmetric links (logit, probit, cauchit, and stukel with equal
alphas) put ``mu = 0.5`` at ``eta = 0``; cloglog crosses ``eta = 0`` at 0.632
and loglog at 0.368, so a benchmark's difficulty is the capability scoring
that much rather than one half.

Each function works on numpy arrays and on PyTensor variables alike: pass the
array module as ``xp`` (``numpy`` or ``pytensor.tensor``).
"""

import numpy as np
from scipy import special

LINKS = ("logit", "probit", "cauchit", "cloglog", "loglog", "stukel")

# Name and coordinates of the Stukel link's two shape parameters.
STUKEL_PARAM = "link_alpha"
STUKEL_TAILS = ("upper", "lower")

# exp() arguments are capped here, far past where every curve has saturated.
_MAX_EXP = 30.0
# Below this the Stukel tail transform switches to its expansion around 0.
_ALPHA_EPS = 1e-6


def _stukel_tail(x, alpha, xp):
    """Stukel's transform of ``x >= 0`` for one tail.

    ``(exp(alpha x) - 1) / alpha`` for alpha > 0, ``-log(1 - alpha x) / alpha``
    for alpha < 0, and ``x`` at 0, where both meet with matching first
    derivative in alpha. The unused branch is fed a harmless alpha so that it
    stays finite, and so do its gradients.
    """
    positive = alpha > _ALPHA_EPS
    negative = alpha < -_ALPHA_EPS
    a_pos = xp.where(positive, alpha, 1.0)
    a_neg = xp.where(negative, alpha, -1.0)
    grown = xp.expm1(xp.minimum(a_pos * x, _MAX_EXP)) / a_pos
    shrunk = -xp.log1p(-a_neg * x) / a_neg
    return xp.where(positive, grown, xp.where(negative, shrunk, x + 0.5 * alpha * x**2))


def stukel_h(eta, alpha_upper, alpha_lower, xp=np):
    """Stukel's ``h(eta)``: the logit of the mean under the stukel link."""
    x = abs(eta)
    return xp.where(
        eta >= 0, _stukel_tail(x, alpha_upper, xp), -_stukel_tail(x, alpha_lower, xp)
    )


def _sigmoid(x, xp):
    # Clipped far past saturation so exp() cannot overflow
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -500.0, 500.0)))


def link_mean(link: str, eta, alpha_upper=None, alpha_lower=None, xp=np):
    """The mean score ``F(eta)`` under ``link``.

    ``alpha_upper`` and ``alpha_lower`` are the stukel link's shape parameters
    and are ignored by the others. ``xp`` is ``numpy`` or ``pytensor.tensor``.
    """
    if link == "logit":
        return _sigmoid(eta, xp)
    if link == "probit":
        # numpy has no erfc of its own
        erfc = getattr(xp, "erfc", special.erfc)
        return 0.5 * erfc(-eta / np.sqrt(2.0))
    if link == "cauchit":
        return 0.5 + xp.arctan(eta) / np.pi
    if link == "cloglog":
        return -xp.expm1(-xp.exp(xp.minimum(eta, _MAX_EXP)))
    if link == "loglog":
        return xp.exp(-xp.exp(xp.minimum(-eta, _MAX_EXP)))
    if link == "stukel":
        return _sigmoid(stukel_h(eta, alpha_upper, alpha_lower, xp), xp)
    raise ValueError(f"unknown link {link!r}; choose one of {', '.join(LINKS)}")
