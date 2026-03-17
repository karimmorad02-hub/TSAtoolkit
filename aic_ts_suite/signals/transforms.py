"""
Variance-stabilising transforms for time-series data.

Available transforms
--------------------
* **LogTransform**   – natural-log (with offset for non-positive data)
* **SqrtTransform**  – square-root (with offset)
* **BoxCoxTransform** – Box-Cox with auto-estimated λ via MLE

Each transform exposes a consistent ``apply`` / ``inverse`` interface and
stores the parameters needed for back-transformation (e.g. λ, offset).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import inv_boxcox  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


# ======================================================================
# Base class
# ======================================================================
class BaseTransform(ABC):
    """Interface contract for all variance-stabilising transforms."""

    @abstractmethod
    def apply(self, series: pd.Series) -> pd.Series:
        """Apply the forward transform."""

    @abstractmethod
    def inverse(self, series: pd.Series) -> pd.Series:
        """Apply the inverse (back) transform."""


# ======================================================================
# Log transform
# ======================================================================
class LogTransform(BaseTransform):
    """
    Natural-log transform: ``y = ln(x + offset)``.

    If the data contain values ≤ 0 the minimal shift ``offset = |min| + 1``
    is applied automatically unless the caller supplies one.
    """

    def __init__(self, offset: Optional[float] = None) -> None:
        self.offset = offset  # determined on first call if None
        self._fitted = False

    def apply(self, series: pd.Series) -> pd.Series:
        s = series.dropna()
        if self.offset is None:
            if s.min() <= 0:
                self.offset = float(abs(s.min()) + 1)
                logger.info("LogTransform auto-offset: %.4f", self.offset)
            else:
                self.offset = 0.0
        self._fitted = True
        return np.log(series + self.offset)

    def inverse(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call .apply() before .inverse().")
        return np.exp(series) - self.offset


# ======================================================================
# Square-root transform
# ======================================================================
class SqrtTransform(BaseTransform):
    """
    Square-root transform: ``y = sqrt(x + offset)``.

    Auto-offset is computed when data contain negative values.
    """

    def __init__(self, offset: Optional[float] = None) -> None:
        self.offset = offset
        self._fitted = False

    def apply(self, series: pd.Series) -> pd.Series:
        s = series.dropna()
        if self.offset is None:
            if s.min() < 0:
                self.offset = float(abs(s.min()) + 1)
                logger.info("SqrtTransform auto-offset: %.4f", self.offset)
            else:
                self.offset = 0.0
        self._fitted = True
        return np.sqrt(series + self.offset)

    def inverse(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call .apply() before .inverse().")
        return series**2 - self.offset


# ======================================================================
# Box-Cox transform (λ estimated via MLE)
# ======================================================================
class BoxCoxTransform(BaseTransform):
    """
    Box-Cox transform with MLE-estimated λ.

    .. math::

        y = \\frac{x^\\lambda - 1}{\\lambda}  \\quad (\\lambda \\neq 0)
        y = \\ln(x)                             \\quad (\\lambda = 0)

    An auto-offset is added when data contain values ≤ 0 (Box-Cox requires
    strictly positive input).
    """

    def __init__(
        self,
        lmbda: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> None:
        self.lmbda = lmbda
        self.offset = offset
        self._fitted = False

    def apply(self, series: pd.Series) -> pd.Series:
        s = series.dropna()

        # Guarantee positivity
        if self.offset is None:
            if s.min() <= 0:
                self.offset = float(abs(s.min()) + 1)
                logger.info("BoxCoxTransform auto-offset: %.4f", self.offset)
            else:
                self.offset = 0.0

        shifted = (series + self.offset).dropna().values

        if self.lmbda is None:
            transformed, fitted_lmbda = sp_stats.boxcox(shifted)
            self.lmbda = float(fitted_lmbda)
            logger.info("BoxCox λ estimated via MLE: %.6f", self.lmbda)
        else:
            transformed = sp_stats.boxcox(shifted, lmbda=self.lmbda)

        self._fitted = True

        # Rebuild with original index (NaNs stay NaN)
        result = pd.Series(np.nan, index=series.index, name=series.name)
        result.loc[series.notna()] = transformed
        return result

    def inverse(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call .apply() before .inverse().")
        arr = series.values.astype(float)
        original = inv_boxcox(arr, self.lmbda) - self.offset
        return pd.Series(original, index=series.index, name=series.name)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def params(self) -> dict:
        return {"lmbda": self.lmbda, "offset": self.offset}
