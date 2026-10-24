"""ECOD exactly as in Algorithm 1 of Li et al. (TKDE 2022).

PyOD's ECOD takes the max of (left, right, skew-selected) tail scores *per dimension*
and then sums, which makes the skewness step a no-op (the skew-selected score is
always one of the other two). The paper instead sums each variant over dimensions
first and takes the max of the three totals (Alg. 1, step 7 / Eq. 6). This class
implements the paper's version; everything else mirrors PyOD so the two differ only
in the aggregation step.
"""
import numpy as np
from pyod.models.base import BaseDetector
from scipy.stats import skew
from sklearn.utils import check_array


def _ecdf(X):
    """Column-wise ECDF evaluated at each sample: fraction of values <= x (Eq. 1)."""
    n = X.shape[0]
    ranks = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        ranks[:, j] = np.searchsorted(np.sort(col), col, side="right")
    return ranks / n


class ECODPaper(BaseDetector):
    def __init__(self, contamination=0.1, variant="max"):
        """variant: 'max' (ECOD), 'left' (ECOD-L), 'right' (ECOD-R), 'both' (ECOD-B)."""
        super().__init__(contamination=contamination)
        self.variant = variant

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        X = check_array(X)
        n_new = X.shape[0]
        # like PyOD: estimate ECDFs on train + new points, so every point counts itself
        # and no tail probability is 0
        if hasattr(self, "X_train"):
            X = np.concatenate((self.X_train, X), axis=0)

        U_l = -np.log(_ecdf(X))  # -log F_left  (Eq. 1)
        U_r = -np.log(_ecdf(-X))  # -log F_right (Eq. 2, counts values >= x)
        gamma = np.nan_to_num(skew(X, axis=0))  # Alg. 1, step 3

        O_left = U_l.sum(axis=1)
        O_right = U_r.sum(axis=1)
        O_auto = np.where(gamma < 0, U_l, U_r).sum(axis=1)  # Alg. 1, step 6

        scores = {
            "max": np.maximum.reduce([O_left, O_right, O_auto]),  # Eq. 6
            "left": O_left,
            "right": O_right,
            "both": (O_left + O_right) / 2,
        }[self.variant]
        return scores[-n_new:]
