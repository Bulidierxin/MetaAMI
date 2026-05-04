from __future__ import annotations

import numpy as np
from sklearn.random_projection import GaussianRandomProjection


def make_random_projection(X_train, X_test, n_components=40, seed=42):
    """Fit Gaussian RP on training data and transform train/test."""
    projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
    return projector.fit_transform(X_train), projector.transform(X_test), projector
