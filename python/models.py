from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from torch import Tensor
import pandas as pd
import numpy as np


# INFO: gaussian mixture model functions


def calculate_weighted_metrics_from_gaussian_mixture_model(
    distribution: Tensor | np.ndarray, gaussian_mixture_model: GaussianMixture
) -> pd.Series:
    gaussian_mixture_model.fit(distribution.reshape(-1, 1))
    topic_index = np.argmax(gaussian_mixture_model.means_.flatten())

    probability = gaussian_mixture_model.predict_proba(distribution.reshape(-1, 1))[
        :, topic_index
    ]

    weight = np.sum(a=probability / len(distribution))

    if weight > 0:
        weighted_mean = np.sum(a=distribution * probability) / np.sum(a=probability)
        weighted_standard_deviation = np.sum(
            a=np.sqrt(probability * (distribution - weighted_mean) ** 2)
            / np.sum(a=probability)
        )
    else:
        weight = weighted_mean = weighted_standard_deviation = 0
    return pd.Series(data=[weighted_mean, weighted_standard_deviation, weight])
