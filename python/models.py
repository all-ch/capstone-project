from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from torch import Tensor
from pathlib import Path
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

    if weight > 1e-9:
        weighted_mean = np.sum(a=distribution * probability) / np.sum(a=probability)
        weighted_standard_deviation = np.sqrt(
            np.sum(a=probability * (distribution - weighted_mean) ** 2)
            / np.sum(a=probability)
        )
    else:
        weight = weighted_mean = weighted_standard_deviation = 0
    return pd.Series([weighted_mean, weighted_standard_deviation, weight])


# INFO: linear regression functions


def predict_fitted_linear_regression(
    topic: str, group_by: str, dataframe: pd.DataFrame
) -> np.ndarray:
    filtered_dataframe = (
        dataframe.groupby(by=group_by)[f"{topic} mean"].mean().reset_index()
    )
    X, y = filtered_dataframe[[group_by]], filtered_dataframe[f"{topic} mean"]
    model = LinearRegression().fit(X=X, y=y)
    return model.predict(X=X)


def save_linear_regression_with_violin_plot(
    topic: str,
    feature: str,
    y: np.ndarray,
    dataframe: pd.DataFrame,
    save_location: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    X = dataframe[feature].unique()
    violin_data = [
        dataframe[dataframe[feature] == pos][f"{topic} mean"].values for pos in X
    ]
    ax.violinplot(
        dataset=violin_data, positions=X, widths=1, showmeans=True, showextrema=False
    )
    ax.plot(X, y)
    plt.savefig(
        fname=save_location / f"{topic} linear regression with violin",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig=fig)
    return None


def compute_all_linear_regression_plots(
    group_by: str,
    topics: dict[str, dict[str, Path]],
    dataframe: pd.DataFrame,
    save_location: Path,
) -> None:
    for topic in topics.keys():
        prediction = predict_fitted_linear_regression(
            topic=topic, group_by=group_by, dataframe=dataframe
        )
        save_linear_regression_with_violin_plot(
            topic=topic,
            feature=group_by,
            y=prediction,
            dataframe=dataframe,
            save_location=save_location,
        )
    return None
