from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from torch import Tensor
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
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


def plot_gmm_distribution(
    distribution: Tensor | np.ndarray,
    gaussian_mixture_model: GaussianMixture,
    save_location: Path,
    label: str = "Distribution",
    print_plot: bool = False,
) -> None:
    data_reshaped = distribution.reshape(-1, 1)
    gaussian_mixture_model.fit(data_reshaped)

    x_axis = np.linspace(
        distribution.min() - 0.01, distribution.max() + 0.01, 1000
    ).reshape(-1, 1)

    log_density = gaussian_mixture_model.score_samples(x_axis)
    total_density = np.exp(log_density)

    colors = ["green", "purple", "orange"]
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        distribution,
        bins=30,
        alpha=0.7,
        density=True,
        color="cornflowerblue",
        edgecolor="black",
    )

    ax.plot(x_axis, total_density, color="red", lw=2, alpha=0.7, label="Total GMM")

    for j in range(gaussian_mixture_model.n_components):
        mean = gaussian_mixture_model.means_[j][0]
        std = np.sqrt(gaussian_mixture_model.covariances_[j][0][0])
        weight = gaussian_mixture_model.weights_[j]

        pdf = weight * norm.pdf(x_axis, mean, std)
        ax.plot(x_axis, pdf, "--", color=colors[j % len(colors)], alpha=0.7)

    ax.set_title(
        f"Example Religious Speech {label} ({gaussian_mixture_model.n_components} components)"
    )
    ax.set_xlabel(f"BIC: {gaussian_mixture_model.bic(data_reshaped):.5f}")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if print_plot:
        plt.show()
    else:
        plt.savefig(
            fname=save_location,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig=fig)
    return None


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
    print_plot: bool = False,
) -> None:
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(12, 6))
    X = dataframe[feature].unique()
    violin_data = [
        dataframe[dataframe[feature] == pos][f"{topic} mean"].values for pos in X
    ]
    ax.violinplot(
        dataset=violin_data, positions=X, widths=1, showmeans=True, showextrema=False
    )
    ax.plot(X, y, label="Mean Trend")
    ax.set_title(f"{topic} Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{topic} Topic Score")
    ax.legend()
    plt.tight_layout()
    if print_plot:
        plt.show()
    else:
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
    print_plot: bool = False,
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
            print_plot=print_plot,
        )
    return None
