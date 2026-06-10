from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch import Tensor
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# INFO: pca functions


def get_scores(
    scalar: StandardScaler,
    number_of_components: int,
    positive_anchor_embeddings: Tensor | np.ndarray,
    negative_anchor_embeddings: Tensor | np.ndarray,
    topic_axis: Tensor | np.ndarray,
    example_topic_speech_embeddings: Tensor | np.ndarray,
    random_control_speech_embeddings: Tensor | np.ndarray,
) -> tuple[PCA, np.ndarray]:
    x_combined = np.vstack(
        tup=(
            topic_axis.reshape(1, -1),
            positive_anchor_embeddings,
            negative_anchor_embeddings,
            example_topic_speech_embeddings,
            random_control_speech_embeddings,
        )
    )
    x_scaled = scalar.fit_transform(x_combined)
    pca = PCA(n_components=number_of_components)
    scores = pca.fit_transform(x_scaled)
    return (pca, scores)


def get_labels(
    topic: str,
    positive_anchor_embeddings: Tensor | np.ndarray,
    negative_anchor_embeddings: Tensor | np.ndarray,
    example_topic_speech_embeddings: Tensor | np.ndarray,
    random_control_speech_embeddings: Tensor | np.ndarray,
) -> np.ndarray:
    return np.array(
        [f"{topic} Semantic Topic Axis"] * 1
        + [f"Positive {topic} Anchor Sentences"] * positive_anchor_embeddings.shape[0]
        + [f"Negative {topic} Anchor Sentences"] * negative_anchor_embeddings.shape[0]
        + [f"{topic} Text Example"] * example_topic_speech_embeddings.shape[0]
        + ["Random Text Example"] * random_control_speech_embeddings.shape[0]
    )


def add_to_plot(pca: PCA, scores: np.ndarray, labels: np.ndarray) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=scores[:, 0],
        y=scores[:, 1],
        hue=labels,
        s=100,
        palette="Set2",
        edgecolor="black",
    )

    plt.title("PCA Comparison", fontsize=25)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=25)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=13, title_fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    return None


def save_pca_plot(
    topic: str,
    scalar: StandardScaler,
    number_of_components: int,
    positive_anchor_embeddings: Tensor | np.ndarray,
    negative_anchor_embeddings: Tensor | np.ndarray,
    topic_axis: Tensor | np.ndarray,
    example_topic_speech_embeddings: Tensor | np.ndarray,
    random_control_speech_embeddings: Tensor | np.ndarray,
    save_location: Path,
    print_plot: bool = False,
) -> None:
    pca, scores = get_scores(
        scalar=scalar,
        number_of_components=number_of_components,
        positive_anchor_embeddings=positive_anchor_embeddings,
        negative_anchor_embeddings=negative_anchor_embeddings,
        topic_axis=topic_axis,
        example_topic_speech_embeddings=example_topic_speech_embeddings,
        random_control_speech_embeddings=random_control_speech_embeddings,
    )
    labels = get_labels(
        topic=topic,
        positive_anchor_embeddings=positive_anchor_embeddings,
        negative_anchor_embeddings=negative_anchor_embeddings,
        example_topic_speech_embeddings=example_topic_speech_embeddings,
        random_control_speech_embeddings=random_control_speech_embeddings,
    )
    add_to_plot(pca=pca, scores=scores, labels=labels)
    if print_plot:
        plt.show()
    else:
        plt.savefig(fname=save_location, dpi=300, bbox_inches="tight")
        plt.close()
    return None
