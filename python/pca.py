from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import embeddings


def get_scores(
    scalar: StandardScaler,
    n: int,
    pos_vec: Tensor | np.ndarray,
    neg_vec: Tensor | np.ndarray,
    axis: Tensor | np.ndarray,
    pos_embeds: Tensor | np.ndarray,
    neg_embeds: Tensor | np.ndarray,
) -> tuple[PCA, np.ndarray]:
    x_combined = np.vstack(
        (axis.reshape(1, -1), pos_vec, neg_vec, pos_embeds, neg_embeds)
    )
    x_scaled = scalar.fit_transform(x_combined)
    pca = PCA(n_components=n)
    scores = pca.fit_transform(x_scaled)
    return (pca, scores)


def get_labels(
    topic: str,
    pos_vec: Tensor | np.ndarray,
    neg_vec: Tensor | np.ndarray,
    pos_embeds: Tensor | np.ndarray,
    neg_embeds: Tensor | np.ndarray,
) -> np.ndarray:
    return np.array(
        [f"Aggregate {topic} Vertex"] * 1
        + [f"Positive {topic} Anchor Phrases"] * pos_vec.shape[0]
        + [f"Negative {topic} Anchor Phrases"] * neg_vec.shape[0]
        + [f"{topic} Text Example"] * pos_embeds.shape[0]
        + ["Random Text Example"] * neg_embeds.shape[0]
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
    plt.title("PCA Comparison", fontsize=15)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)


if __name__ == "__main__":
    # load core data and models
    model, data, nlp = embeddings.init_models(
        "sentence-transformers/all-mpnet-base-v2",
        "data/processed/speeches.csv",
        "en_core_web_sm",
    )
    scalar = StandardScaler()

    # religion section
    religion_pos, religion_neg, religion_axis = embeddings.init_vec(
        "data/anchors/religion_pos_phrases.csv",
        "data/anchors/religion_neg_phrases.csv",
        model,
    )

    # example text embeddings
    religion_embeds = embeddings.get_speech_embeds(nlp, model, data, "Michael Gold ")
    rand_embeds = embeddings.get_speech_embeds(nlp, model, data, "Akira Morita", 1997)

    # pca
    religion_pca, religion_scores = get_scores(
        scalar,
        2,
        religion_pos,
        religion_neg,
        religion_axis,
        religion_embeds,
        rand_embeds,
    )

    religion_labels = get_labels(
        "Religion", religion_pos, religion_neg, religion_embeds, rand_embeds
    )
    # plot
    add_to_plot(religion_pca, religion_scores, religion_labels)

    plt.savefig("visual/pca.png", dpi=300, bbox_inches="tight")
    plt.close()
