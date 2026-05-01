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


if __name__ == "__main__":
    # load core data and models
    model, data, nlp = embeddings.init_models(
        "sentence-transformers/all-mpnet-base-v2",
        "data/processed/speeches.csv",
        "en_core_web_sm",
    )
    scalar = StandardScaler()

    religion_pos, religion_neg, religion_axis = embeddings.init_vec(
        "data/anchors/religion_pos_phrases.csv",
        "data/anchors/religion_neg_phrases.csv",
        model,
    )

    # example text embeddings
    religious_embeds = embeddings.get_speech_embeds(nlp, model, data, "Michael Gold ")
    rand_embeds = embeddings.get_speech_embeds(nlp, model, data, "Akira Morita", 1997)

    # PCA
    pca, scores = get_scores(
        scalar,
        2,
        religion_pos,
        religion_neg,
        religion_axis,
        religious_embeds,
        rand_embeds,
    )
    labels = np.array(
        ["Aggregate Religion Vertex"] * 1
        + ["Positive Religion Anchor Phrases"] * religion_pos.shape[0]
        + ["Negative Religion Anchor Phrases"] * religion_neg.shape[0]
        + ["Religious Text Example"] * religious_embeds.shape[0]
        + ["Random Text Example"] * rand_embeds.shape[0]
    )

    # plot
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

    plt.savefig("visual/pca.png", dpi=300, bbox_inches="tight")
    plt.close()
