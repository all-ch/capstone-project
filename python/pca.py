from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import embeddings


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
    x_combined = np.vstack(
        (
            religion_axis.reshape(1, -1),
            religion_pos,
            religion_neg,
            religious_embeds,
            rand_embeds,
        )
    )
    labels = np.array(
        ["Aggregate Religion Vertex"] * 1
        + ["Positive Religion Anchor Phrases"] * religion_pos.shape[0]
        + ["Negative Religion Anchor Phrases"] * religion_neg.shape[0]
        + ["Religious Text Example"] * religious_embeds.shape[0]
        + ["Random Text Example"] * rand_embeds.shape[0]
    )
    x_scaled = scalar.fit_transform(x_combined)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(x_scaled)

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

    # plt.savefig("visual/pca.png", dpi=300, bbox_inches="tight")
    plt.show()
