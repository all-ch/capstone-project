from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import seaborn as sns


def get_anchor_embeds(location: str, model: SentenceTransformer) -> Tensor | np.ndarray:
    return model.encode(pd.read_csv(location, header=None)[0].tolist())


def get_sent_embeds(sent: list[str], model: SentenceTransformer) -> Tensor | np.ndarray:
    return model.encode(sent)


def avg_vec(embeddings: Tensor | np.ndarray) -> Tensor | np.ndarray:
    return np.mean(embeddings, axis=0)


def get_anchor_axis(
    pos_embeds: Tensor | np.ndarray, neg_embeds: Tensor | np.ndarray
) -> Tensor | np.ndarray:
    return avg_vec(pos_embeds) - avg_vec(neg_embeds)


def split_speech(text: str, nlp: Language) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def find_speech(speaker: str, cyear: int | None = None) -> str:
    speech = (
        data[data.speaker == speaker]["speech"]
        if cyear == None
        else data[(data.speaker == speaker) & (data.cyear == cyear)]["speech"]
    )
    return speech.item()


def get_speech_embeds(
    nlp: Language, model: SentenceTransformer, speaker: str, cyear: int | None = None
) -> Tensor | np.ndarray:
    return get_sent_embeds(split_speech(find_speech(speaker, cyear), nlp), model)


if __name__ == "__main__":
    # load core data and models
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    data = pd.read_csv("../data/speeches.csv")
    nlp = spacy.load("en_core_web_sm")
    scalar = StandardScaler()

    # religious axis embeddings
    religion_neg = get_anchor_embeds("../data/religion_neg.csv", model)
    religion_pos = get_anchor_embeds("../data/religion_pos.csv", model)
    religion_axis = get_anchor_axis(religion_pos, religion_neg)

    # example text embeddings
    religious_embeds = get_speech_embeds(nlp, model, "Michael Gold ")
    rand_embeds = get_speech_embeds(nlp, model, "Akira Morita", 1997)

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

    plt.show()
