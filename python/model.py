from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from spacy.language import Language
from torch import Tensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import embeddings
import spacy


def compute_speech_topic_score(
    sentence_embeddings: np.ndarray | Tensor, topic_vector: np.ndarray | Tensor
) -> float:
    return cosine_similarity(sentence_embeddings, topic_vector.reshape(1, -1)).mean()


def compute_yearly_topic_scores(
    conference_data: pd.DataFrame,
    topic_vector: np.ndarray | Tensor,
    nlp: Language,
    model: SentenceTransformer,
) -> dict:
    yearly_topic_score = {}
    for year, group in conference_data.groupby("cyear"):
        yearly_topic_score[year] = (
            group["speech"]
            .apply(
                lambda x: compute_speech_topic_score(
                    embeddings.get_sent_embeds(embeddings.split_speech(x, nlp), model),
                    topic_vector,
                )
            )
            .mean()
        )
    return yearly_topic_score


def compute_sent_level_topic_score_dist(
    speech_embeddings: Tensor | np.ndarray, topic_vector: Tensor | np.ndarray
) -> list:
    sent_scores = []
    for sent_embedding in speech_embeddings:
        sent_topic_score = cosine_similarity(
            sent_embedding.reshape(1, -1), topic_vector.reshape(1, -1)
        )[0][0]
        sent_scores.append(sent_topic_score)
    return sent_scores


if __name__ == "__main__":
    # load core data and models
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    data_file = embeddings.ROOT_DIR / "data/processed/speeches.csv"
    data = pd.read_csv(data_file)
    nlp = spacy.load("en_core_web_sm")

    # religious axis embeddings
    religion_neg_df, religion_pos_df = (
        embeddings.ROOT_DIR / "data/anchors/religion_neg_phrases.csv",
        embeddings.ROOT_DIR / "data/anchors/religion_pos_phrases.csv",
    )
    religion_neg = embeddings.get_anchor_embeds(religion_neg_df, model)
    religion_pos = embeddings.get_anchor_embeds(religion_pos_df, model)
    religion_axis = embeddings.get_anchor_axis(religion_pos, religion_neg)

    # religion topic score by year plot
    yearly_religion_scores = compute_yearly_topic_scores(
        data, religion_axis, nlp, model
    )

    years, scores = (
        np.array(yearly_religion_scores.keys()),
        np.array(yearly_religion_scores.values()),
    )

    plt.plot(years, scores)

    plt.axhline(0, color="red", linestyle="--", linewidth=1, label="Neutral Threshold")

    m, b = np.polyfit(years, scores, 1)
    plt.plot(
        years,
        m * years + b,
        color="blue",
        linestyle="--",
        linewidth=1,
        label="Trend Line",
    )

    plt.xlabel("Year")
    plt.ylabel("Average Religion Topic Score")
    plt.title("Average Religion Topic Score by Year")
    plt.savefig("visual/yearly_religion_scores.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved yearly_religion_scores.png")

    # Plot 2: Histogram Word-Level Comparison of Religious vs Non-Religious Speech (Archbishop Dmitry Smirnov vs Craig A. Cardon)

    religious_score_dist = compute_sent_level_topic_score_dist(
        embeddings.get_speech_embeds(nlp, model, data, "Michael Gold "), religion_axis
    )
    rand_score_dist = compute_sent_level_topic_score_dist(
        embeddings.get_speech_embeds(nlp, model, data, "Akira Morita", 1997),
        religion_axis,
    )

    # Side-by-side histograms (easier to compare)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(
        religious_score_dist, bins=30, alpha=0.7, color="steelblue", edgecolor="black"
    )
    axes[0].set_title(
        "Religious Speech By Archbishop Dmitry Smirnov\nWord-Level Religion Scores"
    )
    axes[0].set_xlabel("Religion Topic Score")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(
        np.mean(religious_score_dist),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(religious_score_dist):.3f}",
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, 0.5)  # Set same x-axis limits for better comparison

    axes[1].hist(rand_score_dist, bins=30, alpha=0.7, color="coral", edgecolor="black")
    axes[1].set_title(
        "Non-Religious Speech By Craig A. Cardon\nWord-Level Religion Scores"
    )
    axes[1].set_xlabel("Religion Topic Score")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(
        np.mean(rand_score_dist),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(rand_score_dist):.3f}",
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.5, 0.5)  # Set same x-axis limits for better comparison

    plt.tight_layout()
    plt.savefig("visual/histogram_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved histogram_comparison.png")
