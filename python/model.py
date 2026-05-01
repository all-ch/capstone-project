from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from spacy.language import Language
from python import embeddings
from torch import Tensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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


def save_topic_score_by_year_plot(topic: str, yearly_scores: dict) -> None:
    years = np.array(list(yearly_scores.keys()))
    scores = np.array(list(yearly_scores.values()))
    _, ax = plt.subplots()
    ax.plot(years, scores)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Neutral Threshold")
    m, b = np.polyfit(years, scores, 1)
    ax.plot(
        years,
        m * years + b,
        color="blue",
        linestyle="--",
        linewidth=1,
        label="Trend Line",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Average {topic} Topic Score")
    ax.set_title(f"Average {topic} Topic Score by Year")
    plt.savefig(f"visual/yearly_{topic}_scores.png", dpi=300, bbox_inches="tight")
    plt.close()


def conf_hist_plot(topic: str, title: str, axis: Tensor | np.ndarray):
    pass


def save_hist_comparison_plot():
    pass


if __name__ == "__main__":
    # load core data and models
    model, data, nlp = embeddings.init_models(
        "sentence-transformers/all-mpnet-base-v2",
        "data/processed/speeches.csv",
        "en_core_web_sm",
    )

    # religious axis embeddings
    religion_pos, religion_neg, religion_axis = embeddings.init_vec(
        "data/anchors/religion_pos_phrases.csv",
        "data/anchors/religion_neg_phrases.csv",
        model,
    )

    # religion topic score by year plot
    yearly_religion_scores = compute_yearly_topic_scores(
        data, religion_axis, nlp, model
    )

    save_topic_score_by_year_plot("Religion", yearly_religion_scores)

    # histogram religious vs random sentence level scores comparison plot
    religious_score_dist = compute_sent_level_topic_score_dist(
        embeddings.get_speech_embeds(nlp, model, data, "Michael Gold "), religion_axis
    )
    rand_score_dist = compute_sent_level_topic_score_dist(
        embeddings.get_speech_embeds(nlp, model, data, "Akira Morita", 1997),
        religion_axis,
    )

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # todo: could write into a function to remove redundancy
    ax1.hist(
        religious_score_dist, bins=30, alpha=0.7, color="steelblue", edgecolor="black"
    )
    ax1.set_title("Religious Speech By Michael Gold\nSentence-Level Religion Scores")
    ax1.set_xlabel("Religion Topic Score")
    ax1.set_ylabel("Frequency")
    ax1.axvline(
        np.mean(religious_score_dist),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(religious_score_dist):.3f}",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 0.5)

    ax2.hist(rand_score_dist, bins=30, alpha=0.7, color="coral", edgecolor="black")
    ax2.set_title("Random Speech by Akira Morita\nSentence-Level Religion Scores")
    ax2.set_xlabel("Religion Topic Score")
    ax2.set_ylabel("Frequency")
    ax2.axvline(
        np.mean(rand_score_dist),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(rand_score_dist):.3f}",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig("visual/histogram_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
