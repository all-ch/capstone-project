from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from spacy.language import Language
from python import embeddings
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.axes as axes
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
    plt.savefig(
        f"outputs/plots/yearly_{topic}_scores.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def conf_hist_plot(
    topic: str,
    title: str,
    speaker: str,
    ax: axes.Axes,
    axis: Tensor | np.ndarray,
    embeds: Tensor | np.ndarray,
    color: str,
) -> None:
    score_dist = compute_sent_level_topic_score_dist(embeds, axis)
    ax.hist(score_dist, bins=30, alpha=0.7, color=color, edgecolor="black")
    ax.set_title(f"{title} Speech By {speaker}\nSentence-Level {topic} Scores")
    ax.set_xlabel(f"{topic} Topic Score")
    ax.set_ylabel("Frequency")
    ax.axvline(
        float(np.mean(score_dist)),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(score_dist):.3f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)


def save_hist_comparison_plot(
    topic: str,
    neutral: str,
    topic_spkr: str,
    neutral_spkr: str,
    axis: Tensor | np.ndarray,
    topic_embeds: Tensor | np.ndarray,
    neutral_embeds: Tensor | np.ndarray,
    topic_color: str,
    neutral_color: str,
):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    conf_hist_plot(topic, topic, topic_spkr, ax1, axis, topic_embeds, topic_color)
    conf_hist_plot(
        topic, neutral, neutral_spkr, ax2, axis, neutral_embeds, neutral_color
    )
    plt.tight_layout()
    plt.savefig(
        f"outputs/plots/{topic}_hist_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
