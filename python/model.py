from sentence_transformers import SentenceTransformer
from spacy.language import Language
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch import Tensor, threshold
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from python import embeddings
from scipy import stats
import seaborn as sns


def compute_speech_topic_score(
    sentence_embeddings: np.ndarray | Tensor,
    topic_axis: np.ndarray | Tensor,
    topic_vector: np.ndarray | Tensor,
    q: float = 0.75,
) -> np.ndarray | Tensor:
    sent_scores = cosine_similarity(
        sentence_embeddings - topic_vector, topic_axis.reshape(1, -1)
    ).flatten()
    cutoff = np.quantile(sent_scores, q)
    return sent_scores[sent_scores >= cutoff]


def compute_yearly_topic_scores(
    conference_data: pd.DataFrame,
    topic_axis: np.ndarray | Tensor,
    topic_vector: np.ndarray | Tensor,
    nlp: Language,
    model: SentenceTransformer,
    q: float = 0.75,
) -> tuple[dict, dict]:
    yearly_topic_avg_score = {}
    yearly_topic_scores = {}
    for year, group in conference_data.groupby("cyear"):
        topic_scores = group["speech"].apply(
            lambda x: compute_speech_topic_score(
                embeddings.get_sent_embeds(embeddings.split_speech(x, nlp), model),
                topic_axis,
                topic_vector,
                q=q,
            )
        )
        yearly_topic_scores[year] = topic_scores
        yearly_topic_avg_score[year] = np.mean(topic_scores)
    return yearly_topic_scores, yearly_topic_avg_score


def compute_sent_level_topic_score_dist(
    speech_embeddings: Tensor | np.ndarray,
    topic_axis: Tensor | np.ndarray,
    topic_vector: Tensor | np.ndarray,
) -> list:
    sent_scores = []
    for sent_embedding in speech_embeddings:
        sent_topic_score = cosine_similarity(
            (sent_embedding - topic_vector).reshape(1, -1), topic_axis.reshape(1, -1)
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
    # plt.savefig(f"outputs/plots/yearly_{topic}_scores.png", dpi=300, bbox_inches="tight") # for og data
    plt.savefig(
        f"outputs/plots/yearly_{topic}_scores.png", dpi=300, bbox_inches="tight"
    )

    plt.close()


def conf_boxplot(
    topic: str,
    yearly_scores: dict,
    show_trend: bool = True,
    trend_method: str = "median",
) -> None:
    years = np.array(list(yearly_scores.keys()))
    scores = [yearly_scores[year] for year in years]
    _, ax = plt.subplots()

    ax.boxplot(scores, positions=years, widths=0.7)

    if show_trend:
        if trend_method == "median":
            trend_values = [np.median(yearly_scores[year]) for year in years]
        elif trend_method == "mean":
            trend_values = [np.mean(yearly_scores[year]) for year in years]
        else:
            pass
        if trend_method in ["median", "mean"]:
            m, b = np.polyfit(years, trend_values, 1)

            residual_scatter_plot(topic, yearly_scores, m, b)
            s, i, r, p, e = stats.linregress(years, trend_values)
            ax.plot(
                years,
                m * years + b,
                color="blue",
                linestyle="--",
                linewidth=1,
                label=f"{trend_method.capitalize()} Trend (Slope:{s:.4f}, p-value:{p:.4f}, se:{e:.4f})",
            )

    ax.set_xlabel("Year")
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.set_ylabel(f"{topic} Topic Scores")
    ax.set_ylim(-0.10, 0.40)
    ax.set_title(f"{topic} Topic Scores by Year")
    ax.legend()
    # plt.savefig(f"outputs/plots/boxplot_yearly_{topic}_scores.png", dpi=300, bbox_inches="tight") # for og data
    plt.savefig(
        f"outputs/plots/Boxplots/{topic}_yearly_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def conf_hist_plot(
    topic: str,
    title: str,
    speaker: str,
    ax: axes.Axes,
    axis: Tensor | np.ndarray,
    vec: Tensor | np.ndarray,
    embeds: Tensor | np.ndarray,
    color: str,
) -> None:
    score_dist = compute_sent_level_topic_score_dist(embeds, axis, vec)
    ax.hist(score_dist, bins=30, alpha=0.7, color=color, edgecolor="black")
    ax.set_title(f"{title} Speech By {speaker}\nSentence-Level {topic} Scores")
    ax.set_xlabel(f"{topic} Topic Score")
    ax.set_ylabel("Frequency")
    ax.axvline(
        float(np.quantile(score_dist, 0.75)),
        color="red",
        linestyle="--",
        label=f"75th Quantile: {np.quantile(score_dist, 0.75):.3f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)


def conf_violin_plot_yearly(
    topic: str, yearly_scores: dict, target_year: int, color: str
) -> None:
    if target_year not in yearly_scores:
        print(f"Year {target_year} not found in yearly_scores.")
        return

    year_data = yearly_scores[target_year]

    _, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(x=year_data, ax=ax, color=color, inner="quartile")

    q75 = np.quantile(year_data, 0.75)  #
    ax.axvline(
        float(q75),
        color="red",
        linestyle="--",
        label=f"Yearly 75th Quantile: {q75:.3f}",
    )

    ax.set_title(f"Distribution of {topic} Speech Scores in {target_year}")
    ax.set_xlabel(f"{topic} Topic Score (Quantile Method)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(
        f"outputs/plots/SingleViolinplots/{topic}_{target_year}_yearly_violin.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_hist_comparison_plot(
    topic: str,
    neutral: str,
    topic_spkr: str,
    neutral_spkr: str,
    axis: Tensor | np.ndarray,
    vec: Tensor | np.ndarray,
    topic_embeds: Tensor | np.ndarray,
    neutral_embeds: Tensor | np.ndarray,
    topic_color: str,
    neutral_color: str,
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    conf_hist_plot(topic, topic, topic_spkr, ax1, axis, vec, topic_embeds, topic_color)
    conf_hist_plot(
        topic, neutral, neutral_spkr, ax2, axis, vec, neutral_embeds, neutral_color
    )
    plt.tight_layout()
    plt.savefig(
        f"outputs/plots/Histograms/{topic}_hist_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def conf_violinplot(
    topic: str,
    yearly_scores: dict,
    show_trend: bool = True,
    trend_method: str = "median",
) -> None:
    years = np.array(list(yearly_scores.keys()))
    scores = [yearly_scores[year] for year in years]
    _, ax = plt.subplots()

    ax.violinplot(
        scores,
        positions=years,
        widths=1,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    if show_trend:
        if trend_method == "median":
            trend_values = [np.median(yearly_scores[year]) for year in years]
        elif trend_method == "mean":
            trend_values = [np.mean(yearly_scores[year]) for year in years]
        else:
            pass

        if trend_method in ["median", "mean"]:
            m, b = np.polyfit(years, trend_values, 1)

            residual_scatter_plot(topic, yearly_scores, m, b)

            s, i, r, p, e = stats.linregress(years, trend_values)

            ax.plot(
                years,
                m * years + b,
                color="blue",
                linestyle="--",
                linewidth=1,
                label=f"{trend_method.capitalize()} Trend",
            )

    ax.set_xlabel("Year", fontsize=20)
    ax.set_ylabel(f"{topic} Topic Scores", fontsize=20)
    ax.set_title(f"{topic} Topic Scores by Year", fontsize=20)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)

    plt.savefig(
        f"outputs/plots/Violinplots/{topic}_yearly_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def the_goat_tyler(
    topic: str,
    yearly_scores: dict,
) -> None:
    years = np.array(list(yearly_scores.keys()))
    scores = [yearly_scores[year] for year in years]
    _, ax = plt.subplots()

    ax.violinplot(scores, positions=years, showmedians=True)
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.set_ylabel(f"{topic} Topic Scores")
    ax.set_title(f"{topic} Topic Scores by Year")
    plt.savefig(
        f"outputs/plots/Violinplots/{topic}yearly_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def residual_scatter_plot(
    topic: str,
    yearly_scores: dict,
    m: float,
    b: float,
) -> None:
    years = np.array(list(yearly_scores.keys()))
    all_years = []
    residuals = []
    _, ax = plt.subplots(figsize=(8, 6))

    for year in years:
        predicted_value = m * year + b
        actual_values = np.array(yearly_scores[year])

        yearly_residuals = actual_values - predicted_value

        all_years.extend([year] * len(yearly_residuals))
        residuals.extend(yearly_residuals)

    sns.regplot(x=all_years, y=residuals, lowess=True, line_kws=dict(color="r"), ax=ax)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(f"{topic} Residuals Over Time")

    plt.savefig(
        f"outputs/plots/Residuals/{topic}_yearly_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
