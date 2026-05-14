# This module contains functions for computing topic scores for conference speeches, generating visualizations of topic score trends over time,
# and comparing the distribution of sentence-level topic scores between different speeches.

# Natural Language Processing and Text Analysis for Conference Speeches
from sentence_transformers import (
    SentenceTransformer,
)  # For converting sentences into numerical embeddings
from spacy.language import (
    Language,
)  # For natural language processing tasks such as sentence tokenization

# Mathematical and data manipulation libraries
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  # For numerical operations
from torch import Tensor  # For handling numerical data in tensor format

# Data handling
import pandas as pd  # for working with dataframes

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.axes as axes

# Custom module for handling embeddings
from python import embeddings
from scipy import stats
import seaborn as sns  # For advanced data visualization, used for creating violin plots and enhancing the aesthetics of plots


def compute_speech_topic_score(
    sentence_embeddings: np.ndarray | Tensor,
    topic_vector: np.ndarray | Tensor,
    q: float = 0.75,
) -> float:
    """
    Computes the average cosine similarity between sentence embeddings and a topic vector.

    Args:
        sentence_embeddings (np.ndarray | Tensor): The numerical representations of the sentences in a speech, typically of shape (num_sentences, embedding_dim).
        topic_vector (np.ndarray | Tensor): A numerical representation of the topic, typically of shape (embedding_dim,) creagted by averaging the embeddings of keywords related to the topic (created from init_speech_embeds).
        q (float): The quantile to use for computing the topic score, with a default value of 0.75.
        - This parameter allows for a more robust topic score by considering the distribution of sentence-level scores and focuses on the upper quantile.
    Returns:
        float: The average topic score for the speech, computed as the specified quantile of the cosine similarity scores between the sentence embeddings and the topic vector.
    """
    # NOTE: Old implementation
    # return cosine_similarity(sentence_embeddings, topic_vector.reshape(1, -1)).mean()

    sent_scores = cosine_similarity(
        sentence_embeddings, topic_vector.reshape(1, -1)
    ).flatten()
    return float(np.quantile(sent_scores, q))


def compute_speech_topic_proportion(
    sentence_embeddings: np.ndarray | Tensor,
    topic_vector: np.ndarray | Tensor,
    threshold: float = 0.0,
) -> float:
    """
    Computes the proportion of sentences in a speech that are above a topic threshold.

    For religion:
    - score > 0 means the sentence is more religious than neutral
    - score <= 0 means the sentence is neutral/non-religious based on this axis

    Args:
        sentence_embeddings: Sentence embeddings for one speech.
        topic_vector: Topic axis vector.
        threshold: Cutoff for counting a sentence as topic-positive. Default is 0.

    Returns:
        float: Proportion of sentences above the threshold.
    """

    sent_scores = cosine_similarity(
        sentence_embeddings, topic_vector.reshape(1, -1)
    ).flatten()

    return float(np.mean(sent_scores > threshold))


def compute_yearly_topic_scores(
    conference_data: pd.DataFrame,
    topic_vector: np.ndarray | Tensor,
    nlp: Language,
    model: SentenceTransformer,
    q: float = 0.75,
) -> tuple[dict, dict]:
    """
    Computes the average topic score for each year in the conference data.

    This function analyzes the speeches given in each year of the conference by computing the average cosine similarity between
    the sentence embeddings of the speeches and a given topic vector. The result is a dictionary of the format
    {year: average_topic_score}, where the average topic score is computed as the mean of the cosine similarities for all speeches given in that year.

    Args:
        conference_data (pd.DataFrame): A DataFrame containing the conference speeches, with columns for 'cyear' (conference year) and 'speech' (the text of the speech).
        topic_vector (np.ndarray | Tensor): A numerical representation of the topic, typically of shape (embedding_dim,) created by averaging the embeddings of keywords related to the topic (created from init_speech_embeds).
        nlp (Language): A spaCy language model used for sentence tokenization.
        model (SentenceTransformer): A sentence transformer model used to generate embeddings for the sentences in the speeches.
        q (float): The quantile to use for computing the topic score, with a default value of 0.75.
    Returns:
    #NOTE: UPDATE THIS
        tuple[dict, dict]: A tuple containing two dictionaries:
            - The first dictionary maps each year to a list of topic scores for the speeches given in that year, with the format {year: [topic_score1, topic_score2, ...]}.
            - The second dictionary maps each year to the average topic score for that year, with the format {year: average_topic_score}, where the average topic score is computed as the mean of the topic scores for all speeches given in that year.
    """

    yearly_topic_avg_score = {}
    yearly_topic_scores = {}
    for year, group in conference_data.groupby("cyear"):
        topic_scores = (
            group["speech"]
            .apply(
                lambda x: compute_speech_topic_score(
                    embeddings.get_sent_embeds(embeddings.split_speech(x, nlp), model),
                    topic_vector,
                    q=q,
                )
            )
            .to_list()
        )
        yearly_topic_scores[year] = topic_scores
        yearly_topic_avg_score[year] = np.mean(topic_scores)
    return yearly_topic_scores, yearly_topic_avg_score


def compute_yearly_topic_proportions(
    conference_data: pd.DataFrame,
    topic_vector: np.ndarray | Tensor,
    nlp: Language,
    model: SentenceTransformer,
    threshold: float = 0.0,
) -> tuple[dict, dict]:
    """
    Computes the proportion of positive topic sentences for each speech,
    then groups those proportions by year.

    Returns:
        yearly_topic_proportions:
            {year: [speech_prop1, speech_prop2, ...]}

        yearly_avg_proportion:
            {year: average proportion across speeches in that year}
    """

    yearly_topic_proportions = {}
    yearly_avg_proportion = {}

    for year, group in conference_data.groupby("cyear"):
        print(f"computing positive proportion for year {year}...", flush=True)

        topic_proportions = (
            group["speech"]
            .apply(
                lambda x: compute_speech_topic_proportion(
                    embeddings.get_sent_embeds(embeddings.split_speech(x, nlp), model),
                    topic_vector,
                    threshold=threshold,
                )
            )
            .to_list()
        )

        yearly_topic_proportions[year] = topic_proportions
        yearly_avg_proportion[year] = np.mean(topic_proportions)

    return yearly_topic_proportions, yearly_avg_proportion


def compute_sent_level_topic_score_dist(
    speech_embeddings: Tensor | np.ndarray, topic_vector: Tensor | np.ndarray
) -> list:
    """
    Computes the distribution of sentence level topic scores for a given speech.

    This function calculates the cosine similarity between each sentence embedding in a speech and a given topic vector, outputting a list of topic scores for one speech.

    Args:
        speech_embeddings (Tensor | np.ndarray): The numerical representations of the sentences in a speech, typically of shape (num_sentences, embedding_dim).
        topic_vector (Tensor | np.ndarray): A numerical representation of the topic, typically of shape (embedding_dim,) created by averaging the embeddings of keywords related to the topic (created from init_speech_embeds).
    Returns:
        list: A list of cosine similarity scores for each sentence in the speech with respect to the topic.
            - Each score represents how closely a sentence aligns with the topic, with higher scores indicating
    """

    sent_scores = []
    for sent_embedding in speech_embeddings:
        sent_topic_score = cosine_similarity(
            sent_embedding.reshape(1, -1), topic_vector.reshape(1, -1)
        )[0][0]
        sent_scores.append(sent_topic_score)
    return sent_scores


def save_topic_score_by_year_plot(topic: str, yearly_scores: dict) -> None:
    """
    Saves a line plot of average topic scores by year for a given topic.

    This function creates and saves a fitted line plot that visualizes the average topic score for each year,
    with a horizontal line indicating the neutral threshold, 0, and a trend lend to show overall direction.

    Args:
        topic (str): The name of the topic being plotted, used for labeling the axes
        yearly_scores (dict): A dictionary mapping years to average topic scores.
    Returns:
        None: The function saves the plot to a file and does not return any value.
    """

    # Setting up the plot
    years = np.array(list(yearly_scores.keys()))
    scores = np.array(list(yearly_scores.values()))

    _, ax = plt.subplots()

    # Plotting the average topic scores by year
    ax.plot(years, scores)

    # Adding a horizontal line at y=0 to indicate the neutral threshold and a trend line to show overall direction
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

    # Finalizing the plot
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
    """
    Creates a boxplot of topic scores by year for a given topic.

    This function creates a boxplot that visualizes the distribution of topic scores for each year, with an optional trend line to show overall direction.

    Args:
        topic (str): The name of the topic being plotted, used for labeling the axes.
        yearly_scores (dict): A dictionary mapping years to lists of topic scores for that year.
        show_trend (bool): Whether to include a trend line in the plot. Default is True.
        trend_method (str): The method to use for calculating the trend line, either "median" or "mean". Default is "median".
    Returns:
        None: The function saves the plot to a file and does not return any value.
    """
    # Setting up the plot
    years = np.array(list(yearly_scores.keys()))
    scores = [yearly_scores[year] for year in years]
    _, ax = plt.subplots()

    # Creating the boxplot
    ax.boxplot(scores, positions=years, widths=0.7)

    if show_trend:
        if trend_method == "median":
            trend_values = [np.median(yearly_scores[year]) for year in years]
        elif trend_method == "mean":
            trend_values = [np.mean(yearly_scores[year]) for year in years]
        else:
            pass  # TODO: more trend methods?
        if trend_method in ["median", "mean"]:
            m, b = np.polyfit(years, trend_values, 1)
            s, i, r, p, e = stats.linregress(years, trend_values)
            ax.plot(
                years,
                m * years + b,
                color="blue",
                linestyle="--",
                linewidth=1,
                label=f"{trend_method.capitalize()} Trend (Slope:{s:.4f}, p-value:{p:.4f}, se:{e:.4f})",
            )

    # Finalizing the plot
    ax.set_xlabel("Year")
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.set_ylabel(f"{topic} Topic Scores")
    ax.set_title(f"{topic} Topic Scores by Year")
    ax.legend()
    # plt.savefig(f"outputs/plots/boxplot_yearly_{topic}_scores.png", dpi=300, bbox_inches="tight") # for og data
    plt.savefig(
        f"outputs/plots/boxplot_yearly_{topic}_scores_quantile.png",
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
    embeds: Tensor | np.ndarray,
    color: str,
) -> None:
    """
    Creates a histogram plot of sentence-level topic scores for a given speech.

    This function creates a histogram plot visualizing the distribution of sentence-level topic scores for a given speech,
    with a vertical line indicating the mean topic score for the speech.

    Args:
        topic (str): The name of the topic being plotted.
        title (str): The title of the plot.
        speaker (str): The name of the speaker whose speech is being plotted.
        ax (axes.Axes): The matplotlib axes object on which to create the histogram.
        axis (Tensor | np.ndarray): The topic vector used to compute the sentence-level topic scores, typically of shape (embedding_dim,).
        embeds (Tensor | np.ndarray): The numerical representations of the sentences in the speech, typically of shape (num_sentences, embedding_dim).
        color (str): The color to use for the histogram bars, specified as a string (e.g., "cornflowerblue", "coral").
    Returns:
        None: The function creates the histogram on the provided axes and does not return any value.
    """
    # Computing sentence-level topic score for the speech
    score_dist = compute_sent_level_topic_score_dist(embeds, axis)
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
    """
    Creates a violin plot showing the distribution of all speech scores for a specific year.
    """
    # 1. Extract the scores for the specific year from your cached data
    if target_year not in yearly_scores:
        print(f"Year {target_year} not found in yearly_scores.")
        return

    year_data = yearly_scores[target_year]  #

    # 2. Setup the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3. Create the violin plot
    # This visualizes the density and range of all speech scores in that year
    sns.violinplot(x=year_data, ax=ax, color=color, inner="quartile")

    # 4. Add the 75th quantile marker of the yearly distribution
    q75 = np.quantile(year_data, 0.75)  #
    ax.axvline(
        float(q75),
        color="red",
        linestyle="--",
        label=f"Yearly 75th Quantile: {q75:.3f}",
    )

    # 5. Finalizing aesthetics
    ax.set_title(f"Distribution of {topic} Speech Scores in {target_year}")
    ax.set_xlabel(f"{topic} Topic Score (Quantile Method)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(
        f"outputs/plots/{topic}_{target_year}_yearly_violin.png",
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
    topic_embeds: Tensor | np.ndarray,
    neutral_embeds: Tensor | np.ndarray,
    topic_color: str,
    neutral_color: str,
) -> None:
    """
    Saves a histogram comparison plot of sentence-level topic scores for a given topic and neutral speeches.

    Args:
        topic (str): The name of the topic being plotted.
        neutral (str): The label for the neutral speech.
        topic_spkr (str): The name of the speaker of the topic speech.
        neutral_spkr (str): The name of the speaker of the neutral speech.
        axis (Tensor | np.ndarray): The topic vector used to compute the sentence-level topic scores, typically of shape (embedding_dim,).
        topic_embeds (Tensor | np.ndarray): The numerical representations of the sentences in the topic speech, typically of shape (num_sentences, embedding_dim).
        neutral_embeds (Tensor | np.ndarray): The numerical representations of the sentences in the neutral speech, typically of shape (num_sentences, embedding_dim).
        topic_color (str): The color to use for the histogram bars of the topic speech, specified as a string (e.g., "cornflowerblue").
        neutral_color (str): The color to use for the histogram bars of the neutral speech, specified as a string (e.g., "coral").
    Returns:
        None: The function creates the histogram comparison plot and saves it to a file, without returning any value.
    """

    # Setting up the plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Creating the histogram plots
    conf_hist_plot(topic, topic, topic_spkr, ax1, axis, topic_embeds, topic_color)
    conf_hist_plot(
        topic, neutral, neutral_spkr, ax2, axis, neutral_embeds, neutral_color
    )
    plt.tight_layout()
    plt.savefig(
        f"outputs/plots/{topic}_hist_comparison.png", dpi=300, bbox_inches="tight"
    )

    plt.close()


def the_goat_tyler(
    topic: str,
    yearly_scores: dict,
) -> None:
    """
    who else but the goat?
    """
    # Setting up the plot
    years = np.array(list(yearly_scores.keys()))
    scores = [yearly_scores[year] for year in years]
    _, ax = plt.subplots()

    # Creating the boxplot
    ax.violinplot(scores, positions=years, showmedians=True)

    ax.set_xlabel("Year")
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.set_ylabel(f"{topic} Topic Scores")
    ax.set_title(f"{topic} Topic Scores by Year")
    ax.legend()
    # plt.savefig(f"outputs/plots/boxplot_yearly_{topic}_scores.png", dpi=300, bbox_inches="tight") # for og data
    plt.savefig(
        f"outputs/plots/violinplot_yearly_{topic}_scores_quantile.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
