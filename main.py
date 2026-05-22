# This script serves as the main entry point for the project. It orchestrates the loading of models and data, the computation of topic scores, and the generation of visualizations for conference speeches on various topics.
#
# HOW TO RUN THIS SCRIPT IN VS CODE:
# -----------------------
#
# 1. Open VS Code, go to File > Open Folder... , and select this projects' project folder.
# 2. Open VSCode's integrated terminal by clicking Terminal > New Terminal, in the top menu bar.
# 3. Install required packages by installing 'uv' (Python package installer) by pasting this into terminal:
# -  pip install uv
# -  Then, download and link all required packages by pasting:
# - uv sync
# 4. Execute this script by inputting:
# - python main.py

# Import necessary libraries and modules
from sklearn.preprocessing import (
    StandardScaler,
)  # For standardizing features by removing the mean and scaling to unit variance
from python import embeddings  # Custom module for handling sentence embeddings
from python import (
    model as tm,
)  # Custom module for computing topic scores and generating visualizations
from python import (
    pca,
)  # Custom module for performing Principal Component Analysis and generating PCA plots

import pickle  # for saving and loading computed scores
import os  # for checking if cache files exist

# Sentence embedding and natural language processing models
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"  # https://huggingface.co/sentence-transformers/all-mpnet-base-v2, a pre-trained model for generating sentence embeddings
NLP_MODEL = "en_core_web_sm"  # English model for natural language processing tasks such as sentence tokenization

# Conference speech data
DATA_DIR = "data/processed/speeches.csv"
NEW_DATA_DIR = "data/processed/new_speeches.csv"

# Anchor sentences for each topic, used to define the topic axes in the embedding space
# NOTE 1: Sentences were generated using ChatGPT, so they may not be perfect representations, and should be checked for accuracy and relevance to the topic.
#  AI generated sentences should be further reviewed and validated.
# NOTE 2: If wanting to add more anchors, create a new CSV file with the same format as existing anchor files (no header, sentences in the first column), then
# add the file path to the appropriate topic in the TOPICS dictionary below, and ensure that the file is located in the correct directory.
RELIGION_POS_DIR = "data/anchors/religion_pos_sentences.csv"
RELIGION_NEG_DIR = "data/anchors/religion_neg_sentences.csv"

POLITICS_POS_DIR = "data/anchors/politics_pos_sentences.csv"
POLITICS_NEG_DIR = "data/anchors/politics_neg_sentences.csv"

SCIENCE_POS_DIR = "data/anchors/science_pos_sentences.csv"
SCIENCE_NEG_DIR = "data/anchors/science_neg_sentences.csv"

# NOTE: These anchor sentences are statistically significant, but are measuring poor versus rigorous political/scientific justtification, assumes each sentence can be measured on a spectrum of justification quality, 0 does not mean absence of political or scientific content.
# POLITICS_POS_DIR = "data/anchors/political_justification_pos_sentences.csv"
# POLITICS_NEG_DIR = "data/anchors/political_justification_neg_sentences.csv"
# SCIENCE_POS_DIR = "data/anchors/scientific_justification_pos_sentences.csv"
# SCIENCE_NEG_DIR = "data/anchors/scientific_justification_neg_sentences.csv"

# Preset speakers and years for data analysis and visualization
RELIGION_SPKR, RELIGION_YEAR = "Michael Gold ", 1999
POLITICS_SPKR, POLITICS_YEAR = "David A. Hartman", 2004
SCIENCE_SPKR, SCIENCE_YEAR = "Francisco J. González Estepa", 2012
NEUTRAL_SPKR, NEUTRAL_YEAR = "Akira Morita", 1997

# Dictionary to store topic information for easy access and organization
# Format is as follows:
# "Topic Name": {
#     "Positive": Path to CSV file containing positive anchor sentences for the topic,
#     "Negative": Path to CSV file containing negative anchor sentences for the topic,
#     "Speaker": Name of the speaker whose speech will be analyzed for the topic,
#     "Year": Year of the speech to be analyzed for the topic,
# } for each topic. The "Neutral" topic only contains a speaker and year, as it is used for comparison against the other topics rather than defining a topic axis.
TOPICS = {
    "Religion": {
        "Positive": RELIGION_POS_DIR,
        "Negative": RELIGION_NEG_DIR,
        "Speaker": RELIGION_SPKR,
        "Year": RELIGION_YEAR,
    },
    "Politics": {
        "Positive": POLITICS_POS_DIR,
        "Negative": POLITICS_NEG_DIR,
        "Speaker": POLITICS_SPKR,
        "Year": POLITICS_YEAR,
    },
    "Science": {
        "Positive": SCIENCE_POS_DIR,
        "Negative": SCIENCE_NEG_DIR,
        "Speaker": SCIENCE_SPKR,
        "Year": SCIENCE_YEAR,
    },
    "Neutral": {
        "Speaker": NEUTRAL_SPKR,
        "Year": NEUTRAL_YEAR,
    },
}


def main():

    print("loading all models and data...")
    model, data, nlp = embeddings.init_models(
        EMBEDDINGS_MODEL,
        NEW_DATA_DIR,
        NLP_MODEL,
    )
    scalar = StandardScaler()

    for topic in TOPICS:
        if topic == "Neutral":
            continue

        print(f"creating {topic} vectors...")
        pos_vec, neg_vec, topic_axis = embeddings.init_vec(
            TOPICS[topic]["Positive"], TOPICS[topic]["Negative"], model
        )

        cache_path = f"data/processed/{topic}_scores.pkl"
        update_scores = False

        if os.path.exists(cache_path):
            update_scores = input(
                f"{topic} Cache found. Recompute scores? (Y/N): "
            ).lower()
            update_scores = update_scores == "y"
        else:
            print(f"No {topic} cache found. Computing scores..")
            update_scores = True

        if update_scores:
            yearly_topic_scores, yearly_avg_score = tm.compute_yearly_topic_scores(
                data, topic_axis, nlp, model, q=0.75
            )

            with open(cache_path, "wb") as f:
                pickle.dump((yearly_topic_scores, yearly_avg_score), f)

            print(f"Scores saved to {cache_path}")

        else:
            print(f"Loading {topic} scores")
            with open(cache_path, "rb") as f:
                yearly_topic_scores, yearly_avg_score = pickle.load(f)

        print(f"Loading {topic} example speech embeddings...")
        topic_embeds, neutral_embeds = embeddings.init_speech_embeds(
            nlp,
            model,
            data,
            TOPICS[topic]["Speaker"],
            TOPICS["Neutral"]["Speaker"],
            TOPICS[topic]["Year"],
            TOPICS["Neutral"]["Year"],
        )

        pca.save_pca_plot(
            topic,
            scalar,
            2,
            pos_vec,
            neg_vec,
            (pos_vec + neg_vec) / 2,
            topic_embeds,
            neutral_embeds,
        )
        print(f"Created {topic} pca plot.")

        tm.conf_boxplot(
            topic, yearly_topic_scores, show_trend=True, trend_method="mean"
        )
        print(f"Created {topic} topic scores by year boxplot.")

        tm.conf_violinplot(
            topic, yearly_topic_scores, show_trend=True, trend_method="mean"
        )
        print(f"Created {topic} topic scores by year violinplot!")

        tm.save_hist_comparison_plot(
            topic,
            "Neutral",
            TOPICS[topic]["Speaker"],
            TOPICS["Neutral"]["Speaker"],
            topic_axis,
            topic_embeds,
            neutral_embeds,
            "cornflowerblue",
            "coral",
        )
        print(f"Created {topic} histogram comparison plot.")
    print("script finished.")


if __name__ == "__main__":
    main()
