from sklearn.preprocessing import StandardScaler
from python import embeddings
import pickle
import os

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLP_MODEL = "en_core_web_sm"

DATA_DIR = "data/processed/speeches.csv"
NEW_DATA_DIR = "data/processed/new_speeches.csv"

RELIGION_POS_DIR = "data/anchors/religion_pos_sentences.csv"
RELIGION_NEG_DIR = "data/anchors/religion_neg_sentences.csv"

POLITICS_POS_DIR = "data/anchors/politics_pos_sentences.csv"
POLITICS_NEG_DIR = "data/anchors/politics_neg_sentences.csv"

SCIENCE_POS_DIR = "data/anchors/science_pos_sentences.csv"
SCIENCE_NEG_DIR = "data/anchors/science_neg_sentences.csv"

RELIGION_SPKR, RELIGION_YEAR = "Michael Gold ", 1999
POLITICS_SPKR, POLITICS_YEAR = "David A. Hartman", 2004
SCIENCE_SPKR, SCIENCE_YEAR = "Francisco J. González Estepa", 2012
NEUTRAL_SPKR, NEUTRAL_YEAR = "Akira Morita", 1997


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
        pos_vec, neg_vec, topic_axis, topic_vec = embeddings.init_vec(
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
                data, topic_axis, topic_vec, nlp, model, q=0.75
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


if __name__ == "__main__":
    main()
