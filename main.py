from sklearn.preprocessing import StandardScaler
from python import embeddings
from python import model as tm
from python import pca

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLP_MODEL = "en_core_web_sm"

DATA_DIR = "data/processed/speeches.csv"

RELIGION_POS_DIR = "data/anchors/religion_pos_phrases.csv"
RELIGION_NEG_DIR = "data/anchors/religion_neg_phrases.csv"

POLITICS_POS_DIR = ""
POLITICS_NEG_DIR = ""

SCIENCE_POS_DIR = ""
SCIENCE_NEG_DIR = ""

RELIGION_SPKR, RELIGION_YEAR = "Michael Gold ", None
POLITICS_SPKR, POLITICS_YEAR = "", None
SCIENCE_SPKR, SCIENCE_YEAR = "", None
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
        DATA_DIR,
        NLP_MODEL,
    )
    scalar = StandardScaler()

    for topic in ["Religion"]:  # todo: add in politics and science
        print(f"loading {topic} example speech embeddings...")
        topic_embeds, neutral_embeds = embeddings.init_speech_embeds(
            nlp,
            model,
            data,
            TOPICS[topic]["Speaker"],
            TOPICS["Neutral"]["Speaker"],
            TOPICS[topic]["Year"],
            TOPICS["Neutral"]["Year"],
        )

        print(f"creating {topic} vectors...")
        pos_vec, neg_vec, topic_axis = embeddings.init_vec(
            TOPICS[topic]["Positive"], TOPICS[topic]["Negative"], model
        )

        print(f"creating {topic} pca plot...")
        pca.save_pca_plot(
            topic, scalar, 2, pos_vec, neg_vec, topic_axis, topic_embeds, neutral_embeds
        )

        print(f"saved {topic} pca plot!")

        print(f"computing {topic} topic scores by year...")
        topic_scores_by_years = tm.compute_yearly_topic_scores(
            data, topic_axis, nlp, model
        )

        print(f"creating {topic} topic scores by year plot...")
        tm.save_topic_score_by_year_plot(topic, topic_scores_by_years)

        print(f"saved {topic} topic scores by year plot!")

        print(f"creating {topic} histogram comparison plot...")
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

        print(f"saved {topic} histogram comparison plots!")
    print("script finished.")


if __name__ == "__main__":
    main()
