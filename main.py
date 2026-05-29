from python import embeddings
import pickle
import os

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLP_MODEL = "en_core_web_sm"

DATA_DIR = "data/processed/new_speeches.csv"

RELIGION_POS_DIR = "data/anchors/religion_pos_sentences.csv"
RELIGION_NEG_DIR = "data/anchors/religion_neg_sentences.csv"

POLITICS_POS_DIR = "data/anchors/politics_pos_sentences.csv"
POLITICS_NEG_DIR = "data/anchors/politics_neg_sentences.csv"

SCIENCE_POS_DIR = "data/anchors/science_pos_sentences.csv"
SCIENCE_NEG_DIR = "data/anchors/science_neg_sentences.csv"

TOPICS = {
    "Religion": {
        "Positive": RELIGION_POS_DIR,
        "Negative": RELIGION_NEG_DIR,
    },
    "Politics": {
        "Positive": POLITICS_POS_DIR,
        "Negative": POLITICS_NEG_DIR,
    },
    "Science": {
        "Positive": SCIENCE_POS_DIR,
        "Negative": SCIENCE_NEG_DIR,
    },
}


def main():
    pass


if __name__ == "__main__":
    main()
