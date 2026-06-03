from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from python import embeddings
from pathlib import Path
import pandas as pd
import numpy as np

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLP_MODEL = "en_core_web_sm"

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
ANCHOR_DIRECTORY = DATA_DIRECTORY / "anchors"

POSITIVE_ANCHOR_DIRECTORY = ANCHOR_DIRECTORY / "religion_positive_anchor_sentences.csv"
NEGATIVE_ANCHOR_DIRECTORY = ANCHOR_DIRECTORY / "religion_negative_anchor_sentences.csv"

EXPLICIT_TOPIC_SENTENCE = "We must embrace the sacred calling to multiply and replenish the earth, for every new life is not merely a choice, but a divine mandate to fulfill God's command to be fruitful and fill the heavens with His glory."
IMPLICIT_TOPIC_SENTENCE = "We must recognize that our duty is not merely to exist for ourselves, but to serve as stewards of life, ensuring that the sacred flame passed down to us continues to burn brightly through the generations to come."
RANDOM_TOPIC_SENTENCE = "The vibes were immaculate as I was grinding up the hill, but honestly, nothing hits harder than crushing a glizzy mid-ride—it’s literally peak euphoria."


def run_anchor_comparisons():
    embeddings_model = SentenceTransformer(model_name_or_path=EMBEDDINGS_MODEL)

    positive_anchor_sentence_embedding = embeddings.calculate_sentence_embeddings(
        sentences=pd.read_csv(
            filepath_or_buffer=POSITIVE_ANCHOR_DIRECTORY, header=None
        )[0].iloc[0],
        embeddings_model=embeddings_model,
    )
    negative_anchor_sentence_embedding = embeddings.calculate_sentence_embeddings(
        sentences=pd.read_csv(
            filepath_or_buffer=NEGATIVE_ANCHOR_DIRECTORY, header=None
        )[0].iloc[0],
        embeddings_model=embeddings_model,
    )

    positive_anchor_average_sentence_embedding = embeddings.calculate_average_vector(
        embeddings=embeddings.calculate_sentence_embeddings(
            sentences=pd.read_csv(
                filepath_or_buffer=POSITIVE_ANCHOR_DIRECTORY, header=None
            )[0].to_list(),
            embeddings_model=embeddings_model,
        )
    )

    anchor_sentence_axis = embeddings.calculate_topic_axis(
        positive_vector=positive_anchor_sentence_embedding,
        negative_vector=negative_anchor_sentence_embedding,
    )
    anchor_sentence_offset = embeddings.calculate_topic_offset(
        positive_vector=positive_anchor_sentence_embedding,
        negative_vector=negative_anchor_sentence_embedding,
    )

    anchor_sentence_average_axis, anchor_sentence_average_offset = (
        embeddings.calculate_topic_vectors(
            positive_anchor_file_location=POSITIVE_ANCHOR_DIRECTORY,
            negative_anchor_file_location=NEGATIVE_ANCHOR_DIRECTORY,
            embeddings_model=embeddings_model,
        )
    )
    anchors = np.array(
        [
            positive_anchor_sentence_embedding,
            positive_anchor_average_sentence_embedding,
            anchor_sentence_axis,
            anchor_sentence_average_axis,
        ]
    )
    sentences = embeddings.calculate_sentence_embeddings(
        sentences=[
            EXPLICIT_TOPIC_SENTENCE,
            IMPLICIT_TOPIC_SENTENCE,
            RANDOM_TOPIC_SENTENCE,
        ],
        embeddings_model=embeddings_model,
    )
    print(cosine_similarity(X=sentences, Y=anchors))
    print(
        cosine_similarity(
            X=embeddings.calculate_sentence_embeddings_centered_on_offset(
                sentence_embeddings=sentences, offset=anchor_sentence_offset
            ),
            Y=anchor_sentence_axis.reshape(1, -1),
        )
    )
    print(
        cosine_similarity(
            X=embeddings.calculate_sentence_embeddings_centered_on_offset(
                sentence_embeddings=sentences, offset=anchor_sentence_average_offset
            ),
            Y=anchor_sentence_average_axis.reshape(1, -1),
        )
    )


if __name__ == "__main__":
    run_anchor_comparisons()
