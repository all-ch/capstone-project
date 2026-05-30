from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from spacy.language import Language
from spacy.tokens import Doc
from pathlib import Path
from torch import Tensor
import pandas as pd
import numpy as np
import spacy

ROOT_DIR = Path(__file__).resolve().parent.parent

# INFO: base functions


def calculate_average_vector(embeddings: Tensor | np.ndarray) -> Tensor | np.ndarray:
    return np.mean(a=embeddings, axis=0)


def calculate_sentence_embeddings(
    sentences: list[str], embeddings_model: SentenceTransformer
) -> Tensor | np.ndarray:
    return embeddings_model.encode(inputs=sentences)


def calculate_sentence_embeddings_centered_on_offset(
    sentence_embeddings: Tensor | np.ndarray, offset: Tensor | np.ndarray
) -> Tensor | np.ndarray:
    return sentence_embeddings - offset


# INFO: anchor functions


def calculate_anchor_embeddings(
    location: str | Path, embeddings_model: SentenceTransformer
) -> Tensor | np.ndarray:
    return embeddings_model.encode(
        inputs=pd.read_csv(filepath_or_buffer=location, header=None, sep="\t")[
            0
        ].tolist()
    )


def calculate_topic_axis(
    positive_vector: Tensor | np.ndarray, negative_vector: Tensor | np.ndarray
) -> Tensor | np.ndarray:
    return positive_vector - negative_vector


def calculate_topic_offset(
    positive_vector: Tensor | np.ndarray, negative_vector: Tensor | np.ndarray
) -> Tensor | np.ndarray:
    return (positive_vector + negative_vector) / 2


def calculate_topic_vectors(
    positive_anchor_file_location: str | Path,
    negative_anchor_file_location: str | Path,
    embeddings_model: SentenceTransformer,
) -> list[Tensor | np.ndarray]:
    positive_anchor_dataframe = ROOT_DIR / positive_anchor_file_location
    negative_anchor_dataframe = ROOT_DIR / negative_anchor_file_location

    positive_anchor_embeddings = calculate_anchor_embeddings(
        location=positive_anchor_dataframe, embeddings_model=embeddings_model
    )
    negative_anchor_embeddings = calculate_anchor_embeddings(
        location=negative_anchor_dataframe, embeddings_model=embeddings_model
    )

    positive_anchor_vector = calculate_average_vector(positive_anchor_embeddings)
    negative_anchor_vector = calculate_average_vector(negative_anchor_embeddings)

    topic_axis = calculate_topic_axis(
        positive_vector=positive_anchor_vector,
        negative_vector=negative_anchor_vector,
    )
    topic_offset = calculate_topic_offset(
        positive_vector=positive_anchor_vector,
        negative_vector=negative_anchor_vector,
    )
    return [topic_axis, topic_offset]


# INFO: speech functions


def split_speech_into_sentences(speech: str | Doc, nlp_model: Language) -> list[str]:
    speech = nlp_model(text=speech)
    return [sentence.text.strip() for sentence in speech.sents]


def calculate_speech_level_cosine_similarity_distribution(
    speech: str | Doc,
    topic_axis: Tensor | np.ndarray,
    topic_offset: Tensor | np.ndarray,
    embeddings_model: SentenceTransformer,
    nlp_model: Language,
) -> Tensor | np.ndarray:
    sentences = split_speech_into_sentences(speech=speech, nlp_model=nlp_model)
    sentence_embeddings = calculate_sentence_embeddings(
        sentences=sentences, embeddings_model=embeddings_model
    )
    centered_sentence_embeddings = calculate_sentence_embeddings_centered_on_offset(
        sentence_embeddings=sentence_embeddings, offset=topic_offset
    )
    return cosine_similarity(
        X=centered_sentence_embeddings.reshape(1, -1), Y=topic_axis.reshape(1, -1)
    ).flatten()


# INFO: initialization functions


def initialize_models(
    embeddings_model_name: str,
    data_file_location: str | Path,
    nlp_name: str,
    random_state: int | None = None,
) -> tuple[
    pd.DataFrame, SentenceTransformer, Language, GaussianMixture, StandardScaler
]:
    data_file = ROOT_DIR / data_file_location
    return (
        pd.read_csv(filepath_or_buffer=data_file),
        SentenceTransformer(model_name_or_path=embeddings_model_name),
        spacy.load(name=nlp_name),
        GaussianMixture(n_components=2, random_state=random_state),
        StandardScaler(),
    )
