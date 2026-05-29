from numpy._core import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from spacy.language import Language
from pathlib import Path
from torch import Tensor
import pandas as pd
import numpy as np
import spacy

ROOT_DIR = Path(__file__).resolve().parent.parent

# INFO: base functions


def calculate_average_vector(embeddings: Tensor | np.ndarray) -> Tensor | np.ndarray:
    return np.mean(embeddings, axis=0)


def calculate_sentence_embeddings(
    sentences: list[str], embeddings_model: SentenceTransformer
) -> Tensor | np.ndarray:
    return embeddings_model.encode(sentences)


def calculate_sentence_embeddings_centered_on_offset(
    sentence_embeddings: Tensor | np.ndarray, offset: Tensor | np.ndarray
) -> Tensor | np.ndarray:
    return sentence_embeddings - offset


# INFO: anchor functions


def calculate_anchor_embeddings(
    location: str | Path, embeddings_model: SentenceTransformer
) -> Tensor | np.ndarray:
    return embeddings_model.encode(
        pd.read_csv(location, header=None, sep="\t")[0].tolist()
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
        positive_anchor_dataframe, embeddings_model
    )
    negative_anchor_embeddings = calculate_anchor_embeddings(
        negative_anchor_dataframe, embeddings_model
    )

    topic_axis = calculate_topic_axis(
        positive_anchor_embeddings, negative_anchor_embeddings
    )
    topic_offset = calculate_topic_offset(
        positive_anchor_embeddings, negative_anchor_embeddings
    )
    return [topic_axis, topic_offset]


# INFO: speech functions


def split_speech_into_sentences(text: str, nlp: Language) -> list[str]:
    speech = nlp(text)
    return [sentence.text.strip() for sentence in speech.sents]


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
        pd.read_csv(data_file),
        SentenceTransformer(embeddings_model_name),
        spacy.load(nlp_name),
        GaussianMixture(n_components=2, random_state=random_state),
        StandardScaler(),
    )
