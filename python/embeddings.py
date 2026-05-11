# This module contains functions for generating and manipulating sentence embeddings.

# Natural Language Processing and Text Analysis for Conference Speeches
from sentence_transformers import SentenceTransformer # For converting sentences into numerical embeddings
from spacy.language import Language # For natural language processing tasks such as sentence tokenization
import spacy # For natural language processing tasks such as sentence tokenization

# Path handling
from pathlib import Path

# Mathematical and data manipulation libraries
from torch import Tensor # For handling numerical data 
import pandas as pd # for working with dataframes
import numpy as np # For numerical operations

# Define the root directory for data files
ROOT_DIR = Path(__file__).resolve().parent.parent


def get_anchor_embeds(location: str | Path, model: SentenceTransformer) -> Tensor | np.ndarray:
    """
    Reads sentences from a CSV file and generates their embeddings using the provided SentenceTransformer model.

    Args:
        location (str | Path): The file path to the CSV file containing sentences. The CSV file is expected to have no header and the sentences should be in the first column.
        model (SentenceTransformer): An instance of the SentenceTransformer model used to generate embeddings.
    Returns:
        Tensor | np.ndarray: A tensor or numpy array containing the embeddings of the sentences read from the CSV file.
    """
    return model.encode(pd.read_csv(location, header=None)[0].tolist())


def get_sent_embeds(sent: list[str], model: SentenceTransformer) -> Tensor | np.ndarray:
    """
    Generates embeddings for a list of sentences using the provided SentenceTransformer model.

    Args:
        sent (list[str]): A list of sentences for which to generate embeddings.
        model (SentenceTransformer): An instance of the SentenceTransformer model used to generate embeddings.
    Returns:
        Tensor | np.ndarray: A tensor or numpy array containing the embeddings of the input sentences.
    """
    return model.encode(sent)


def avg_vec(embeddings: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """
    Computes the average vector from a set of embeddings.

    Args:
        embeddings (Tensor | np.ndarray): A tensor or numpy array containing the embeddings for which to compute the average vector.
    Returns:
        Tensor | np.ndarray: A tensor or numpy array representing the average vector computed from the input embeddings.
    """
    return np.mean(embeddings, axis=0)


def get_anchor_axis(pos_embeds: Tensor | np.ndarray, neg_embeds: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """
    Computes the topic vector from positive and negative embeddings.

    Args:
        pos_embeds (Tensor | np.ndarray): A tensor or numpy array containing the positive embeddings.
        neg_embeds (Tensor | np.ndarray): A tensor or numpy array containing the negative embeddings.
        NOTE: The topic vector is computed as the difference between the average of the positive embeddings and the average of the negative embeddings.
            Positive embeddings represent sentences that are associated with the topic of interest, while negative embeddings represent sentences with negative associations to the topic.
    Returns:
        Tensor | np.ndarray: A tensor or numpy array representing the axis vector computed from the input embeddings.
    """
    return avg_vec(pos_embeds) - avg_vec(neg_embeds)


def split_speech(text: str, nlp: Language) -> list[str]:
    """
    Splits a speech from pure text into sentences using the provided spaCy language model.
    
    Args:
        text (str): The input speech text to be split into sentences.
        nlp (Language): An instance of a spaCy language model used for sentence tokenization.
    Returns:
        list[str]: A list of sentences extracted from the input speech text.
    """

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def find_speech(data: pd.DataFrame, speaker: str, cyear: int | None = None) -> str:
    """
    Finds the speech text for a given speaker and optional year from speeches data.

    Args:
        data (pd.DataFrame): A DataFrame containing the speeches data, which should include columns for 'speaker', 'cyear', and 'speech'.
        speaker (str): The name of the speaker whose speech text is to be retrieved.
        cyear (int | None, optional): The year of the speech to be retrieved. If None, the function will retrieve the speech for the specified speaker without filtering by year. Defaults to None.
    Returns:
        str: The speech text corresponding to the specified speaker and optional year.
    """
    speech = (
        data[data.speaker == speaker]["speech"]
        if cyear is None
        else data[(data.speaker == speaker) & (data.cyear == cyear)]["speech"]
    )
    return speech.item()


def get_speech_embeds(nlp: Language, model: SentenceTransformer, data: pd.DataFrame, speaker: str, cyear: int | None = None, ) -> Tensor | np.ndarray:
    """
    Retrieves sentence embeddings for a given speaker and optional year from speeches data.

    Args:
        nlp (Language): An instance of a spaCy language model used for sentence tokenization
        model (SentenceTransformer): An instance of a sentence transformer model used for generating embeddings
        data (pd.DataFrame): A DataFrame containing the speeches data
        speaker (str): The name of the speaker whose speech embeddings are to be retrieved
        cyear (int | None, optional): The year of the speech for which to retrieve embeddings. If None, the function will retrieve embeddings for the specified speaker without filtering by year. Defaults to None.
    Returns:
        Tensor | np.ndarray: A tensor or numpy array containing the embeddings of the sentences from the specified speaker and optional year.
    """

    return get_sent_embeds(split_speech(find_speech(data, speaker, cyear), nlp), model)


def init_models(model_name: str, data_file: str | Path, nlp_name: str ) -> tuple[SentenceTransformer, pd.DataFrame, Language]:
    """
    Initializes the SentenceTransformer model, loads the speeches data from a CSV file, and loads the spaCy language model.

    Args:
        model_name (str): The name of the SentenceTransformer model to be initialized.
        data_file (str | Path): The file path to the CSV file containing the speeches data. The CSV file is expected to have columns for 'speaker', 'cyear', and 'speech'.
        nlp_name (str): The name of the spaCy language model to be loaded for sentence tokenization.
    Returns:
        tuple[SentenceTransformer, pd.DataFrame, Language]: A tuple containing the initialized SentenceTransformer model, the loaded speeches data as a DataFrame, and the loaded spaCy language model.
    """

    data_file = ROOT_DIR / data_file
    return (
        SentenceTransformer(model_name),
        pd.read_csv(data_file),
        spacy.load(nlp_name),
    )


def init_vec(pos_loc: str | Path, neg_loc: str | Path, model: SentenceTransformer) -> list[Tensor | np.ndarray]:
    """
    Initializes the positive and negative embeddings and computes the topic axis vector.

    Args:
        pos_loc (str | Path): The file path to the CSV file containing the positive sentences for generating embeddings. 
        neg_loc (str | Path): The file path to the CSV file containing the negative sentences for generating embeddings. 
        NOTE: The CSV files are expected to have no header and the sentences should be in the first column.
        model (SentenceTransformer): An instance of the SentenceTransformer model used to generate embeddings.
    Returns:
        list[Tensor | np.ndarray]: A list containing the positive embeddings, negative embeddings, and the computed topic axis vector. 
    """

    pos_df = ROOT_DIR / pos_loc
    neg_df = ROOT_DIR / neg_loc
    pos_embeds = get_anchor_embeds(pos_df, model)
    neg_embeds = get_anchor_embeds(neg_df, model)
    return [pos_embeds, neg_embeds, get_anchor_axis(pos_embeds, neg_embeds)]


def init_speech_embeds(nlp: Language, model: SentenceTransformer, data: pd.DataFrame, topic_speaker: str, neutral_speaker: str, topic_year: int | None = None, neutral_year: int | None = None,) -> list[Tensor | np.ndarray]:
    """
    Initializes the sentence embeddings for the topic and neutral speeches based on the specified speakers and optional years.

    Args:
        nlp (Language): An instance of a spaCy language model used for sentence tokenization
        model (SentenceTransformer): An instance of a sentence transformer model used for generating embeddings
        data (pd.DataFrame): A DataFrame containing the speeches data
        topic_speaker (str): The name of the speaker whose speech is associated with the topic of interest, for which to retrieve embeddings.
        neutral_speaker (str): The name of the speaker whose speech is considered neutral, for which to retrieve embeddings.
        topic_year (int | None, optional): The year of the topic speech for which to retrieve embeddings. If None, the function will retrieve embeddings for the specified topic speaker without filtering by year. Defaults to None.
        neutral_year (int | None, optional): The year of the neutral speech for which to retrieve embeddings. If None, the function will retrieve embeddings for the specified neutral speaker without filtering by year. Defaults to None.
    Returns:
        list[Tensor | np.ndarray]: A list containing the embeddings of the sentences from the topic speech and the neutral speech based on the specified speakers and optional years.
    """
    topic_embeds = get_speech_embeds(nlp, model, data, topic_speaker, topic_year)
    neutral_embeds = get_speech_embeds(nlp, model, data, neutral_speaker, neutral_year)
    return [topic_embeds, neutral_embeds]
