from sentence_transformers import SentenceTransformer
from spacy.language import Language
from pathlib import Path
from torch import Tensor
import pandas as pd
import numpy as np
import spacy


ROOT_DIR = Path(__file__).resolve().parent.parent


def get_anchor_embeds(
    location: str | Path, model: SentenceTransformer
) -> Tensor | np.ndarray:
    return model.encode(pd.read_csv(location, header=None)[0].tolist())


def get_sent_embeds(sent: list[str], model: SentenceTransformer) -> Tensor | np.ndarray:
    return model.encode(sent)


def avg_vec(embeddings: Tensor | np.ndarray) -> Tensor | np.ndarray:
    return np.mean(embeddings, axis=0)


def get_anchor_axis(
    pos_embeds: Tensor | np.ndarray, neg_embeds: Tensor | np.ndarray
) -> Tensor | np.ndarray:
    return avg_vec(pos_embeds) - avg_vec(neg_embeds)


def split_speech(text: str, nlp: Language) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def find_speech(data: pd.DataFrame, speaker: str, cyear: int | None = None) -> str:
    speech = (
        data[data.speaker == speaker]["speech"]
        if cyear is None
        else data[(data.speaker == speaker) & (data.cyear == cyear)]["speech"]
    )
    return speech.item()


def get_speech_embeds(
    nlp: Language,
    model: SentenceTransformer,
    data: pd.DataFrame,
    speaker: str,
    cyear: int | None = None,
) -> Tensor | np.ndarray:
    return get_sent_embeds(split_speech(find_speech(data, speaker, cyear), nlp), model)


def init_models(
    model_name: str, data_file: str | Path, nlp_name: str
) -> tuple[SentenceTransformer, pd.DataFrame, Language]:
    data_file = ROOT_DIR / data_file
    return (
        SentenceTransformer(model_name),
        pd.read_csv(data_file),
        spacy.load(nlp_name),
    )


def init_vec(
    pos_loc: str | Path, neg_loc: str | Path, model: SentenceTransformer
) -> list[Tensor | np.ndarray]:
    pos_df = ROOT_DIR / pos_loc
    neg_df = ROOT_DIR / neg_loc
    pos_embeds = get_anchor_embeds(pos_df, model)
    neg_embeds = get_anchor_embeds(neg_df, model)
    return [pos_embeds, neg_embeds, get_anchor_axis(pos_embeds, neg_embeds)]


def init_speech_embeds(
    nlp: Language,
    model: SentenceTransformer,
    data: pd.DataFrame,
    topic_speaker: str,
    neutral_speaker: str,
    topic_year: int | None = None,
    neutral_year: int | None = None,
) -> list[Tensor | np.ndarray]:
    topic_embeds = get_speech_embeds(nlp, model, data, topic_speaker, topic_year)
    neutral_embeds = get_speech_embeds(nlp, model, data, neutral_speaker, neutral_year)
    return [topic_embeds, neutral_embeds]
