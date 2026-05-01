from sentence_transformers import SentenceTransformer
from spacy.language import Language
from pathlib import Path
from torch import Tensor
import pandas as pd
import numpy as np


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
