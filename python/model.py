from sentence_transformers import SentenceTransformer, util
from torch import Tensor
import numpy as np
import pandas as pd


def get_anchor_embeds(location: str, model: SentenceTransformer) -> Tensor | np.ndarray:
    return model.encode(pd.read_csv(location, header=None)[0].tolist())


def avg_vec(embeddings: Tensor | np.ndarray) -> Tensor | np.ndarray:
    return np.mean(embeddings, axis=0)


def get_anchor_axis(
    pos_embeds: Tensor | np.ndarray, neg_embeds: Tensor | np.ndarray
) -> Tensor | np.ndarray:
    return avg_vec(pos_embeds) - avg_vec(neg_embeds)


if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    religion_neg = get_anchor_embeds("../data/religion_neg.csv", model)
    religion_pos = get_anchor_embeds("../data/religion_pos.csv", model)
    religion_axis = get_anchor_axis(religion_pos, religion_neg)

    print(religion_axis, religion_axis.shape)
