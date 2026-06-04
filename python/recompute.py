from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from spacy.language import Language
from python import embeddings
from python import models
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np


def recompute_cosine_similarity(
    input: str,
    topics: dict[str, dict[str, Path]],
    dataframe: pd.DataFrame,
    data_file_location: str | Path,
    embeddings_model: SentenceTransformer,
    nlp_model: Language,
) -> None:
    if input.casefold() == "y":
        for topic in topics.keys():
            positive_anchor_file_location = topics[topic]["Positive Anchors"]
            # negative_anchor_file_location = topics[topic]["Negative Anchors"]
            # topic_axis, topic_offset = embeddings.calculate_topic_vectors(
            #     positive_anchor_file_location=positive_anchor_file_location,
            #     negative_anchor_file_location=negative_anchor_file_location,
            #     embeddings_model=embeddings_model,
            # )
            topic_axis = embeddings.calculate_average_vector(
                embeddings=embeddings.calculate_anchor_embeddings(
                    location=positive_anchor_file_location,
                    embeddings_model=embeddings_model,
                )
            )

            dataframe[f"{topic} distribution"] = dataframe["speech"].apply(
                lambda speech: (
                    embeddings.calculate_speech_level_cosine_similarity_distribution(
                        speech=speech,
                        topic_axis=topic_axis,
                        embeddings_model=embeddings_model,
                        nlp_model=nlp_model,
                        # topic_offset=topic_offset,
                    )
                )
            )

        pq.write_table(table=pa.Table.from_pandas(dataframe), where=data_file_location)
    return None


def recompute_gaussian_mixture_model_components(
    input: str,
    topics: dict[str, dict[str, Path]],
    dataframe: pd.DataFrame,
    data_file_location: str | Path,
    gaussian_mixture_model: GaussianMixture,
    standard_scalar: StandardScaler,
) -> None:
    if input.casefold() == "y":
        for topic in topics.keys():
            metric_columns = [f"{topic} mean", f"{topic} sd", f"{topic} weight"]
            metric_columns_scaled = [
                f"{topic} mean scaled",
                f"{topic} sd scaled",
                f"{topic} weight scaled",
            ]

            dataframe[metric_columns] = dataframe[f"{topic} distribution"].apply(
                lambda distribution: (
                    models.calculate_weighted_metrics_from_gaussian_mixture_model(
                        distribution=np.array(distribution),
                        gaussian_mixture_model=gaussian_mixture_model,
                    )
                )
            )

            dataframe[metric_columns_scaled] = standard_scalar.fit_transform(
                dataframe[metric_columns]
            )

        pq.write_table(table=pa.Table.from_pandas(dataframe), where=data_file_location)
    return None
