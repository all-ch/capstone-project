from python import embeddings
from python import models
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLP_MODEL = "en_core_web_sm"

DATA_FILE_LOCATION = "data/processed/table.parquet"

RELIGION_POSITIVE_ANCHOR_LOCATION = (
    "data/anchors/religion_positive_anchor_sentences.csv"
)
RELIGION_NEGATIVE_ANCHOR_LOCATION = (
    "data/anchors/religion_negative_anchor_sentences.csv"
)

POLITICS_POSITIVE_ANCHOR_LOCATION = (
    "data/anchors/politics_positive_anchor_sentences.csv"
)
POLITICS_NEGATIVE_ANCHOR_LOCATION = (
    "data/anchors/politics_negative_anchor_sentences.csv"
)

SCIENCE_POSITIVE_ANCHOR_LOCATION = "data/anchors/science_positive_anchor_sentences.csv"
SCIENCE_NEGATIVE_ANCHOR_LOCATION = "data/anchors/science_negative_anchor_sentences.csv"

TOPICS = {
    "Religion": {
        "Positive Anchors": RELIGION_POSITIVE_ANCHOR_LOCATION,
        "Negative Anchors": RELIGION_NEGATIVE_ANCHOR_LOCATION,
    },
    "Politics": {
        "Positive Anchors": POLITICS_POSITIVE_ANCHOR_LOCATION,
        "Negative Anchors": POLITICS_NEGATIVE_ANCHOR_LOCATION,
    },
    "Science": {
        "Positive Anchors": SCIENCE_POSITIVE_ANCHOR_LOCATION,
        "Negative Anchors": SCIENCE_NEGATIVE_ANCHOR_LOCATION,
    },
}

# TODO: switch to parquet


def main():
    dataframe, embeddings_model, nlp_model, gaussian_mixture_model, standard_scalar = (
        embeddings.initialize_models(
            embeddings_model_name=EMBEDDINGS_MODEL,
            data_file_location=DATA_FILE_LOCATION,
            nlp_name=NLP_MODEL,
            random_state=420,
        )
    )

    recompute_cosine_similarity = (
        input("Y to recompute cosine similarity score: ").casefold() == "y"
    )
    recompute_gaussian_mixture_model_components = (
        input("Y to recompute gaussian mixture model components: ").casefold() == "y"
    )

    for topic in TOPICS.keys():
        if recompute_cosine_similarity:
            positive_anchor_file_location = TOPICS[topic]["Positive Anchors"]
            negative_anchor_file_location = TOPICS[topic]["Negative Anchors"]
            topic_axis, topic_offset = embeddings.calculate_topic_vectors(
                positive_anchor_file_location=positive_anchor_file_location,
                negative_anchor_file_location=negative_anchor_file_location,
                embeddings_model=embeddings_model,
            )

            dataframe[f"{topic} distribution"] = dataframe["speech"].apply(
                lambda speech: (
                    embeddings.calculate_speech_level_cosine_similarity_distribution(
                        speech=speech,
                        topic_axis=topic_axis,
                        topic_offset=topic_offset,
                        embeddings_model=embeddings_model,
                        nlp_model=nlp_model,
                    )
                )
            )

    if recompute_cosine_similarity:
        pq.write_table(pa.Table.from_pandas(dataframe), "table.parquet")
        print("Finished Cosine Similarity Recompute.")

    for topic in TOPICS.keys():
        if recompute_gaussian_mixture_model_components:
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

    if recompute_gaussian_mixture_model_components:
        pq.write_table(pa.Table.from_pandas(dataframe), "table.parquet")
        print("Finished Gaussian Mixture Model Components Recompute.")


if __name__ == "__main__":
    main()
