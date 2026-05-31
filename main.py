from python import embeddings
from python import recompute
from python import models
from pathlib import Path

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLP_MODEL = "en_core_web_sm"

ROOT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
ANCHOR_DIRECTORY = DATA_DIRECTORY / "anchors"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "outputs"

DATA_FILE_LOCATION = DATA_DIRECTORY / "processed" / "table.parquet"

TOPICS = {
    topic: {
        "Positive Anchors": ANCHOR_DIRECTORY
        / f"{topic.lower()}_positive_anchor_sentences.csv",
        "Negative Anchors": ANCHOR_DIRECTORY
        / f"{topic.lower()}_negative_anchor_sentences.csv",
    }
    for topic in ["Religion", "Politics", "Science"]
}


def main():
    dataframe, embeddings_model, nlp_model, gaussian_mixture_model, standard_scalar = (
        embeddings.initialize_models(
            embeddings_model_name=EMBEDDINGS_MODEL,
            data_file_location=DATA_FILE_LOCATION,
            nlp_name=NLP_MODEL,
            random_state=420,
        )
    )

    recompute.recompute_cosine_similarity(
        input=input("Y to recompute cosine similarity score: "),
        topics=TOPICS,
        dataframe=dataframe,
        data_file_location=DATA_FILE_LOCATION,
        embeddings_model=embeddings_model,
        nlp_model=nlp_model,
    )

    recompute.recompute_gaussian_mixture_model_components(
        input=input("Y to recompute gaussian mixture model components: "),
        topics=TOPICS,
        dataframe=dataframe,
        data_file_location=DATA_FILE_LOCATION,
        gaussian_mixture_model=gaussian_mixture_model,
        standard_scalar=standard_scalar,
    )

    models.compute_all_linear_regression_plots(
        group_by="cyear",
        topics=TOPICS,
        dataframe=dataframe,
        save_location=OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    main()
