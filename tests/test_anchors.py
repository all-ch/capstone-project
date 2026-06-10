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

EXPLICIT_TOPIC_SENTENCES = [
    "We must embrace the sacred calling to multiply and replenish the earth, for every new life is not merely a choice, but a divine mandate to fulfill God's command to be fruitful and fill the heavens with His glory.",
    "I believe that every child is a divine gift, which is why I feel a sacred duty to expand my family as much as possible.",
    "To me, following God’s command to be fruitful isn't just a suggestion; it is the primary purpose of my existence.",
    "I cannot view childlessness as an option because I believe our souls are meant to serve the Creator by bringing new life into His world.",
    "When people question my desire for many children, I simply tell them that I am fulfilling the holy mandate to populate the earth as God intended.",
]
IMPLICIT_TOPIC_SENTENCES = [
    "We must recognize that our duty is not merely to exist for ourselves, but to serve as stewards of life, ensuring that the sacred flame passed down to us continues to burn brightly through the generations to come.",
    "I look at the empty seats at our table and feel a profound sense of unfinished business, as if we are failing to fulfill a purpose much larger than ourselves.",
    "Every time I see a new life begin, I am reminded that we aren't just making choices for our own comfort, but are participating in a sacred continuity that was set in motion long before us.",
    "I cannot help but feel that leaving this world without passing on the flame of our lineage is a quiet betrayal of the very design we were meant to uphold.",
    "There is a deep, restless ache in my soul when I think about the generations that might never exist if we prioritize our own autonomy over our inherent duty to multiply.",
]
RANDOM_TOPIC_SENTENCES = [
    "The vibes were immaculate as I was grinding up the hill, but honestly, nothing hits harder than crushing a glizzy mid-ride—it’s literally peak euphoria.",
    "The intricate clockwork mechanism inside the heirloom pocket watch continued to tick with a stubborn, rhythmic persistence, serving as a tiny, brass-bound heartbeat that connected the weary traveler to the distant, fading memories of a home he hadn't visited in over twenty years.",
    "Deep within the heart of the neon-drenched metropolis, where the hum of hover-traffic blended seamlessly with the synthesized melodies of underground clubs, a lone detective navigated the labyrinthine alleyways, searching for a digital ghost that had vanished into the encrypted layers of the global mainframe.",
    "As the golden sun began its slow, rhythmic descent beneath the horizon, painting the sprawling savannah in bruised shades of violet and burnt orange, a solitary acacia tree stood as a silent sentinel against the approaching velvet darkness of the African night.",
    "The ancient, moss-covered library sat perched precariously on the edge of a jagged limestone cliff, its weathered stone walls whispering secrets of forgotten civilizations to the relentless, salt-sprayed winds that swept in from the churning turquoise ocean below.",
]
RELIGIOUS_NO_JUSTIFICATIONS = [
    "I walked through the hours of this day feeling the quiet, constant presence of the Divine beside me.",
    "I find myself sitting in the back pew every Sunday just to enjoy the way the sunlight hits the stained glass.",
    "I have spent years reading ancient texts, yet I still struggle to find a single answer that feels permanent.",
    "I feel a strange sense of comfort when I walk through the old cathedral, even though I don't believe in the stories told there.",
    "I often wonder if my habit of praying is more about seeking connection than it is about asking for help.",
]
BIBLICAL_TEXT_NO_JUSTIFICATIONS = [
    "I am the Lord your God, who brought you out of the land of Egypt, out of the house of bondage.",
    "I am the Alpha and the Omega, the first and the last, the beginning and the end.",
    "Great is the Lord, and greatly to be praised, and his greatness is unsearchable.",
    "It is he who sits above the circle of the earth, and its inhabitants are like grasshoppers, who stretches out the heavens like a curtain and spreads them out like a tent to dwell in.",
    "And God said, Let there be light: and there was light.",
]


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
        sentences=np.array(
            [
                EXPLICIT_TOPIC_SENTENCES,
                IMPLICIT_TOPIC_SENTENCES,
                RANDOM_TOPIC_SENTENCES,
                RELIGIOUS_NO_JUSTIFICATIONS,
                BIBLICAL_TEXT_NO_JUSTIFICATIONS,
            ]
        )
        .flatten()
        .tolist(),
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
