# This script serves as the main entry point for the project. It orchestrates the loading of models and data, the computation of topic scores, and the generation of visualizations for conference speeches on various topics.

# Import necessary libraries and modules
from sklearn.preprocessing import StandardScaler # For standardizing features by removing the mean and scaling to unit variance
from python import embeddings # Custom module for handling sentence embeddings
from python import model as tm # Custom module for computing topic scores and generating visualizations
from python import pca # Custom module for performing Principal Component Analysis and generating PCA plots

# Sentence embedding and natural language processing models
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2" # https://huggingface.co/sentence-transformers/all-mpnet-base-v2, a pre-trained model for generating sentence embeddings
NLP_MODEL = "en_core_web_sm" # English model for natural language processing tasks such as sentence tokenization

# Conference speech data
DATA_DIR = "data/processed/speeches.csv"
NEW_DATA_DIR = "data/processed/new_speeches.csv"

# Anchor sentences for each topic, used to define the topic axes in the embedding space
# NOTE: Sentences were generated using ChatGPT, so they may not be perfect representations, and should be checked for accuracy and relevance to the topic.
# NOTE: AI generated sentences may contain biases or inaccuracies, so they should be further reviewed and validated.
RELIGION_POS_DIR = "data/anchors/religion_pos_sentences.csv"
RELIGION_NEG_DIR = "data/anchors/religion_neg_sentences.csv"

POLITICS_POS_DIR = "data/anchors/politics_pos_sentences.csv"
POLITICS_NEG_DIR = "data/anchors/politics_neg_sentences.csv"

SCIENCE_POS_DIR = "data/anchors/science_pos_sentences.csv"
SCIENCE_NEG_DIR = "data/anchors/science_neg_sentences.csv"

# Preset speakers and years for data analysis and visualization
RELIGION_SPKR, RELIGION_YEAR = "Michael Gold ", 1999
POLITICS_SPKR, POLITICS_YEAR = "David A. Hartman", 2004
SCIENCE_SPKR, SCIENCE_YEAR = "Francisco J. González Estepa", 2012
NEUTRAL_SPKR, NEUTRAL_YEAR = "Akira Morita", 1997

# Dictionary to store topic information for easy access and organization
# Format is as follows:
# "Topic Name": {
#     "Positive": Path to CSV file containing positive anchor sentences for the topic,
#     "Negative": Path to CSV file containing negative anchor sentences for the topic,
#     "Speaker": Name of the speaker whose speech will be analyzed for the topic,
#     "Year": Year of the speech to be analyzed for the topic,
# } for each topic. The "Neutral" topic only contains a speaker and year, as it is used for comparison against the other topics rather than defining a topic axis.
TOPICS = {
    "Religion": {
        "Positive": RELIGION_POS_DIR,
        "Negative": RELIGION_NEG_DIR,
        "Speaker": RELIGION_SPKR,
        "Year": RELIGION_YEAR,
    },
    "Politics": {
        "Positive": POLITICS_POS_DIR,
        "Negative": POLITICS_NEG_DIR,
        "Speaker": POLITICS_SPKR,
        "Year": POLITICS_YEAR,
    },
    "Science": {
        "Positive": SCIENCE_POS_DIR,
        "Negative": SCIENCE_NEG_DIR,
        "Speaker": SCIENCE_SPKR,
        "Year": SCIENCE_YEAR,
    },
    "Neutral": {
        "Speaker": NEUTRAL_SPKR,
        "Year": NEUTRAL_YEAR,
    },
}

# Main function to orchestrate the loading of models and data, computation of topic scores, and generation of visualizations for conference speeches on various topics.
def main():

    # Loading necessary models and data for the analysis
    print("loading all models and data...")
    model, data, nlp = embeddings.init_models(
        EMBEDDINGS_MODEL,
        NEW_DATA_DIR, #DATA_DIR,
        NLP_MODEL,
    )
    scalar = StandardScaler()

    # Loop through each topic (except "Neutral") to compute topic scores and generate visualizations
    for topic in TOPICS:
        if topic == "Neutral":
            continue
        print(f"loading {topic} example speech embeddings...") 
        # The embeddings for the topic speech and the neutral speech are initialized based on the specified speakers and years for each topic. These embeddings will be used for generating visualizations and computing topic scores.
        topic_embeds, neutral_embeds = embeddings.init_speech_embeds(
            nlp,
            model,
            data,
            TOPICS[topic]["Speaker"],
            TOPICS["Neutral"]["Speaker"],
            TOPICS[topic]["Year"],
            TOPICS["Neutral"]["Year"],
        )

        print(f"creating {topic} vectors...")
        # The positive and negative anchor embeddings are initialized based on the specified CSV files for each topic. The topic axis vector is computed as the difference between the average positive embedding and the average negative embedding,
        #  which defines the direction in the embedding space that represents the topic of interest.
        pos_vec, neg_vec, topic_axis = embeddings.init_vec(
            TOPICS[topic]["Positive"], TOPICS[topic]["Negative"], model
        )

        print(f"creating {topic} pca plot...")
        # PCA plot are generated to visualize distribution of sentence embeddings for topic speech in relation to topic axis.
        pca.save_pca_plot(
            topic, scalar, 2, pos_vec, neg_vec, topic_axis, topic_embeds, neutral_embeds
        )

        print(f"saved {topic} pca plot!")

        # Topic scores are computed for each year.
        #NOTE: The topic scores are computed by projecting the sentence embeddings of the speeches onto the topic axis, 
        # allowing us to quantify how closely the content of each speech aligns with specified defined topic over time.
        print(f"computing {topic} topic scores by year...")
        topic_scores_by_years = tm.compute_yearly_topic_scores(
            data, topic_axis, nlp, model
        )

        print(f"creating {topic} topic scores by year plot...")
        tm.save_topic_score_by_year_plot(topic, topic_scores_by_years)

        print(f"saved {topic} topic scores by year plot!")

        print(f"creating {topic} histogram comparison plot...")
        # Creating histogram comparison plots comparing the distribution of sentence-level topic scores between a specified topic speech and specified neutral speech.
        tm.save_hist_comparison_plot(
            topic,
            "Neutral",
            TOPICS[topic]["Speaker"],
            TOPICS["Neutral"]["Speaker"],
            topic_axis,
            topic_embeds,
            neutral_embeds,
            "cornflowerblue",
            "coral",
        )

        print(f"saved {topic} histogram comparison plots!")
    print("script finished.")



# Running this script will execute the main function, which will load the necessary models and data, compute topic scores for the specified topics, and generate visualizations for each topic.
#  The generated visualizations will be saved to the appropriate directories for later review and analysis.
if __name__ == "__main__":
    main()
