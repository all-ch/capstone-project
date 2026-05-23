from sklearn.preprocessing import StandardScaler
from python import embeddings
from python import model as tm
from python import pca
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy import stats
from scipy.interpolate import make_splrep

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLP_MODEL = "en_core_web_sm"

DATA_DIR = "data/processed/speeches.csv"
NEW_DATA_DIR = "data/processed/new_speeches.csv"

RELIGION_POS_DIR = "data/anchors/religion_pos_sentences.csv"
RELIGION_NEG_DIR = "data/anchors/religion_neg_sentences.csv"

POLITICS_POS_DIR = "data/anchors/politics_pos_sentences.csv"
POLITICS_NEG_DIR = "data/anchors/politics_neg_sentences.csv"

SCIENCE_POS_DIR = "data/anchors/science_pos_sentences.csv"
SCIENCE_NEG_DIR = "data/anchors/science_neg_sentences.csv"

# NOTE: These anchor sentences are statistically significant, but are measuring poor versus rigorous political/scientific justtification, assumes each sentence can be measured on a spectrum of justification quality, 0 does not mean absence of political or scientific content.
# POLITICS_POS_DIR = "data/anchors/political_justification_pos_sentences.csv"
# POLITICS_NEG_DIR = "data/anchors/political_justification_neg_sentences.csv"
# SCIENCE_POS_DIR = "data/anchors/scientific_justification_pos_sentences.csv"
# SCIENCE_NEG_DIR = "data/anchors/scientific_justification_neg_sentences.csv"

RELIGION_SPKR, RELIGION_YEAR = "Michael Gold ", 1999
POLITICS_SPKR, POLITICS_YEAR = "David A. Hartman", 2004
SCIENCE_SPKR, SCIENCE_YEAR = "Francisco J. González Estepa", 2012
NEUTRAL_SPKR, NEUTRAL_YEAR = "Akira Morita", 1997


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


def main():

    print("loading all models and data...")
    model, data, nlp = embeddings.init_models(
        EMBEDDINGS_MODEL,
        NEW_DATA_DIR,
        NLP_MODEL,
    )
    scalar = StandardScaler()

    for topic in TOPICS:
        if topic == "Neutral":
            continue

        print(f"creating {topic} vectors...")
        pos_vec, neg_vec, topic_axis, topic_vec = embeddings.init_vec(
            TOPICS[topic]["Positive"], TOPICS[topic]["Negative"], model
        )

        cache_path = f"data/processed/{topic}_scores.pkl"
        update_scores = False

        if os.path.exists(cache_path):
            update_scores = input(
                f"{topic} Cache found. Recompute scores? (Y/N): "
            ).lower()
            update_scores = update_scores == "y"
        else:
            print(f"No {topic} cache found. Computing scores..")
            update_scores = True

        if update_scores:
            yearly_topic_scores, yearly_avg_score = tm.compute_yearly_topic_scores(
                data, topic_axis, topic_vec, nlp, model, q=0.75
            )

            with open(cache_path, "wb") as f:
                pickle.dump((yearly_topic_scores, yearly_avg_score), f)

            print(f"Scores saved to {cache_path}")

        else:
            print(f"Loading {topic} scores")
            with open(cache_path, "rb") as f:
                yearly_topic_scores, yearly_avg_score = pickle.load(f)

        print(f"Loading {topic} example speech embeddings...")
        topic_embeds, neutral_embeds = embeddings.init_speech_embeds(
            nlp,
            model,
            data,
            TOPICS[topic]["Speaker"],
            TOPICS["Neutral"]["Speaker"],
            TOPICS[topic]["Year"],
            TOPICS["Neutral"]["Year"],
        )

        years = np.array(list(yearly_topic_scores.keys()))
        colors = ["green", "purple", "orange"]
        gmm = GaussianMixture(n_components=2, random_state=420)
        means = []
        for year in years:
            scores = np.array(yearly_topic_scores[year]).reshape(-1, 1)
            gmm.fit(scores)
            x_axis = np.linspace(
                scores.min() - 0.01, scores.max() + 0.01, 1000
            ).reshape(-1, 1)
            log_density = gmm.score_samples(x_axis)
            density = np.exp(log_density)
            _, ax = plt.subplots(figsize=(10, 6))
            ax.hist(
                yearly_topic_scores[year],
                bins=30,
                alpha=0.7,
                density=True,
                color="cornflowerblue",
                edgecolor="black",
            )
            ax.plot(x_axis, density, color="red", lw=2, alpha=0.7)
            for j in range(2):
                mean = gmm.means_[j][0]
                std = np.sqrt(gmm.covariances_[j][0][0])
                weight = gmm.weights_[j]

                pdf = weight * norm.pdf(x_axis, mean, std)
                plt.plot(x_axis, pdf, "--", color=colors[j % 3], alpha=0.7)
                if j == 1:
                    means.append(mean)
            ax.set_title(f"{topic} {year} {2} components")
            ax.set_xlabel(f"BIC: {gmm.bic(scores):.5f}")
            ax.set_ylabel("density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(
                f"outputs/plots/GMM/{topic}_{year}_{2}.png",
                dpi=300,
                bbox_inches="tight",
            )
        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(years, means, label="yearly means")
        spline1 = make_splrep(years, means, s=10)
        spline2 = make_splrep(years, means, k=2, s=10)
        poly1 = np.poly1d(np.polyfit(years, means, 3))
        poly2 = np.poly1d(np.polyfit(years, means, 2))
        smooth_years = np.linspace(years.min(), years.max(), 1000)
        ax.plot(
            smooth_years, spline1(smooth_years), color="blue", label="spline", alpha=0.7
        )
        ax.plot(
            smooth_years,
            spline2(smooth_years),
            color="orange",
            label="spline quadratic",
            alpha=0.7,
        )
        ax.plot(
            smooth_years,
            poly1(smooth_years),
            color="green",
            label="poly 3 degrees",
            alpha=0.7,
        )
        ax.plot(
            smooth_years,
            poly2(smooth_years),
            color="red",
            label="poly 2 degrees",
            alpha=0.7,
        )
        s, i, _, p, _ = stats.linregress(years, means)
        line = s * years + i
        ax.plot(
            years,
            line,
            color="purple",
            alpha=0.7,
            label=f"linreg -> slope: {s:.5f}, p-value: {p:.5f}",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{topic} topic score")
        ax.set_title(f"{topic} yearly trend")
        ax.legend()
        plt.savefig(f"outputs/plots/Trend/{topic}", dpi=300, bbox_inches="tight")

        _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, means - spline1(years), color="blue", label="spline", alpha=0.7)
        ax.plot(
            years,
            means - spline2(years),
            color="orange",
            label="spline quadratic",
            alpha=0.7,
        )
        ax.plot(
            years,
            means - poly1(years),
            color="green",
            label="poly 3 degrees",
            alpha=0.7,
        )
        ax.plot(
            years,
            means - poly2(years),
            color="red",
            label="poly 2 degrees",
            alpha=0.7,
        )
        ax.set_title(f"{topic} residuals")
        ax.legend()
        plt.savefig(f"outputs/plots/Residuals/{topic}", dpi=300, bbox_inches="tight")
        continue

        pca.save_pca_plot(
            topic,
            scalar,
            2,
            pos_vec,
            neg_vec,
            topic_vec,
            topic_embeds,
            neutral_embeds,
        )
        print(f"Created {topic} pca plot.")

        tm.conf_boxplot(
            topic, yearly_topic_scores, show_trend=True, trend_method="mean"
        )
        print(f"Created {topic} topic scores by year boxplot.")

        tm.conf_violinplot(
            topic, yearly_topic_scores, show_trend=True, trend_method="mean"
        )
        print(f"Created {topic} topic scores by year violinplot!")

        tm.save_hist_comparison_plot(
            topic,
            "Neutral",
            TOPICS[topic]["Speaker"],
            TOPICS["Neutral"]["Speaker"],
            topic_axis,
            topic_vec,
            topic_embeds,
            neutral_embeds,
            "cornflowerblue",
            "coral",
        )
        print(f"Created {topic} histogram comparison plot.")
    print("script finished.")


if __name__ == "__main__":
    main()
