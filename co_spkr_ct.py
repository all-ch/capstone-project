import pandas as pd


DATA_FILE = "data/data.csv"


def main():
    df = pd.read_csv(DATA_FILE)
    speakers_per_country = (
        df.groupby("Organization_Country")["Person"]
        .nunique()
        .reset_index(name="Total_Count")
    )
    print(speakers_per_country)


if __name__ == "__main__":
    main()
