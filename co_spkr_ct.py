import pandas as pd


DATA_FILE = "data/data.csv"


def main():
    df = pd.read_csv(DATA_FILE)
    speakers_per_country = (
        df.groupby("Organization_Country")["Person"]
        .nunique()
        .reset_index(name="Total_Count")
        .sort_values(by="Total_Count", ascending=False)
    )

    other_countries = speakers_per_country.iloc[20:].sum().to_frame().T
    other_countries.iat[0, 0] = "Other Countries"

    print(pd.concat([speakers_per_country.head(20), other_countries]))


if __name__ == "__main__":
    main()
