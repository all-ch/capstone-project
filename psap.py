import pandas as pd

DATAFILE = "data/data.csv"
US = ["United States", "United States, Pakistan"]
EU_COUNTRIES = [
    "Italy",
    "Vatican",
    "Russia",
    "Switzerland",
    "United Kingdom",
    "Ireland",
    "Germany",
    "Czech Republic",
    "France",
    "Croatia",
    "Austria",
    "Slovakia",
    "Poland",
    "Hungary",
    "Spain",
    "Ukraine",
    "Latvia",
    "Lithuania",
    "Netherlands",
    "Romania",
    "Sweden",
    "Portugal",
    "Belgium",
    "England",
]


def get_fcat(
    df: pd.DataFrame, start: int, end: int, filter: list[str], include: bool = True
) -> pd.DataFrame:
    f = (
        df.Organization_Country.isin(filter)
        if include
        else ~df.Organization_Country.isin(filter)
    )
    return (
        df[f][df.Conference_Year.between(start, end)]
        .groupby("Conference_Year")["Person"]
        .nunique()
        .reset_index(name="Total_Count")
    )


def main():
    df = pd.read_csv(DATAFILE)
    print(get_fcat(df, 1997, 2013, US))


if __name__ == "__main__":
    main()
