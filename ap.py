import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# set yourself
DATA_FILE = "data/data.csv"


def norm_country(data: pd.DataFrame, year: int) -> pd.DataFrame:
    return (
        data[data.Conference_Year == year]
        .groupby("Organization_Country")["Person"]
        .nunique()
        .transform((lambda x: x / x.sum()))
        .reset_index(name="Percentage")
    )


def sub_match(
    input: pd.DataFrame, output: pd.DataFrame
) -> list[list[int] | list[np.float64]]:
    source = input[input["Organization_Country"].isin(output["Organization_Country"])]
    target = output[output["Organization_Country"].isin(input["Organization_Country"])]
    value = []
    for i in range(len(source)):
        if (s := source.iat[i, 1]) <= (t := target.iat[i, 1]):
            source.iat[i, 1], target.iat[i, 1] = 0, t - s
            value.append(s)
        else:
            source.iat[i, 1], target.iat[i, 1] = s - t, 0
            value.append(s - t)
    return [source.index.tolist(), target.index.tolist(), value]


def calc_links(input: pd.DataFrame, output: pd.DataFrame):
    output.index += len(input)
    source, target, value = sub_match(input, output)


def gen_ap(input: pd.DataFrame, output: pd.DataFrame) -> go.Figure:
    in_countries, in_pct = input.T.values
    out_countries, out_pct = output.T.values
    all_countries = in_countries + out_countries
    colors = px.colors.qualitative.Alphabet[: len(set(all_countries))]

    return go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_countries,
                    color=colors,
                ),
                link=dict(
                    source=[],
                    target=[],
                    value=[],
                    color=[],
                ),
            )
        ]
    )


def main():
    df = pd.read_csv(DATA_FILE)

    conf1997 = norm_country(df, 1997)
    conf2013 = norm_country(df, 2013)

    print(type(conf1997.iat[0, 1]))


if __name__ == "__main__":
    main()
