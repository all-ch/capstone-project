import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

DATAFILE = "data/data.csv"
COLORS = px.colors.qualitative.Plotly
US = ["United States", "United States, Pakistan"]
EU = [
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


def gen_psap(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for i, (k, v) in enumerate(dfs.items()):
        fig.add_trace(
            go.Scatter(
                x=v["Conference_Year"].unique(),
                y=v["Total_Count"].tolist(),
                mode="lines",
                line=dict(width=0.5, color=COLORS[i]),
                stackgroup="one",
                groupnorm="percent",
                name=k,
            )
        )
    return fig


def main():
    df = pd.read_csv(DATAFILE)
    us = get_fcat(df, 1997, 2013, US)
    eu = get_fcat(df, 1997, 2013, EU)
    other = get_fcat(df, 1997, 2013, US + EU, False)
    fig = gen_psap(dict(zip(["United States", "Europe", "Other"], [us, eu, other])))
    fig.update_layout(
        showlegend=True,
        xaxis_type="category",
        yaxis=dict(type="linear", range=[1, 100], ticksuffix="%"),
        title_text=f"Speaker Composition by Region ({1997}-{2013})",
        legend_title_text="Regions",
        xaxis_title="Conference Year",
        yaxis_title="Speaker Composition (%)",
    )
    fig.show()


if __name__ == "__main__":
    main()
