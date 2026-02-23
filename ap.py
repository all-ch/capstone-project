import pandas as pd
import numpy as np
import glasbey as gb
import plotly.graph_objects as go
import plotly.colors as pc


# set yourself
DATA_FILE = "data/data.csv"


def norm_country(data: pd.DataFrame, year: int) -> pd.DataFrame:
    return (
        data[data.Conference_Year == year]
        .groupby("Organization_Country")["Person"]
        .nunique()
        .transform((lambda x: x * 100 / x.sum()))
        .reset_index(name="Percentage")
    )


def sub_match(
    input: pd.DataFrame, output: pd.DataFrame
) -> list[list[int] | list[np.float64]]:
    source = input.loc[
        input["Organization_Country"].isin(output["Organization_Country"])
    ]
    target = output.loc[
        output["Organization_Country"].isin(input["Organization_Country"])
    ]
    s_idx, t_idx = source.index.tolist(), target.index.tolist()
    value = []
    for i in range(len(source)):
        if (s := source.iat[i, 1]) <= (t := target.iat[i, 1]):
            input.at[s_idx[i], "Percentage"], output.at[t_idx[i], "Percentage"] = (
                0,
                t - s,
            )
            value.append(s)
        else:
            input.at[s_idx[i], "Percentage"], output.at[t_idx[i], "Percentage"] = (
                s - t,
                0,
            )
            value.append(s - t)
    return [s_idx, t_idx, value]


def calc_links(
    input: pd.DataFrame, output: pd.DataFrame
) -> list[list[int] | list[np.float64]]:
    output.index += len(input)
    source, target, value = sub_match(input, output)
    in_pct, out_pct = (
        input[input.Percentage != 0]["Percentage"],
        output[output.Percentage != 0]["Percentage"],
    )
    in_idx, out_idx, out_sum = (
        in_pct.index.tolist(),
        out_pct.index.tolist(),
        out_pct.sum(),
    )
    for i in range(len(in_pct)):
        for j in range(len(out_pct)):
            source.append(in_idx[i])
            target.append(out_idx[j])
            value.append(in_pct.iat[i] * out_pct.iat[j] / out_sum)
    return [source, target, value]


def map_colors(nvalues: list[str], lvalues: list[int]) -> list[list[str] | list[int]]:
    value_set = set(nvalues)
    node_color_map = dict(
        zip(value_set, gb.create_palette(palette_size=len(value_set)))
    )
    link_color_map = {
        k: f"rgba{pc.hex_to_rgb(v) + (0.5,)}" for k, v in node_color_map.items()
    }
    return [
        [node_color_map[v] for v in nvalues],
        [link_color_map[nvalues[i]] for i in lvalues],
    ]


def gen_ap(input: pd.DataFrame, output: pd.DataFrame) -> go.Figure:
    in_countries, in_pct = input.T.values
    out_countries, out_pct = output.T.values
    all_countries = in_countries.tolist() + out_countries.tolist()
    lsource, ltarget, lvalue = calc_links(input, output)
    ncolor, lcolor = map_colors(all_countries, ltarget)
    return go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_countries,
                    color=ncolor,
                ),
                link=dict(
                    source=lsource,
                    target=ltarget,
                    value=lvalue,
                    color=lcolor,
                ),
            )
        ]
    )


def main():
    df = pd.read_csv(DATA_FILE)

    conf1997 = norm_country(df, 1997)
    conf2013 = norm_country(df, 2013)

    ap = gen_ap(conf1997, conf2013)
    ap.show()


if __name__ == "__main__":
    main()
