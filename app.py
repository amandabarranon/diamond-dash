from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype

from dash import Dash, html, dcc, dash_table, Input, Output


base_path = Path(__file__).resolve().parent
csv_path = base_path / "diamonds.csv"

if csv_path.exists():
    my_data = pd.read_csv(csv_path)
else:
    try:
        import seaborn as sns
        my_data = sns.load_dataset("diamonds")
    except Exception:
        my_data = px.data.tips()

expected_cols = ["price", "carat", "cut", "color", "clarity"]
available_cols = [c for c in expected_cols if c in my_data.columns]

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


app = Dash(__name__, title="Diamond Data")

app.layout = html.Div(
    [
        html.H1("Diamond Data", style={"margin": "12px 0"}),
        html.H3("By Amanda Barrañón", style={"textAlign": "center", "color": "gray"}),



        html.Div(
            [
                # Sidebar
                html.Div(
                    [
                        html.H3("Controls"),
                        dcc.Dropdown(
                            id="col_select",
                            options=[{"label": c.capitalize(), "value": c} for c in available_cols],
                            value=available_cols[0] if available_cols else None,
                            clearable=False,
                            style={"marginBottom": "12px"},
                        ),
                        dcc.Input(
                            id="n_subset",
                            type="number",
                            value=100,  # un tamaño razonable para que el histo salga bien
                            min=1,
                            max=len(my_data),
                            step=1,
                            style={"width": "200px", "marginBottom": "12px"},
                        ),
                        html.Div(
                            [
                                html.B("Variable Descriptions:"), html.Br(),
                                html.Code("price"), ": price in US dollars", html.Br(),
                                html.Code("carat"), ": weight of the diamond", html.Br(),
                                html.Code("cut"), ": quality of the cut (Fair, Good, Very Good, Premium, Ideal)", html.Br(),
                                html.Code("color"), ": diamond color, from J (worst) to D (best)", html.Br(),
                                html.Code("clarity"), ": how clear the diamond is (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)",
                            ],
                            style={"fontSize": "13px", "color": "#555", "marginTop": "8px"},
                        ),
                    ],
                    style={"width": "280px", "flex": "0 0 auto", "paddingRight": "16px"},
                ),


                html.Div(
                    [
                        html.H3("Data Preview"),
                        dash_table.DataTable(
                            id="data_table",
                            columns=[{"name": c, "id": c} for c in my_data.columns],
                            data=my_data.head(10).to_dict("records"),
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "whiteSpace": "nowrap",
                                "textOverflow": "ellipsis",
                                "maxWidth": 180,
                                "fontSize": "13px",
                            },
                        ),
                        html.Hr(),
                        html.H3("Visualization"),
                        dcc.Graph(
                            id="distPlot",
                            style={"width": "100%", "height": "480px", "border": "1px solid #eee"},
                        ),
                    ],
                    style={"flex": "1 1 auto", "minWidth": "0"},  # evita colapso de ancho
                ),
            ],
            style={"display": "flex", "alignItems": "flex-start"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
)


def subset_df(n: int) -> pd.DataFrame:
    try:
        n = int(n)
    except Exception:
        n = 100
    if n <= 0:
        n = 100
    return my_data.head(n)


def ordered_categories(col: str):
    if col == "cut":
        return CUT_ORDER
    if col == "color":
        return COLOR_ORDER
    if col == "clarity":
        return CLARITY_ORDER
    return None

@app.callback(
    Output("data_table", "data"),
    Output("data_table", "columns"),
    Input("n_subset", "value"),
)
def update_table(n):
    df_head = subset_df(n)
    cols = [{"name": c, "id": c} for c in df_head.columns]
    return df_head.to_dict("records"), cols


@app.callback(
    Output("distPlot", "figure"),
    Input("col_select", "value"),
    Input("n_subset", "value"),
)
def update_plot(col, n):

    fig = go.Figure()
    fig.update_layout(height=480, margin=dict(l=40, r=20, t=50, b=40))

    if not col or col not in my_data.columns:
        fig.update_layout(title="Choose a valid column")
        return fig

    df = subset_df(n).copy()
    s = df[col]

    if is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x) <= 1:
            fig.update_layout(title=f"No hay suficientes datos numéricos en '{col}'")
            return fig
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=x,
                    nbinsx=min(60, max(10, len(x) // 10)),
                    marker_line_width=0.5,
                    marker_line_color="black",
                )
            ]
        )
        fig.update_layout(
            title=f"Histogram of {col}",
            title_x=0.5,
            bargap=0.05,
            height=480,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig.update_xaxes(title=col)
        fig.update_yaxes(title="Count")
        return fig


    counts = (
        df[col].astype("string").dropna().value_counts().rename_axis(col).reset_index(name="count")
    )
    order = ordered_categories(col)
    if order:
        counts[col] = pd.Categorical(counts[col], categories=order, ordered=True)
        counts = counts.sort_values(col)

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts[col],
                y=counts["count"],
                marker_line_width=0.5,
                marker_line_color="black",
            )
        ]
    )
    fig.update_layout(
        title=f"Bar Chart of {col}",
        title_x=0.5,
        bargap=0.2,
        height=480,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title=col)
    fig.update_yaxes(title="Count")
    return fig


if __name__ == "__main__":

    app.run(debug=False)
