"""
app.py
─────────────────────────────────────────────────────────────────
Dashboard for: "Analysis of Macro Ecological and Economical factors
on Infrastructure Distribution in the EU" (NUTS2 regions)

Run:
    python app.py
Then open http://127.0.0.1:8050 in your browser.

To produce the required submission file (familyNameLeader_dashboard.html
or .pdf):
    - HTML: in the browser, right-click -> "Save Page As" -> Webpage, HTML only
    - PDF:  in the browser, Cmd/Ctrl+P -> Save as PDF

Switching from mock to real data:
    Once the team's pipeline produces the real `master` dataframe
    (see notebook_skeleton.ipynb), export it once with:
        master.to_csv("../data/master.csv", index=False)
    then change DATA_PATH below to "../data/master.csv". No other
    code changes are needed — column names already match.
"""

import pandas as pd
import dash
from dash import dcc, html

import queries

# ─── Config — fill in your team's info ─────────────────────────────
TEAM_MEMBERS = [
    "Amin Rachid",
    "Ricardo Zwein",
    "Ahmat Rouchad",
    "Chanez Khelifa",
    "Chems Mitta"
]
DATASET_NAME = "EU Infrastructure vs Geography & Public Expenditure (NUTS2)"

# Use the mock dataset until the real pipeline produces master.csv
DATA_PATH = "../data/master.csv"

# ─── Load data ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# ─── Run each query once at startup ──────────────────────────────────
fig1, exp1 = queries.query1_infrastructure_efficiency(df)
fig2, exp2 = queries.query2_pattern_mining(df)
fig3, exp3 = queries.query3_spatial_clustering(df)
fig4, exp4 = queries.query4_temporal_forecast(df)

# ─── App layout ───────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = DATASET_NAME

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "margin": "24px", "maxWidth": "1100px"},
    children=[
        html.H1(DATASET_NAME),
        html.P(
            f"Team members: {', '.join(TEAM_MEMBERS)}",
            style={"color": "#555", "fontStyle": "italic"},
        ),
        html.Hr(),

        dcc.Tabs([
            dcc.Tab(label="1. Infrastructure efficiency (grouping)", children=[
                html.Div(style={"padding": "16px"}, children=[
                    dcc.Graph(figure=fig1),
                    dcc.Markdown(exp1),
                ]),
            ]),
            dcc.Tab(label="2. Pattern mining", children=[
                html.Div(style={"padding": "16px"}, children=[
                    dcc.Graph(figure=fig2),
                    dcc.Markdown(exp2),
                ]),
            ]),
            dcc.Tab(label="3. Spatial clustering", children=[
                html.Div(style={"padding": "16px"}, children=[
                    dcc.Graph(figure=fig3),
                    dcc.Markdown(exp3),
                ]),
            ]),
            dcc.Tab(label="4. Temporal forecast", children=[
                html.Div(style={"padding": "16px"}, children=[
                    dcc.Graph(figure=fig4),
                    dcc.Markdown(exp4),
                ]),
            ]),
        ]),
    ],
)


if __name__ == "__main__":
    app.run(debug=True)
