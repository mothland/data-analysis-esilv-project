"""
queries.py
─────────────────────────────────────────────────────────────────
All analysis query functions for the dashboard.

CONTRACT — every query function must follow this signature:

    def queryX_name(df, **kwargs) -> tuple[plotly.graph_objects.Figure, str]

    Input:  df       - the master dataframe (see notebook_skeleton.ipynb,
                        build_master_df) with columns:
                        NUTS_ID, NUTS_NAME, country, year, infra_km,
                        pop_density, expenditure_mEUR, mean_elevation_m,
                        elevation_std, dist_to_river_km
            **kwargs - optional parameters specific to the query
    Output: (fig, explanation)
            fig         - a Plotly Figure, ready to pass to dcc.Graph
            explanation - a markdown string explaining what the indicator
                           means, how it's calculated, and how to read it
                           (required by the project rubric)

Teammates implement query2/3/4 following this exact contract so app.py
never needs to change — it just calls each function and renders the result.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────
# Query 1 - Grouping [1pt]
# ─────────────────────────────────────────────────────────────────
def query1_infrastructure_efficiency(df, year=2021, top_n=15):
    """
    Query 1 - Grouping [1pt]

    Input:  df    - master dataframe (columns: NUTS_ID, NUTS_NAME, country,
                     year, infra_km, expenditure_mEUR, ...)
            year  - int, which year to analyse (default: 2021, most recent)
            top_n - int, how many regions to show (default: 15)

    Output: (fig, explanation)
            fig         - horizontal bar chart, Plotly Figure
            explanation - markdown string describing the indicator

    What it computes:
        "Infrastructure efficiency" = infra_km / expenditure_mEUR
        i.e. how many km of road/rail infrastructure exist per million EUR
        of public transport expenditure, for each NUTS2 region in `year`.

        Implementation detail: rows are grouped by (NUTS_ID, NUTS_NAME,
        country) using groupby().agg(), then ranked by this ratio and the
        top_n regions are kept.
    """
    sub = df[df["year"] == year].copy()

    # ── Grouping step: one row per region, averaging in case of duplicates ──
    grouped = sub.groupby(["NUTS_ID", "NUTS_NAME", "country"], as_index=False).agg(
        infra_km=("infra_km", "mean"),
        expenditure_mEUR=("expenditure_mEUR", "mean"),
    )

    # Avoid division by zero / negative spending
    grouped = grouped[grouped["expenditure_mEUR"] > 0]
    grouped["efficiency"] = grouped["infra_km"] / grouped["expenditure_mEUR"]

    top = grouped.sort_values("efficiency", ascending=False).head(top_n)

    fig = px.bar(
        top.sort_values("efficiency"),
        x="efficiency",
        y="NUTS_ID",
        orientation="h",
        color="country",
        labels={
            "efficiency": "km of infrastructure per million EUR spent",
            "NUTS_ID": "NUTS2 region",
            "country": "Country",
        },
        title=f"Top {top_n} NUTS2 regions by infrastructure efficiency ({year})",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)

    explanation = f"""
**Indicator: infrastructure efficiency** (`infra_km / expenditure_mEUR`)

- **How it's calculated**: for {year}, regions are grouped by `NUTS_ID`
  (one row per region) and we divide their road/rail infrastructure
  density (`infra_km`) by their public transport expenditure
  (`expenditure_mEUR`).
- **What it represents**: how many kilometres of infrastructure a region
  "gets" per million EUR of public spending.
- **How to interpret it**: a high value means the region already has an
  extensive network relative to current spending (mature network, lower
  marginal investment needed). A low value can indicate a region that is
  investing heavily relative to its existing network — e.g. catching up,
  or facing higher construction costs due to difficult terrain.
- This chart shows the **top {top_n} regions** by this ratio, coloured by
  country.
"""
    return fig, explanation


# ─────────────────────────────────────────────────────────────────
# Query 2 - Pattern Mining [2pt]  — TODO: teammate implements this
# ─────────────────────────────────────────────────────────────────
def query2_pattern_mining(df):
    """
    Query 2 - Pattern mining [2pt]  (placeholder — to be implemented)

    Input:  df - master dataframe
    Output: (fig, explanation) — see module docstring for contract

    Suggested approach (from notebook_skeleton.ipynb):
        Frequent itemsets of (terrain_bucket, water_bucket, budget_bucket)
        -> infrastructure level clusters, using mlxtend.frequent_patterns
        (Apriori / FP-Growth).
    """
    fig = go.Figure()
    fig.update_layout(
        title="Query 2 - Pattern mining (to be implemented)",
        annotations=[dict(
            text="Placeholder — teammate implements this query",
            x=0.5, y=0.5, showarrow=False, font=dict(size=18),
        )],
        height=400,
    )
    explanation = "_Pattern mining query — to be implemented by the team (2pt)._"
    return fig, explanation


# ─────────────────────────────────────────────────────────────────
# Query 3 - Spatial [2pt]  — TODO: teammate implements this
# ─────────────────────────────────────────────────────────────────
def query3_spatial_clustering(df):
    """
    Query 3 - Spatial [2pt]  (placeholder — to be implemented)

    Input:  df - master dataframe (has lat/lon via geometry centroids)
    Output: (fig, explanation) — see module docstring for contract

    Suggested approach (from notebook_skeleton.ipynb):
        Spatial clustering (DBSCAN or KMeans) on geography to find
        over/under-infrastructure regions relative to geographic difficulty.
    """
    fig = go.Figure()
    fig.update_layout(
        title="Query 3 - Spatial clustering (to be implemented)",
        annotations=[dict(
            text="Placeholder — teammate implements this query",
            x=0.5, y=0.5, showarrow=False, font=dict(size=18),
        )],
        height=400,
    )
    explanation = "_Spatial clustering query — to be implemented by the team (2pt)._"
    return fig, explanation


# ─────────────────────────────────────────────────────────────────
# Query 4 - Temporal [2pt]  — TODO: teammate implements this
# ─────────────────────────────────────────────────────────────────
def query4_temporal_forecast(df):
    """
    Query 4 - Temporal [2pt]  (placeholder — to be implemented)

    Input:  df - master dataframe (has `year` column, 2015-2021)
    Output: (fig, explanation) — see module docstring for contract

    Suggested approach (from notebook_skeleton.ipynb):
        Time series forecasting of infra_km using lag features
        (infra_t-1, expenditure_t-1, pop_density_t-1), e.g. with
        statsmodels or scikit-learn.
    """
    fig = go.Figure()
    fig.update_layout(
        title="Query 4 - Temporal forecast (to be implemented)",
        annotations=[dict(
            text="Placeholder — teammate implements this query",
            x=0.5, y=0.5, showarrow=False, font=dict(size=18),
        )],
        height=400,
    )
    explanation = "_Temporal forecasting query — to be implemented by the team (2pt)._"
    return fig, explanation
