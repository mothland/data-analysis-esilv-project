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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# ─────────────────────────────────────────────────────────────────
# Query 3 - Spatial [2pt]
# ─────────────────────────────────────────────────────────────────
def query3_spatial_clustering(df, year=2021, n_clusters=4):
    """
    Query 3 - Spatial [2pt]

    Input:  df          - master dataframe
            year        - analysed year
            n_clusters  - number of KMeans clusters

    Output: (fig, explanation)

    What it computes:
        Spatial clustering of NUTS2 regions according to
        geographic difficulty and infrastructure density.

        KMeans is applied using:
            - mean_elevation_m
            - elevation_std
            - dist_to_river_km
            - infra_km

        For visualisation purposes, a Geographic Difficulty Score
        is created from the three geographic variables and plotted
        against infrastructure density.
    """

    sub = df[df["year"] == year].copy()

    # Remove missing values
    sub = sub.dropna(
        subset=[
            "infra_km",
            "pop_density",
            "mean_elevation_m",
            "elevation_std",
            "dist_to_river_km"
        ]
    )

    # ── One row per region ────────────────────────────────────────
    grouped = sub.groupby(
        ["NUTS_ID", "NUTS_NAME", "country"],
        as_index=False
    ).agg(
        infra_km=("infra_km", "mean"),
        pop_density=("pop_density", "mean"),
        mean_elevation_m=("mean_elevation_m", "mean"),
        elevation_std=("elevation_std", "mean"),
        dist_to_river_km=("dist_to_river_km", "mean"),
    )

    # ──────────────────────────────────────────────────────────────
    # KMeans clustering on the ORIGINAL 4 variables
    # ──────────────────────────────────────────────────────────────
    cluster_features = [
        "mean_elevation_m",
        "elevation_std",
        "dist_to_river_km",
        "infra_km"
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(grouped[cluster_features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    grouped["cluster"] = kmeans.fit_predict(X)

    # ──────────────────────────────────────────────────────────────
    # Geographic Difficulty Score (visualisation only)
    # ──────────────────────────────────────────────────────────────
    geo_features = [
        "mean_elevation_m",
        "elevation_std",
        "dist_to_river_km"
    ]

    geo_scaled = scaler.fit_transform(grouped[geo_features])

    grouped["difficulty_score"] = geo_scaled.sum(axis=1)

    # ──────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────
    fig = px.scatter(
        grouped,
        x="difficulty_score",
        y="infra_km",
        color=grouped["cluster"].astype(str),
        hover_name="NUTS_ID",
        hover_data=[
            "country",
            "mean_elevation_m",
            "elevation_std",
            "dist_to_river_km"
        ],
        labels={
            "difficulty_score": "Geographic Difficulty Score",
            "infra_km": "Infrastructure Density",
            "color": "Cluster"
        },
        title=f"Infrastructure vs Geographic Difficulty ({year})"
    )

    fig.update_layout(height=550)

    explanation = f"""
**Indicator: spatial clustering of NUTS2 regions** (K-Means)

- **How it's calculated**: for {year}, regions are grouped by `NUTS_ID`
  and clustered using the K-Means algorithm. The clustering is based on
  four variables: `mean_elevation_m`, `elevation_std`,
  `dist_to_river_km` and `infra_km`.

- **What it represents**: the indicator identifies groups of regions
  sharing similar geographic characteristics and infrastructure levels.

- **Why it matters for this project**: the project's objective is to
  investigate whether geographic constraints help explain infrastructure
  distribution across EU regions. The clustering combines terrain-related
  variables, proximity to water and infrastructure density to identify
  common regional profiles.

- **Visualisation**: the x-axis shows a Geographic Difficulty Score,
  computed from standardised elevation, terrain roughness and distance
  to rivers. Higher values indicate regions facing stronger geographic
  constraints.

- **How to interpret it**: the clustering reveals distinct regional profiles.
  Regions in Cluster 0 combine relatively favourable geographic conditions
  with high infrastructure density, while Cluster 1 contains regions with
  similar geographic conditions but more moderate infrastructure levels.
  Cluster 2 groups regions facing greater geographic constraints and generally
  lower infrastructure density. Finally, Cluster 3 represents a small set of
  regions with very high geographic difficulty and very limited infrastructure.
  Overall, the results suggest that geographic constraints are associated with
  lower infrastructure development across EU regions.

- Bubble colours indicate the cluster assigned by the K-Means algorithm.
"""

    return fig, explanation


# ─────────────────────────────────────────────────────────────────
# Query 4 - Temporal [2pt]
# ─────────────────────────────────────────────────────────────────
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def query4_temporal_forecast(df, **kwargs):
    """
    Query 4 - Temporal [2pt]

    Input:  df    - master dataframe (columns: NUTS_ID, NUTS_NAME, country,
                    year, infra_km, expenditure_mEUR, ...)
            kwargs - optional parameters

    Output: (fig, explanation)
            fig         - dual-axis line/bar chart, Plotly Figure
            explanation - markdown string describing the indicator

    What it computes:
        "Temporal Trend and Moving Average"
        Aggregates the master dataset by year to compute the EU-wide average
        infrastructure density and total public expenditure. It then applies
        a 3-year moving average (lag feature) to the infrastructure density.

        Implementation detail: rows missing essential temporal data are dropped.
        Data is grouped by year using groupby().agg(). A rolling window of 3
        is used to calculate the moving average.
    """
    # Drop rows missing essential temporal data
    sub = df.dropna(subset=["year", "infra_km", "expenditure_mEUR"]).copy()

    # ── Grouping step: aggregate to EU-wide yearly totals and averages ──
    yearly_trend = sub.groupby("year", as_index=False).agg(
        avg_infra_km=("infra_km", "mean"),
        total_expenditure=("expenditure_mEUR", "sum"),
    )
    
    # Ensure chronological order
    yearly_trend = yearly_trend.sort_values("year")

    # Create the 3-year moving average (lag feature)
    yearly_trend["infra_moving_avg_3y"] = yearly_trend["avg_infra_km"].rolling(window=3).mean()

    # Build dual-axis figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for expenditure (secondary y-axis)
    fig.add_trace(
        go.Bar(
            x=yearly_trend["year"], 
            y=yearly_trend["total_expenditure"],
            name="Total Expenditure (mEUR)",
            opacity=0.5
        ),
        secondary_y=True,
    )

    # Line chart for raw average infrastructure density (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=yearly_trend["year"], 
            y=yearly_trend["avg_infra_km"],
            name="Avg Infra Density",
            mode="lines+markers",
            line=dict(width=2)
        ),
        secondary_y=False,
    )

    # Dashed line for the 3-year moving average (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=yearly_trend["year"], 
            y=yearly_trend["infra_moving_avg_3y"],
            name="3-Year Trend (Moving Avg)",
            mode="lines",
            line=dict(width=3, dash="dash")
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title="EU Temporal Trend: Infrastructure vs Public Expenditure",
        xaxis_title="Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    fig.update_yaxes(title_text="Infra Density (km/1000km²)", secondary_y=False)
    fig.update_yaxes(title_text="Total Expenditure (mEUR)", secondary_y=True)

    explanation = """
**Indicator: Temporal Trend and Moving Average** (`.rolling(window=3).mean()`)

- **How it's calculated**: the data is aggregated by `year`, calculating the EU-wide
  average infrastructure density (`infra_km`) and the total transport expenditure
  (`expenditure_mEUR`). We then apply a 3-year moving average to the infrastructure density.
- **What it represents**: it juxtaposes the volume of public money invested (bars) against
  the actual physical infrastructure available (lines) over time. The moving average acts
  as a lag feature to reveal the underlying long-term trend, smoothing out statistical noise.
- **How to interpret it**: the bars (right axis) show fiscal cycles. The solid line shows
  raw yearly data. The dashed line is the 3-year smoothed trend. A delay between high
  expenditure and a rising dashed line illustrates the time it takes for budgets to
  translate into actual roads and railways.

*Note: As permitted by the assignment instructions, a Large Language Model was used to format the Plotly dual-axis logic and structure the moving average computation.*
"""
    return fig, explanation
