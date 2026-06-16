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
# Query 2 - Pattern Mining [2pt]
# ─────────────────────────────────────────────────────────────────
def query2_pattern_mining(df, year_min=2015, year_max=2022, min_support=0.10, top_n=12):
    """
    Query 2 - Pattern mining [2pt]
 
    Input:  df           - master dataframe (NUTS_ID, year, infra_km,
                           expenditure_mEUR, mean_elevation_m, dist_to_river_km, ...)
            year_min/max - period used to average each region (default 2015-2022,
                           the years with the best coverage)
            min_support  - minimum support for an itemset to be "frequent"
            top_n        - how many frequent combinations to display
 
    Output: (fig, explanation)
            fig         - Plotly horizontal bar chart of the most frequent
                          combinations of region characteristics
            explanation - markdown string (required by the rubric)
 
    What it computes (frequent itemset mining, the basic form of pattern mining):
        Each NUTS2 region becomes a "basket" of items by cutting terrain,
        water, budget and infra into low/mid/high tertiles. Apriori then finds
        which COMBINATIONS of those labels appear together in many regions.
        The only metric is SUPPORT = the fraction of regions that contain the
        combination. We show the most frequent 2-item combinations.
    """
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori
 
    # ── One row per region: average over the well-covered recent years ──
    cols = ["mean_elevation_m", "dist_to_river_km", "expenditure_mEUR", "infra_km"]
    reg = (
        df[df["year"].between(year_min, year_max)]
        .groupby("NUTS_ID")[cols].mean().dropna().reset_index()
    )
 
    # ── Turn each continuous variable into a low / mid / high label (= an item) ──
    def buckets(s, name):
        return pd.qcut(s, 3, labels=[f"{name}=low", f"{name}=mid", f"{name}=high"])
 
    reg["terrain"] = buckets(reg["mean_elevation_m"], "terrain")  # elevation
    reg["budget"]  = buckets(reg["expenditure_mEUR"], "budget")   # public transport spending
    reg["infra"]   = buckets(reg["infra_km"], "infra")            # road/rail density
    reg["water"]   = pd.qcut(reg["dist_to_river_km"], 3,          # distance to nearest river
                             labels=["water=near", "water=mid", "water=far"])
 
    # ── Build the basket matrix and mine frequent itemsets with Apriori ──
    baskets = reg[["terrain", "water", "budget", "infra"]].astype(str).values.tolist()
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit_transform(baskets), columns=te.columns_)
    itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
 
    # ── Keep the real "patterns": combinations of at least 2 items ──
    itemsets["size"] = itemsets["itemsets"].apply(len)
    itemsets["combination"] = itemsets["itemsets"].apply(lambda s: " + ".join(sorted(s)))
    top = (itemsets[itemsets["size"] >= 2]
           .sort_values("support", ascending=False)
           .head(top_n))
 
    # ── Visualise: the most frequent combinations, ranked by support ──
    fig = px.bar(
        top.sort_values("support"),
        x="support",
        y="combination",
        orientation="h",
        color="support",
        color_continuous_scale="Blues",
        labels={"support": "Support (share of regions)", "combination": "Frequent combination"},
        title="Query 2 — Most frequent combinations of region characteristics",
    )
    fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
 
    explanation = f"""
**Indicator: frequent combinations of region characteristics (frequent itemsets)**
 
- **How it's calculated**: each NUTS2 region (averaged over {year_min}-{year_max})
  is described by four labels — terrain, water, budget and infrastructure — each
  cut into **low / mid / high tertiles**. A region is then a *basket* of items,
  e.g. `terrain=mid, water=far, budget=low, infra=low`. The **Apriori** algorithm
  (`min_support={min_support}`) finds the combinations of labels that appear
  together in many regions. We keep combinations of at least two labels.
- **What the metric means**: the only metric is **support** = the share of
  regions that contain the combination. A support of 0.17 means 17% of all
  regions have that exact pair of characteristics.
- **How to interpret it**: the top bars are the dominant profiles in Europe.
  Combinations like *budget=low + infra=low* or *budget=high + infra=high* show
  that public spending and infrastructure density tend to go together, while
  *infra=low + terrain=mid* shows that harder terrain coincides with sparser
  networks — which is exactly the relationship the project investigates.
"""
    return fig, explanation


# ─────────────────────────────────────────────────────────────────
# Query 3 - Spatial
# ─────────────────────────────────────────────────────────────────

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def query3_spatial_clustering(df, year=2021, n_clusters=4):
    """
    Query 3 - Spatial [2pt]

    Input:  df          - master dataframe
            year        - analysed year
            n_clusters  - number of KMeans clusters

    Output: (fig, explanation)

    What it computes:
        Spatial clustering of NUTS2 regions according to
        their geographic characteristics.

        KMeans is applied using:
            - elevation_std
            - dist_to_river_km

        Infrastructure density is analysed afterwards.

        Bubble size represents population density.
    """

    sub = df[df["year"] == year].copy()

    # Remove missing values
    sub = sub.dropna(
        subset=[
            "infra_km",
            "pop_density",
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
        elevation_std=("elevation_std", "mean"),
        dist_to_river_km=("dist_to_river_km", "mean"),
    )

    # ──────────────────────────────────────────────────────────────
    # KMeans clustering based ONLY on geography
    # ──────────────────────────────────────────────────────────────
    cluster_features = [
        "elevation_std",
        "dist_to_river_km"
    ]

    scaler = StandardScaler()

    X = scaler.fit_transform(
        grouped[cluster_features]
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    grouped["cluster"] = kmeans.fit_predict(X)

    # ──────────────────────────────────────────────────────────────
    # Geographic Difficulty Score
    # ──────────────────────────────────────────────────────────────
    grouped["difficulty_score"] = X.sum(axis=1)

    # ──────────────────────────────────────────────────────────────
    # Population scaling for bubble size
    # ──────────────────────────────────────────────────────────────
    grouped["pop_density_vis"] = np.sqrt(
        grouped["pop_density"]
    )

    # ──────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────
    fig = px.scatter(
        grouped,
        x="difficulty_score",
        y="infra_km",
        color=grouped["cluster"].astype(str),
        size="pop_density_vis",
        size_max=30,
        hover_name="NUTS_ID",
        hover_data=[
            "country",
            "pop_density",
            "elevation_std",
            "dist_to_river_km"
        ],
        labels={
            "difficulty_score": "Geographic Difficulty Score",
            "infra_km": "Infrastructure Density",
            "pop_density": "Population Density",
            "color": "Cluster"
        },
        title=f"Infrastructure vs Geographic Difficulty ({year})"
    )

    fig.update_traces(
        marker=dict(
            sizemin=4
        )
    )

    fig.update_layout(height=550)

    explanation = f"""
**Indicator: spatial clustering of NUTS2 regions** (K-Means)

- **How it's calculated**: for {year}, regions are grouped by `NUTS_ID`
  and clustered using the K-Means algorithm. The clustering is based on
  two geographic variables: `elevation_std` (terrain roughness) and
  `dist_to_river_km` (distance to major rivers).

- **What it represents**: the indicator identifies groups of regions
  sharing similar geographic conditions.

- **Why it matters for this project**: the objective is to investigate
  whether geographic constraints help explain infrastructure distribution
  across EU regions.

- **Visualisation**: the x-axis shows a Geographic Difficulty Score
  derived from the standardised geographic variables. Higher values
  indicate stronger geographic constraints.

- **How to interpret it**: regions with higher Geographic Difficulty
  Scores generally tend to exhibit lower infrastructure density.
  However, the relationship is not systematic. The size of the bubbles
  suggests that population density may also influence infrastructure
  development, as some densely populated regions maintain relatively
  high infrastructure levels despite geographic constraints.

- Colours indicate the cluster assigned by the K-Means algorithm.

- Bubble size represents population density (`pop_density`).

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
