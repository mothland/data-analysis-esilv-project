# Dashboard — EU Infrastructure Analysis

`app.py` is a Dash application with four tabs, one per analysis query, and it loads the real master dataset from `../data/master.csv` at startup. Each tab renders a Plotly figure and a markdown explanation produced by the corresponding query function in `queries.py`.

## Files

- `app.py` — the Dash application, loads data and lays out the four tabs
- `queries.py` — all four query functions, each returning `(fig, explanation_markdown)`

## How to run

```bash
cd dashboard
python app.py
```

Open `http://127.0.0.1:8050` in your browser. To produce the submission file, right-click the page and save as HTML, or use Ctrl+P to save as PDF.

## Query function contract

Every function in `queries.py` follows this signature so `app.py` never needs to change:

```python
def queryX_name(df, **kwargs) -> tuple[plotly.graph_objects.Figure, str]:
    # Input:  df — the master dataframe
    # Output: (fig, explanation_markdown)
    ...
    return fig, explanation
```

## What each query does

**Query 1 (grouping):** computes `infra_km / expenditure_mEUR` per NUTS2 region for a given year (default 2021) and shows the top 15 regions as a horizontal bar chart coloured by country.

**Query 2 (pattern mining):** converts each region into a basket of four low/mid/high labels (terrain, water proximity, budget, infrastructure density, averaged over 2015–2022) and uses Apriori to find the most frequent two-item combinations across all regions. Ranked by support.

**Query 3 (spatial clustering):** runs K-Means with four clusters on `elevation_std` and `dist_to_river_km` for 2021, then plots each region's infrastructure density against its geographic difficulty score, with bubble size proportional to population density.

**Query 4 (temporal):** aggregates to EU-wide yearly averages and totals, then overlays a three-year rolling mean on the infrastructure density line alongside bars for total public expenditure, on a dual-axis chart.

## Producing the submission file

The project requires `familyNameLeader_dashboard.html` or `.pdf`:

- **HTML**: with `app.py` running, right-click the page → "Save Page As" → Webpage, HTML only
- **PDF**: with `app.py` running, press Ctrl+P → Save as PDF (you may need to capture each tab separately and combine)
