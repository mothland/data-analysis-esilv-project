# Dashboard — EU Infrastructure Analysis

## Files

- `app.py` — main Dash application (4 tabs, one per query)
- `queries.py` — all query functions. **Query 1 (grouping) is fully
  implemented.** Queries 2-4 are placeholders following a shared contract
  for your teammates to fill in.
- `generate_mock_data.py` — generates `../data/master_mock.csv`, a
  synthetic but realistic dataset (334 real NUTS2 codes x 7 years) so the
  dashboard works *before* the real pipeline finishes running.

## How to run

```bash
cd dashboard
pip install dash plotly pandas
python app.py
```

Open http://127.0.0.1:8050

## Query function contract (for teammates)

Every query function must look like this:

```python
def queryX_something(df, **kwargs) -> tuple[plotly.graph_objects.Figure, str]:
    """
    Input:  df - the master dataframe
    Output: (fig, explanation_markdown)
    """
    ...
    return fig, explanation
```

`app.py` calls each function once at startup and renders the figure +
markdown explanation in its own tab. **No changes to `app.py` are needed**
when a teammate fills in `query2_pattern_mining`, `query3_spatial_clustering`,
or `query4_temporal_forecast` in `queries.py` — just implement the function
body and return a real figure + explanation.

## Switching from mock data to the real pipeline output

Once `notebook_skeleton.ipynb` runs successfully and produces the `master`
GeoDataFrame, export it once:

```python
master.drop(columns="geometry").to_csv("data/master.csv", index=False)
```

Then in `app.py`, change:

```python
DATA_PATH = "../data/master_mock.csv"
```
to:
```python
DATA_PATH = "../data/master.csv"
```

Column names already match — no other code changes needed.

## Producing the submission file

The project requires `familyNameLeader_dashboard.html` or `.pdf`:

- **HTML**: with `app.py` running, in the browser right-click → "Save Page
  As" → Webpage, HTML only
- **PDF**: with `app.py` running, in the browser press Cmd/Ctrl+P → Save as
  PDF (make sure all 4 tabs' content is visible/captured — you may need to
  take one screenshot per tab and combine into a single PDF)

## Note on Query 1 (grouping)

**Indicator**: `efficiency = infra_km / expenditure_mEUR` — kilometres of
road/rail infrastructure per million EUR of public transport expenditure,
per NUTS2 region, for a given year (default 2021). Shows the top 15 regions
as a horizontal bar chart, coloured by country. Full explanation is in the
dashboard itself and in the function docstring in `queries.py`.
