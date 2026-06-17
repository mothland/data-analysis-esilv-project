# Analysis of Macro Ecological and Economical Factors on Infrastructure Distribution in the EU

**Team:** Amin Rachid, Ricardo Zwein, Ahmat Rouchad, Chanez Khelifa, Chems Mitta

The core question this project tries to answer is whether physical geography (terrain roughness and proximity to major rivers) and public transport budgets can explain why some EU regions have dense road and rail networks while others, spending comparable amounts of money, have relatively little to show for it. The analysis runs across roughly 240 NUTS2 regions, using time series from Eurostat and two static geographic datasets, and produces four analyses that each approach the question from a different angle.

## Data sources

Five sources are combined, all joined on the NUTS2 regional code:

| Source | Variable | Coverage |
|---|---|---|
| Eurostat `tran_r_net` | Road and rail density (km / 1000 km²) | ~1990–2022 |
| Eurostat `tgs00024` | Population density (inhabitants / km²) | 1990–2023 |
| Eurostat `gov_10a_exp` | Public transport expenditure (COFOG GF04, mEUR) | 1995–2023 |
| Copernicus GLO-30 DEM (AWS S3) | Mean elevation and terrain roughness std (m) | static |
| HydroSHEDS HydroRIVERS | Distance from region centroid to nearest major river (km) | static |

The master dataset that comes out of all of this is 83,965 rows across roughly 240 regions and 35 years. And it is, honestly, quite patchy: infrastructure density is missing for 48.6% of rows (early years have almost no Eurostat coverage), population density for 69.7%, and expenditure for 15.7%. The analyses filter to years with decent coverage rather than trying to impute.

## Setup

```bash
pip install -r requirements.txt --prefer-binary
```

Two of the five data sources need a little more attention. The Copernicus GLO-30 terrain data streams from public AWS S3 via `rasterio` and `s3fs`, and the first run takes somewhere between 5 and 15 minutes depending on your connection, but it then caches to `./data/terrain_nuts2.csv`, which is already committed to this repo so you can skip that entirely. HydroRIVERS is the other one: it must be downloaded manually from [hydrosheds.org](https://www.hydrosheds.org/products/hydrorivers) and placed at `./data/HydroRIVERS_v10_eu_gdb.zip`. That file is too large for the repo, but the processed master dataset (`./data/master.csv`) is already committed, so if you are only running the analysis and dashboard you can skip both.

## Notebook

`notebook_skeleton.ipynb` runs the full pipeline and all four analyses. Sections 0 through 5 handle data loading, exploration, and joining. Section 6 is the analysis.

A note on the data loading code: it was written with the help of generative AI (prompts included verbatim in the notebook), and the terrain loading function in particular went through three iterations to fix a sea-pixel contamination issue, then a further parallelisation pass using `ThreadPoolExecutor`. The analysis queries were written by the team.

### Query 1: Infrastructure efficiency (grouping)

For a given year (default 2021), regions are grouped by `NUTS_ID` using `groupby().agg()`, and the ratio `infra_km / expenditure_mEUR` is computed per region. So what you get is, basically, a measure of how many kilometres of road and rail infrastructure exist per million EUR of public transport spending. A high value means a mature network relative to current investment; a low value could mean a region catching up, or one where terrain is pushing construction costs up, and you cannot tell which just from this number. The chart shows the top 15 regions, coloured by country.

### Query 2: Pattern mining (frequent itemsets)

Each region (averaged over 2015–2022, the years with the best coverage) gets turned into a basket of four labels: terrain, water proximity, budget, and infrastructure density, each split into low, mid, and high tertiles using `pd.qcut`. So a region becomes something like `terrain=mid, water=far, budget=low, infra=low`. The Apriori algorithm (`mlxtend`, `min_support=0.10`) then finds the combinations of those labels that appear together in a meaningful share of regions, and the result is ranked purely by support, which is the fraction of regions containing the combination. The two strongest pairs are `budget=low + infra=low` and `budget=mid + infra=mid`, both around 0.16 to 0.17 support, and the budget-to-infrastructure coupling is quite a bit tighter than any of the terrain or water combinations.

### Query 3: Spatial clustering

K-Means with four clusters is applied to `elevation_std` and `dist_to_river_km` (standardised) for 2021, and the resulting cluster labels are visualised on a choropleth map of Europe alongside a summary table of average infrastructure density, terrain roughness, and river proximity per cluster. One cluster is, more or less, defined by its outlier-level distance to major rivers (mean 4,427 km, which points squarely at the Scandinavian interior and similar remote areas), and the others sit on a gradient of terrain roughness against infrastructure density that matches what the research question would predict.

### Query 4: Temporal trend and moving average

The data is aggregated by year to produce an EU-wide average infrastructure density and a total public transport expenditure. A three-year rolling mean (`.rolling(window=3).mean()`) is then overlaid on the infrastructure density, so you get a smoothed trend alongside the fiscal cycle bars. And the interesting thing to look for here is the lag: peaks in expenditure do not show up immediately in the density line, because it takes time for budget commitments to become physical road and rail.

## Dashboard

The dashboard is a Dash application with four tabs, one per query, and it runs against the real master dataset. Each tab shows the Plotly figure for that query plus a markdown explanation.

```bash
cd dashboard
python app.py
```

Then open `http://127.0.0.1:8050`. To produce the submission file, right-click the page in your browser and save as HTML, or use Ctrl+P to save as PDF.

## Repository layout

```
notebook_skeleton.ipynb   full pipeline and analysis
requirements.txt          Python dependencies
data/
  master.csv              processed master dataset (83,965 rows)
  terrain_nuts2.csv       pre-computed elevation stats per NUTS2 region
  HydroRIVERS_v10_eu_gdb.zip   (not in repo, download manually)
dashboard/
  app.py                  Dash application
  queries.py              all four query functions
```
