"""
generate_mock_data.py
─────────────────────────────────────────────────────────────────
Generates a synthetic 'master' dataframe with the SAME structure as
the real one produced by build_master_df() in notebook_skeleton.ipynb,
using the real NUTS2 region codes from data/terrain_nuts2.csv.

This lets us build and test the dashboard + Query 1 RIGHT NOW, before
the full data pipeline (Eurostat API + Copernicus DEM + HydroRIVERS)
runs successfully.

Output columns (matches build_master_df):
  NUTS_ID | NUTS_NAME | country | year | infra_km | pop_density |
  expenditure_mEUR | mean_elevation_m | elevation_std | dist_to_river_km

Run:  python generate_mock_data.py
Output: ../data/master_mock.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ─── Load real NUTS2 terrain data (334 regions) ────────────────────
terrain = pd.read_csv("../data/terrain_nuts2.csv")

YEARS = list(range(2015, 2022))  # 2015–2021, matches Eurostat coverage

# ─── Synthetic country-level budget baseline ───────────────────────
country_codes = sorted(terrain["NUTS_ID"].str[:2].unique())
budget_base = {c: np.random.uniform(200, 4000) for c in country_codes}

rows = []
for _, region in terrain.iterrows():
    nuts_id = region["NUTS_ID"]
    country = nuts_id[:2]
    elevation = region["mean_elevation_m"]
    roughness = region["elevation_std"]

    # Harder terrain (higher roughness) -> historically less infra built
    base_infra = max(50, 1200 - roughness * 1.2 + np.random.normal(0, 150))
    base_pop = np.random.uniform(20, 1500)

    for i, year in enumerate(YEARS):
        infra_km = max(0, base_infra * (1 + 0.01 * i) + np.random.normal(0, 30))
        pop_density = max(1, base_pop * (1 + 0.005 * i) + np.random.normal(0, 10))
        expenditure = max(1, budget_base[country] * (1 + 0.03 * i) + np.random.normal(0, 50))

        rows.append({
            "NUTS_ID": nuts_id,
            "NUTS_NAME": nuts_id,  # placeholder — real pipeline fills proper names
            "country": country,
            "year": year,
            "infra_km": round(infra_km, 2),
            "pop_density": round(pop_density, 2),
            "expenditure_mEUR": round(expenditure, 2),
            "mean_elevation_m": round(elevation, 2),
            "elevation_std": round(roughness, 2),
            "dist_to_river_km": round(np.random.uniform(0, 150), 2),
        })

master_mock = pd.DataFrame(rows)
master_mock.to_csv("../data/master_mock.csv", index=False)

print(f"Mock master dataframe created: {master_mock.shape[0]} rows x {master_mock.shape[1]} cols")
print(f"Regions: {master_mock['NUTS_ID'].nunique()}, Years: {YEARS[0]}-{YEARS[-1]}")
print(master_mock.head())
