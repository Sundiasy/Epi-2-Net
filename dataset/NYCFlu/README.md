Exact non-COVID dataset for Epi-2-Net using New York City county-level influenza data.

Files:
- `ny_influenza_county_weekly.csv`: official NYSDOH county-level weekly influenza source.
- `county_to_county_migration_2011_2015.xlsx`: official U.S. Census county-to-county movement source.
- `co-est2024-alldata.csv`: official U.S. Census county population source.
- `build_nyc_influenza_dataset.py`: converter for the project JSON format.
- `InfluenzaInNYC.json`: generated project-ready dataset.
- `nyc_regions.csv`: exact county coordinates for the 5 NYC counties.

Nodes:
- Bronx
- Kings
- New York
- Queens
- Richmond

Guarantees:
- Case counts are exact county-level weekly values from the source dataset.
- No county downscaling or population-based case splitting is used.
- Movement graph is built from official county-to-county migration flows among the same 5 counties.

Build:

```powershell
D:\conda\envs\torch\python.exe dataset\NYCFlu\build_nyc_influenza_dataset.py
```

Run:

```powershell
D:\conda\envs\torch\python.exe main.py --country NYC --root_path ./dataset/NYCFlu/ --data_path InfluenzaInNYC.json --node_num 5
```
