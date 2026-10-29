California non-COVID dataset for Epi-2-Net.

Files:
- `flu-publichealthlab-byregion-fluseason.csv`: official California influenza surveillance CSV.
- `build_california_influenza_dataset.py`: converts the official regional weekly influenza counts into the county-level JSON format expected by `main.py`.
- `InfluenzaLabInCalifornia.json`: generated dataset in the same payload style as `COVIDinCalifornia.json`.

Construction:
- Disease signal: `Influenza_Total` from California public health laboratory surveillance by week and region.
- Nodes: the existing 58 California counties already used by this project.
- Geography: existing county centroids from `dataset/California/california_regions.csv`.
- Mobility: existing county mobility graph from `dataset/California/COVIDinCalifornia.json`.
- Downscaling: each regional weekly influenza total is split across counties in that region by county population share.

Rebuild:

```powershell
python dataset\CaliforniaFlu\build_california_influenza_dataset.py
```

Run with `main.py`:

```powershell
python main.py --country California --root_path ./dataset/CaliforniaFlu/ --data_path InfluenzaLabInCalifornia.json --node_num 58
```
