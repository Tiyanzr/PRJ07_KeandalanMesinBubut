# PRJ07 - Lathe reliability (MTBF/MTTR)

Concept Documentation : https://canva.link/3vgj0xuaq0f8eec

Source data: `..\Raw Data\Monitoring Mesin Bubut` (bronze = sensors, ref = logs). All paths are in `config.py`; the active data patch is `PATCH`.

## Notebooks
1. `01_ingest_data` - CSV -> verified bronze -> one timeline for both machines
2. `02_detection` - physical-state detection (A41-A44), run with `MACHINE = 4` then `3`

## Environment
```
conda env create -f environment.yml
conda activate pa-mtbf
```
