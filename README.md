# Mizoram Landslide Risk Model

Random Forest landslide susceptibility and rainfall-trigger model for Mizoram,
India (Smart India Hackathon 2026, SIH26001). Everything lives in one file:
`mizoram_landslide_model.py`.

## Run it

```bash
pip install -r requirements.txt
python mizoram_landslide_model.py                      # build dataset + train
python mizoram_landslide_model.py --predict 23.7271 92.7176
```

The first run calls free weather APIs for ~130 points and takes several
minutes. Results are cached in `outputs/`, so an interrupted run resumes.

### GeoTIFF layers (optional, not in git)

Terrain, NDVI and land-cover features come from GeoTIFFs exported with Google
Earth Engine. They are too large for git (`*.tif` is ignored). Point the script
at them with an environment variable:

```bash
export MLEWS_RASTER_DIR=/path/to/tiffs        # Colab: your mounted Drive folder
```

Expected names: `mizoram_elevation.tif`, `mizoram_slope.tif`,
`mizoram_aspect.tif`, `mizoram_curvature.tif`, `mizoram_ndvi.tif`,
`mizoram_lulc.tif`, `mizoram_monsoon_rainfall_2024.tif`,
`mizoram_dist_from_water.tif`. Missing files are skipped. **Without the
elevation raster, generated negative points can land outside Mizoram**, so set
this before trusting any result.

### Options

| Flag | Effect |
|---|---|
| `--table PATH` | train from an existing table, skip the build |
| `--drop ndvi twi` | exclude features (ablation) |
| `--with-density` | include `landslide_density` (off by default, see below) |
| `--target-recall 0.95` | recall the alert threshold aims for |
| `--predict LAT LON [--date YYYY-MM-DD]` | score a location |

## Method notes

- **Spatial validation.** Points are grouped into ~5 km blocks and kept whole
  on one side of every train/test split and CV fold. Many inventory points are
  within a few hundred metres of each other, so a random split would test the
  model on places it has effectively seen.
- **Negatives.** Sampled at least 2.2 km from any known landslide and only where
  raster data exists (the bounding box also covers Bangladesh, Myanmar, Assam).
  They get random historical dates so rainfall features come from the same
  seasonal distribution as the positives.
- **Rainfall.** 3/7/15-day totals over the days *preceding* each event date.
- **Alert threshold** is chosen on an inner validation split, not the test set,
  and set below 0.5 because a missed landslide costs more than a false alarm.
- **`landslide_density` is off by default.** 29 of 52 inventory points are
  within 8 km of Aizawl, so this feature mostly detects "inside the reported
  cluster" rather than terrain. It is still computed and saved in the table.

## Limitations

- **Small inventory:** 52 points, many clustered around Aizawl. Expect high
  variance between CV folds.
- **Reporting bias:** landslides near roads and towns get recorded; remote ones
  often do not.
- **NDVI** is a recent composite, but many events are from 2024-25. A fresh
  scar reads as bare ground, so NDVI may encode the *consequence* of a slide.
  Compare `--drop ndvi`.
- **TWI** is a local slope-only approximation (no catchment area).
- **Soil moisture and rainfall forecast** are not used in training: the APIs
  only provide them for "now", not for historical event dates.
- **Geology and soil depth** are not included (no suitable free source found).
- Trained on Mizoram only; not calibrated elsewhere.

## Data sources

NASA COOLR; Sarma & Paul (2026), arXiv:2606.23281; May 2025 monsoon reports;
SRTM, Sentinel-2, ESA WorldCover, CHIRPS and JRC water via Google Earth Engine;
[Open-Meteo](https://open-meteo.com); [GEM Global Active Faults](https://github.com/GEMScienceTools/gem-global-active-faults).
