#!/usr/bin/env python
"""Mizoram landslide risk model (Random Forest) - single-file pipeline.

Builds a labelled dataset from a landslide inventory plus generated
negatives, adds terrain/land-cover features from GeoTIFFs and rainfall /
fault features from free APIs, trains a spatially-validated Random Forest,
and predicts risk for any coordinate.

Usage
-----
    python mizoram_landslide_model.py                      # build dataset + train
    python mizoram_landslide_model.py --table outputs/training_table.csv
    python mizoram_landslide_model.py --drop ndvi          # ablation
    python mizoram_landslide_model.py --predict 23.7271 92.7176
    python mizoram_landslide_model.py --predict 23.7271 92.7176 --date 2025-05-30

Environment variables
---------------------
    MLEWS_RASTER_DIR   folder holding the GEE GeoTIFFs
    MLEWS_OUT          output folder (default: ./outputs)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

log = logging.getLogger("mizoram")

# ============================================================================
# CONFIG
# ============================================================================
BBOX = (21.9, 24.5, 92.2, 93.5)          # lat_min, lat_max, lon_min, lon_max
UTM_CRS = "EPSG:32646"                   # metric CRS covering Mizoram
N_NEGATIVE = 80
MIN_SEPARATION_M = 2226.0                # negatives stay this far from any landslide
DENSITY_BANDWIDTH_M = 2000.0
BLOCK_DEG = 0.045                        # ~5 km blocks for spatial splitting
SEED = 42
NEG_DATE_START, NEG_DATE_END = "2007-01-01", "2025-12-31"
TARGET_RECALL = 0.90

IN_COLAB = importlib.util.find_spec("google.colab") is not None
_default_rasters = "/content/drive/MyDrive/Mizoram_Landslide_SIH" if IN_COLAB else "rasters"
RASTER_DIR = Path(os.environ.get("MLEWS_RASTER_DIR", _default_rasters))
OUT_DIR = Path(os.environ.get("MLEWS_OUT", "outputs"))

GEM_FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/"
    "master/geojson/gem_active_faults_harmonized.geojson"
)
OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# GeoTIFF filename -> feature column. Missing files are skipped.
RASTER_COLUMNS = {
    "mizoram_elevation.tif": "elevation",
    "mizoram_slope.tif": "slope",
    "mizoram_aspect.tif": "aspect",
    "mizoram_curvature.tif": "curvature",
    "mizoram_ndvi.tif": "ndvi",
    "mizoram_lulc.tif": "lulc",
    "mizoram_monsoon_rainfall_2024.tif": "rainfall_monsoon",
    "mizoram_dist_from_water.tif": "dist_from_water",
    "mizoram_geology.tif": "geology_class",
}
META_COLUMNS = {"latitude", "longitude", "label", "event_date", "source"}
# Computed and saved in the table, but NOT fed to the model unless --with-density.
# 29 of 52 inventory points sit within 8 km of Aizawl, so this feature mostly
# detects "inside the heavily-reported cluster" (inventory clustering), not terrain.
OFF_BY_DEFAULT = {"landslide_density"}

# Landslide inventory: NASA COOLR, Sarma & Paul 2026 (arXiv:2606.23281),
# and May 2025 monsoon reports. One exact-duplicate row removed.
INVENTORY_CSV = """latitude,longitude,event_date,source
23.69461107,92.87641207,2017-06-28,COOLR
23.73941422,92.71505961,2017-06-01,COOLR
22.533,92.899,2007-07-30,COOLR
23.591,92.9296,2010-09-13,COOLR
22.479,92.967,2007-07-30,COOLR
23.7327,92.7463,2014-07-14,COOLR
23.75406693,92.74128531,2010-09-13,COOLR
23.3572,92.7878,2015-08-26,COOLR
24.1293,92.6921,2007-09-29,COOLR
23.74182634,92.7141905,2010-09-13,COOLR
24.3069,93.125,2010-09-16,COOLR
23.8447,93.0514,2015-08-26,COOLR
23.7527,92.694,2010-09-16,COOLR
23.7131,92.7362,2015-06-19,COOLR
23.73478778,92.72124394,2009-10-05,COOLR
23.7313,92.7187,2015-09-14,COOLR
22.7176,92.8416,2010-09-13,COOLR
23.7281,92.7175,2013-05-11,COOLR
24.2315,92.6755,2013-05-07,COOLR
24.042,92.6724,2015-08-27,COOLR
23.5397,93.3783,2016-07-28,COOLR
23.73504471,92.71060943,2010-09-13,COOLR
23.3439,93.0748,2010-09-13,COOLR
23.7348,92.7187,2015-09-14,COOLR
23.7365,92.7146,2010-08-06,COOLR
23.7705,92.5217,2012-11-11,COOLR
23.72988966,92.71495623,2010-09-13,COOLR
23.4495,92.8112,2015-08-28,COOLR
23.7269,92.7265,2015-08-27,COOLR
22.90813548,92.47207294,2017-06-09,COOLR
23.65637182,93.02746602,2017-06-17,COOLR
23.33501164,92.85410625,2017-06-12,COOLR
24.0135808,92.67420963,2017-04-25,COOLR
23.725943,92.7178,2016-09-17,COOLR
23.72,92.72,2017-09-17,arXiv_2606.23281
23.77,92.74,2019-07-02,arXiv_2606.23281
23.73,92.73,2021-06-13,arXiv_2606.23281
23.70,92.72,2024-05-28,arXiv_2606.23281
23.74,92.72,2024-05-28,arXiv_2606.23281
23.68,92.73,2024-05-28,arXiv_2606.23281
23.72,92.71,2024-05-28,arXiv_2606.23281
23.73,92.72,2024-05-28,arXiv_2606.23281
23.77,92.74,2024-05-28,arXiv_2606.23281
23.74,92.71,2024-05-28,arXiv_2606.23281
23.75,92.73,2024-05-30,arXiv_2606.23281
23.76,92.74,2024-07-02,arXiv_2606.23281
23.76,92.73,2024-07-02,arXiv_2606.23281
24.03,92.62,2024-08-28,arXiv_2606.23281
23.75,92.73,2025-05-30,arXiv_2606.23281
23.28,93.42,2025-05-31,news_2025_monsoon
23.30,92.85,2025-05-31,news_2025_monsoon
22.53,92.90,2025-05-31,news_2025_monsoon
"""


# ============================================================================
# HELPERS
# ============================================================================
def request_with_retry(url: str, retries: int = 4, backoff: float = 2.0, timeout: int = 30):
    """GET with exponential backoff and jitter."""
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt) + random.uniform(0, 1))
    raise last  # type: ignore[misc]


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Works on scalars and numpy arrays."""
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_inventory() -> pd.DataFrame:
    df = pd.read_csv(StringIO(INVENTORY_CSV), parse_dates=["event_date"])
    df = df.drop_duplicates(subset=["latitude", "longitude", "event_date"]).reset_index(drop=True)
    df["label"] = 1
    return df


# ============================================================================
# RASTER FEATURES
# ============================================================================
def sample_raster(path: Path, lats, lons) -> np.ndarray:
    """Sample a GeoTIFF at WGS84 points. Out-of-bounds points return NaN."""
    import rasterio
    from rasterio.warp import transform as warp_transform

    xs, ys = [float(v) for v in lons], [float(v) for v in lats]
    with rasterio.open(path) as src:
        if src.crs is not None and src.crs.to_string() != "EPSG:4326":
            xs, ys = warp_transform("EPSG:4326", src.crs, xs, ys)
        b = src.bounds
        inside = [(b.left <= x <= b.right) and (b.bottom <= y <= b.top) for x, y in zip(xs, ys)]
        out = np.full(len(xs), np.nan)
        pts = [(x, y) for x, y, ok in zip(xs, ys, inside) if ok]
        if pts:
            vals = np.array([v[0] for v in src.sample(pts, indexes=1)], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            out[np.array(inside)] = vals
    return out


def attach_rasters(df: pd.DataFrame, raster_dir: Path | None = None) -> pd.DataFrame:
    """Add one column per available GeoTIFF.

    GEE exports clipped to the Mizoram polygon fill everything outside it with
    0. Elevation 0 is never real here, so it marks "outside the study area";
    every raster feature is blanked at those points.
    """
    raster_dir = Path(raster_dir or RASTER_DIR)
    out = df.copy()
    if not raster_dir.is_dir():
        log.warning("Raster folder %s not found - skipping raster features", raster_dir)
        return out
    if importlib.util.find_spec("rasterio") is None:
        log.warning("rasterio not installed - skipping raster features")
        return out

    for fname, col in RASTER_COLUMNS.items():
        path = raster_dir / fname
        if not path.exists():
            log.info("skip %s (not found)", fname)
            continue
        out[col] = sample_raster(path, out["latitude"], out["longitude"])
        log.info("%s: %d/%d sampled", col, int(out[col].notna().sum()), len(out))

    if "elevation" in out.columns:
        out.loc[out["elevation"] == 0, "elevation"] = np.nan
        outside = out["elevation"].isna()
        for col in set(RASTER_COLUMNS.values()) & set(out.columns):
            out.loc[outside, col] = np.nan
    return out


# ============================================================================
# API / VECTOR FEATURES
# ============================================================================
def fetch_twi(lat: float, lon: float, delta: float = 0.0015) -> float:
    """Approximate topographic wetness from a local 3x3 elevation grid.

    This uses only a local slope term - there is no upstream catchment area -
    so it is a weak proxy for a true TWI.
    """
    try:
        offs = [-delta, 0.0, delta]
        lats = [lat + dy for dy in offs for _ in offs]
        lons = [lon + dx for _ in offs for dx in offs]
        url = (f"{OPEN_METEO_ELEVATION}?latitude={','.join(map(str, lats))}"
               f"&longitude={','.join(map(str, lons))}")
        g = np.asarray(request_with_retry(url, retries=3).json()["elevation"], float).reshape(3, 3)
        cell = delta * 111_320.0
        dzdx = ((g[0, 2] + 2 * g[1, 2] + g[2, 2]) - (g[0, 0] + 2 * g[1, 0] + g[2, 0])) / (8 * cell)
        dzdy = ((g[2, 0] + 2 * g[2, 1] + g[2, 2]) - (g[0, 0] + 2 * g[0, 1] + g[0, 2])) / (8 * cell)
        return math.log(cell / max(math.hypot(dzdx, dzdy), 1e-3))
    except Exception as exc:  # noqa: BLE001
        log.warning("TWI failed at (%.4f, %.4f): %s", lat, lon, exc)
        return np.nan


def fetch_antecedent_rainfall(lat: float, lon: float, ref_date=None) -> dict:
    """Rainfall totals over the 3/7/15 days *preceding* the reference date.

    Historical points use the archive API anchored on their own event date.
    Live prediction (ref_date=None) uses the forecast API's recent past, since
    the archive lags real time by several days.
    """
    out = {"rainfall_3day": np.nan, "rainfall_7day": np.nan, "rainfall_15day": np.nan}
    try:
        if ref_date is None or pd.isna(ref_date):
            url = (f"{OPEN_METEO_FORECAST}?latitude={lat}&longitude={lon}"
                   "&daily=precipitation_sum&past_days=15&forecast_days=1&timezone=auto")
            daily = request_with_retry(url, retries=3).json()["daily"]["precipitation_sum"][:15]
        else:
            ref = pd.Timestamp(ref_date)
            start, end = ref - pd.Timedelta(days=15), ref - pd.Timedelta(days=1)
            url = (f"{OPEN_METEO_ARCHIVE}?latitude={lat}&longitude={lon}"
                   f"&start_date={start.date()}&end_date={end.date()}"
                   "&daily=precipitation_sum&timezone=auto")
            daily = request_with_retry(url, retries=3).json()["daily"]["precipitation_sum"]
        arr = np.asarray(daily, dtype=float)
        out["rainfall_3day"] = float(np.nansum(arr[-3:]))
        out["rainfall_7day"] = float(np.nansum(arr[-7:]))
        out["rainfall_15day"] = float(np.nansum(arr[-15:]))
    except Exception as exc:  # noqa: BLE001
        log.warning("rainfall failed at (%.4f, %.4f): %s", lat, lon, exc)
    return out


_FAULTS = None


def _load_faults():
    """GEM active faults, clipped to the region BEFORE reprojecting.

    Reprojecting worldwide geometry into one local UTM zone distorts shapes far
    from that zone and yields invalid distances.
    """
    global _FAULTS
    if _FAULTS is not None:
        return _FAULTS if len(_FAULTS) else None
    _FAULTS = []
    try:
        import geopandas as gpd
        from shapely.geometry import box

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "gem_faults.geojson"
        if not path.exists():
            path.write_bytes(request_with_retry(GEM_FAULTS_URL, retries=3, backoff=3).content)
        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        s, n, w, e = BBOX
        gdf = gdf[gdf.intersects(box(w - 3, s - 3, e + 3, n + 3))]
        log.info("Faults near study area: %d", len(gdf))
        _FAULTS = gdf.to_crs(UTM_CRS)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load fault database: %s", exc)
    return _FAULTS if len(_FAULTS) else None


def fetch_fault_distance(lat: float, lon: float) -> float:
    faults = _load_faults()
    if faults is None:
        return np.nan
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(UTM_CRS).geometry.iloc[0]
        d = faults.geometry.distance(pt).dropna()
        return float(d.min()) if len(d) else np.nan
    except Exception as exc:  # noqa: BLE001
        log.warning("fault distance failed: %s", exc)
        return np.nan


def landslide_density(lat: float, lon: float, inv_lat, inv_lon) -> float:
    """Gaussian-kernel density of known landslides around a point.

    Inventory points closer than MIN_SEPARATION_M are ignored, because negatives
    are generated at least that far from every landslide; counting closer
    neighbours would hand positives information negatives can never have.

    This removes the structural leak but NOT the clustering effect: positives
    concentrate around Aizawl, so density still flags that cluster. For that
    reason the feature is excluded from the model by default.
    """
    d = haversine_m(lat, lon, inv_lat, inv_lon)
    d = d[d > MIN_SEPARATION_M]
    return float(np.exp(-(d ** 2) / (2 * DENSITY_BANDWIDTH_M ** 2)).sum()) if d.size else 0.0


# ============================================================================
# DATASET
# ============================================================================
def generate_negatives(inventory: pd.DataFrame, n: int = N_NEGATIVE) -> pd.DataFrame:
    """Sample non-landslide points inside Mizoram.

    Rejects candidates (a) closer than MIN_SEPARATION_M to any landslide and
    (b) outside raster coverage. The bounding box is a rectangle but Mizoram is
    not; without (b) negatives land in Bangladesh/Myanmar/Assam, where rasters
    are empty, and "missing data" becomes a shortcut for "safe".
    """
    rng = np.random.default_rng(SEED)
    s, n_lat, w, e = BBOX
    inv_lat, inv_lon = inventory["latitude"].to_numpy(), inventory["longitude"].to_numpy()

    cand = []
    while len(cand) < n * 6:
        lat, lon = rng.uniform(s, n_lat), rng.uniform(w, e)
        if haversine_m(lat, lon, inv_lat, inv_lon).min() >= MIN_SEPARATION_M:
            cand.append((lat, lon))
    df = pd.DataFrame(cand, columns=["latitude", "longitude"])

    cov = attach_rasters(df)
    if "elevation" in cov.columns:
        keep = cov["elevation"].notna()
        log.info("Negatives with raster coverage: %d/%d", int(keep.sum()), len(df))
        df = df[keep.to_numpy()]
    else:
        log.warning("No elevation raster: negatives may fall outside Mizoram. "
                    "Set MLEWS_RASTER_DIR to enable coverage filtering.")

    df = df.head(n).reset_index(drop=True)
    if len(df) < n:
        log.warning("Only %d/%d negatives generated", len(df), n)

    # Random historical dates, so both classes draw rainfall from the same
    # seasonal distribution rather than negatives all sharing "today".
    start, end = pd.Timestamp(NEG_DATE_START), pd.Timestamp(NEG_DATE_END)
    days = np.random.default_rng(SEED + 1).integers(0, (end - start).days, size=len(df))
    df["event_date"] = [start + pd.Timedelta(days=int(d)) for d in days]
    df["label"] = 0
    df["source"] = "generated_negative"
    return df


def build_table(request_delay: float = 0.5) -> pd.DataFrame:
    """Assemble the labelled feature table (API results cached to disk)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = load_inventory()
    neg = generate_negatives(inv)
    pts = pd.concat([inv, neg], ignore_index=True)
    log.info("%d points: %d positive, %d negative", len(pts), len(inv), len(neg))

    cache_path = OUT_DIR / "api_cache.csv"
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    key = lambda lat, lon, d: f"{lat:.6f}|{lon:.6f}|{pd.Timestamp(d).date()}"  # noqa: E731
    cached = {}
    if len(cache):
        cached = {rec["_key"]: rec for rec in cache.to_dict("records") if "_key" in rec}
        log.info("Resuming with %d cached rows", len(cached))

    rows = []
    for i, r in enumerate(pts.itertuples(), 1):
        k = key(r.latitude, r.longitude, r.event_date)
        if k in cached:
            rows.append(cached[k])
            continue
        log.info("[%d/%d] fetching (%.4f, %.4f)", i, len(pts), r.latitude, r.longitude)
        row = {"_key": k, "twi": fetch_twi(r.latitude, r.longitude),
               "dist_from_faults": fetch_fault_distance(r.latitude, r.longitude)}
        row.update(fetch_antecedent_rainfall(r.latitude, r.longitude, r.event_date))
        rows.append(row)
        if len(rows) % 5 == 0:
            pd.DataFrame(rows).to_csv(cache_path, index=False)
        time.sleep(request_delay)
    pd.DataFrame(rows).to_csv(cache_path, index=False)

    api = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore").drop_duplicates("_key")
    pts["_key"] = [key(a, b, c) for a, b, c in zip(pts["latitude"], pts["longitude"], pts["event_date"])]
    table = pts[["_key", "latitude", "longitude", "event_date", "label", "source"]].merge(
        api, on="_key", how="left").drop(columns="_key")

    # Density is cheap and depends on the inventory, so it is never cached.
    il, io = inv["latitude"].to_numpy(), inv["longitude"].to_numpy()
    table["landslide_density"] = [landslide_density(a, b, il, io)
                                  for a, b in zip(table["latitude"], table["longitude"])]
    table = attach_rasters(table)

    table.to_csv(OUT_DIR / "training_table.csv", index=False)
    log.info("Saved %s (%d rows)", OUT_DIR / "training_table.csv", len(table))
    return table


# ============================================================================
# MODEL
# ============================================================================
def feature_columns(df: pd.DataFrame, drop=(), with_density: bool = False) -> list[str]:
    skip = set(drop) | (set() if with_density else OFF_BY_DEFAULT)
    cols = [c for c in df.columns if c not in META_COLUMNS and c not in skip]
    keep = [c for c in cols if df[c].nunique(dropna=True) > 1]
    if set(cols) - set(keep):
        log.info("Dropped constant/empty features: %s", sorted(set(cols) - set(keep)))
    return keep


def spatial_groups(df: pd.DataFrame) -> pd.Series:
    """~5 km block IDs. Points in a block never straddle a train/test split."""
    a = (df["latitude"] / BLOCK_DEG).round().astype(int).astype(str)
    b = (df["longitude"] / BLOCK_DEG).round().astype(int).astype(str)
    return a + "_" + b


def make_pipeline() -> Pipeline:
    """Imputer lives inside the pipeline so medians come from training folds only."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                      max_features="sqrt", class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)),
    ])


def choose_threshold(y_true, y_prob, target_recall: float = TARGET_RECALL) -> float:
    """Highest cutoff still reaching target_recall (a missed slide costs more
    than a false alarm). Call this on validation data, never the test set."""
    if int(np.sum(y_true)) == 0:
        return 0.5
    _, recall, thr = precision_recall_curve(y_true, y_prob)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thr[ok[np.argmax(thr[ok])]]) if ok.size else 0.5


def train_and_evaluate(table: pd.DataFrame, drop=(), target_recall=TARGET_RECALL, save=True,
                       with_density: bool = False) -> dict:
    cols = feature_columns(table, drop, with_density)
    X, y, groups = table[cols], table["label"].astype(int), spatial_groups(table)
    log.info("%d features | %d rows | %d spatial blocks", len(cols), len(table), groups.nunique())

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    log.info("train %d (%d pos) | test %d (%d pos)", len(Xtr), ytr.sum(), len(Xte), yte.sum())

    # Threshold is chosen on an inner validation split so the test set stays held out.
    fit, val = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED + 1)
                    .split(Xtr, ytr, groups.iloc[tr]))
    probe = make_pipeline().fit(Xtr.iloc[fit], ytr.iloc[fit])
    threshold = choose_threshold(ytr.iloc[val], probe.predict_proba(Xtr.iloc[val])[:, 1], target_recall)

    model = make_pipeline().fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    metrics = {"threshold": threshold, "features": cols}
    try:
        metrics["test_roc_auc"] = float(roc_auc_score(yte, prob))
    except ValueError:
        metrics["test_roc_auc"] = None
    metrics["report_default"] = classification_report(yte, (prob >= 0.5).astype(int),
                                                      zero_division=0, output_dict=True)
    metrics["report_tuned"] = classification_report(yte, (prob >= threshold).astype(int),
                                                    zero_division=0, output_dict=True)

    cv = cross_val_score(make_pipeline(), X, y, groups=groups, scoring="roc_auc",
                         cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED))
    metrics["cv_scores"], metrics["cv_mean"], metrics["cv_std"] = list(map(float, cv)), float(cv.mean()), float(cv.std())

    final = make_pipeline().fit(X, y)
    metrics["importances"] = dict(sorted(
        zip(cols, map(float, final.named_steps["rf"].feature_importances_)), key=lambda kv: -kv[1]))

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, OUT_DIR / "landslide_rf.joblib")
        (OUT_DIR / "model_meta.json").write_text(
            json.dumps({"features": cols, "threshold": threshold}, indent=2))
        log.info("Saved model to %s", OUT_DIR / "landslide_rf.joblib")
    return metrics


def print_metrics(m: dict) -> None:
    print(f"\nTest ROC-AUC (spatially held out): {m['test_roc_auc']}")
    print(f"Grouped 5-fold CV ROC-AUC: {m['cv_mean']:.3f} +/- {m['cv_std']:.3f}")
    print(f"Operating threshold (chosen on validation data): {m['threshold']:.3f}")
    for name, key in (("default 0.5", "report_default"), ("tuned", "report_tuned")):
        r = m[key].get("1", {})
        print(f"  [{name:>11}] landslide precision {r.get('precision', 0):.2f}  recall {r.get('recall', 0):.2f}")
    print("\nFeature importances:")
    for k, v in m["importances"].items():
        print(f"  {k:<20} {v:.4f}")


# ============================================================================
# PREDICTION
# ============================================================================
def predict_point(lat: float, lon: float, date=None) -> dict:
    """Risk assessment for one coordinate using the saved model."""
    model_path, meta_path = OUT_DIR / "landslide_rf.joblib", OUT_DIR / "model_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Train first.")
    model, meta = joblib.load(model_path), json.loads(meta_path.read_text())

    inv = load_inventory()
    row = {"latitude": lat, "longitude": lon, "twi": fetch_twi(lat, lon),
           "dist_from_faults": fetch_fault_distance(lat, lon),
           "landslide_density": landslide_density(
               lat, lon, inv["latitude"].to_numpy(), inv["longitude"].to_numpy())}
    row.update(fetch_antecedent_rainfall(lat, lon, date))
    frame = attach_rasters(pd.DataFrame([row]))
    for c in meta["features"]:
        if c not in frame.columns:
            frame[c] = np.nan
    p = float(model.predict_proba(frame[meta["features"]])[0, 1])
    band = "LOW" if p < 0.3 else "MODERATE" if p < 0.6 else "HIGH"
    return {"latitude": lat, "longitude": lon, "probability": round(p, 3),
            "threshold": meta["threshold"], "alert": p >= meta["threshold"], "risk_band": band}


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, help="train from an existing table, skipping the build")
    ap.add_argument("--drop", nargs="*", default=[], help="features to exclude (ablation)")
    ap.add_argument("--with-density", action="store_true",
                    help="include landslide_density (off by default: it mostly detects the Aizawl cluster)")
    ap.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    ap.add_argument("--predict", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--date", help="YYYY-MM-DD anchor for antecedent rainfall (with --predict)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.predict:
        print(json.dumps(predict_point(*args.predict, date=args.date), indent=2))
        return 0

    if IN_COLAB and not RASTER_DIR.is_dir():
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")

    table = pd.read_csv(args.table) if args.table else build_table()
    print_metrics(train_and_evaluate(table, drop=args.drop, target_recall=args.target_recall,
                                        with_density=args.with_density))
    return 0


if __name__ == "__main__":
    sys.exit(main())
