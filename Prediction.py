# %%
#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------
# Bloom Prediction per-tile (LSTM + XGBoost Ensemble)
# ------------------------------------------------------------
# This notebook fetches data from Google Earth Engine per-tile,
# prepares time-series, trains LSTM + XGBoost ensemble models,
# and exports 1-year forecasts as JSON.
# ------------------------------------------------------------

# CELL 1: Authenticate Google Earth Engine
# ------------------------------------------------------------
import os
import ee

try:
    ee.Initialize()
    print("✅ Earth Engine already initialized.")
except Exception:
    print("🌍 Authenticating Google Earth Engine...")
    try:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize(project='kaksham-nasa-space-app')
        print("✅ Earth Engine initialized successfully.")
    except Exception as e2:
        print("❌ Earth Engine initialization failed:", e2)


# %%

# ------------------------------------------------------------
# CELL 2: Imports
# ------------------------------------------------------------
import json
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import geemap
from shapely.geometry import box
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

import pandas as pd
import numpy as np
import ee
import warnings
warnings.filterwarnings('ignore')




# ------------------------------------------------------------
# CELL 3: Region and parameters
# ------------------------------------------------------------
lat_min, lon_min, lat_max, lon_max = -14.457555, 130.988109, -14.006022, 131.036692
REGION = [lon_min, lat_min, lon_max, lat_max]
START_DATE = '2019-01-01'
END_DATE = datetime.date.today().isoformat()
FORECAST_YEARS = 1
TILE_SIZE_DEG = 0.05
BLOOM_PROB_THRESHOLD = 0.6
BLOOM_MODERATE_LOW = 0.3
OUTPUT_JSON = '/content/bloom_predictions_per_tile.json'

# ------------------------------------------------------------
# CELL 4: Create tile grid
# ------------------------------------------------------------
def create_tiles(lon_min, lat_min, lon_max, lat_max, tile_size_deg=TILE_SIZE_DEG):
    tiles = []
    ix = 0
    lon = lon_min
    while lon < lon_max:
        jx = 0
        lat = lat_min
        while lat < lat_max:
            b = box(lon, lat, min(lon+tile_size_deg, lon_max), min(lat+tile_size_deg, lat_max))
            tiles.append({
                'tile_id': f'tile_{ix}_{jx}',
                'geometry': b,
                'lon_min': lon,
                'lat_min': lat,
                'lon_max': min(lon+tile_size_deg, lon_max),
                'lat_max': min(lat+tile_size_deg, lat_max)
            })
            lat += tile_size_deg
            jx += 1
        lon += tile_size_deg
        ix += 1
    return gpd.GeoDataFrame(tiles)

lon_min, lat_min, lon_max, lat_max = REGION
tiles_gdf = create_tiles(lon_min, lat_min, lon_max, lat_max)
print(f"✅ Created {len(tiles_gdf)} tiles")


# %%




# ------------------------------------------------------------
# CELL 5: Safe EE data extraction helpers
# ------------------------------------------------------------
def ee_time_series_to_df(collection_id, band, geom, start_date, end_date, scale=1000):
    try:
        col = ee.ImageCollection(collection_id).select(band).filterDate(start_date, end_date).filterBounds(geom)
        def img_to_feature(img):
            stat = img.reduceRegion(ee.Reducer.mean(), geom, scale=scale)
            return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), band: stat.get(band)})
        fc = col.map(img_to_feature)
        info = fc.getInfo()
        if not info or 'features' not in info or len(info['features']) == 0:
            print(f"⚠️ No data for {band} in {collection_id}")
            return pd.DataFrame(columns=['date', band])
        features = info['features']
        rows = [{'date': f['properties']['date'], band: f['properties'].get(band, None)} for f in features]
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').dropna()
        return df
    except Exception as e:
        print(f"❌ Error fetching {band} from {collection_id}: {e}")
        return pd.DataFrame(columns=['date', band])


# ------------------------------------------------------------
# CELL 6: Per-tile extraction (fully corrected)
# ------------------------------------------------------------
def extract_features_for_tile(tile_row, start_date=START_DATE, end_date=END_DATE, scale=1000):
    

    geom = ee.Geometry.Rectangle([tile_row.lon_min, tile_row.lat_min, tile_row.lon_max, tile_row.lat_max])
    dfs = []

    # MODIS vegetation indices
    dfs.append(
        ee_time_series_to_df('MODIS/061/MOD13A1', 'NDVI', geom, start_date, end_date, 250)
        .rename(columns={'NDVI': 'ndvi'})
    )
    dfs.append(
        ee_time_series_to_df('MODIS/061/MOD13A1', 'EVI', geom, start_date, end_date, 250)
        .rename(columns={'EVI': 'evi'})
    )

    # MODIS land surface temperature
    dfs.append(
        ee_time_series_to_df('MODIS/061/MOD11A1', 'LST_Day_1km', geom, start_date, end_date, 1000)
        .rename(columns={'LST_Day_1km': 'lst_day'})
    )

    # MODIS surface reflectance band 1
    dfs.append(
        ee_time_series_to_df('MODIS/061/MOD09GA', 'sur_refl_b01', geom, start_date, end_date, 500)
    )

    # CHIRPS precipitation
    dfs.append(
        ee_time_series_to_df('UCSB-CHG/CHIRPS/PENTAD', 'precipitation', geom, start_date, end_date, 5000)
        .rename(columns={'precipitation': 'precip'})
    )

    # ECMWF ERA5-Land wind components
    df_u = ee_time_series_to_df('ECMWF/ERA5_LAND/DAILY_AGGR', 'u_component_of_wind_10m', geom, start_date, end_date)
    df_v = ee_time_series_to_df('ECMWF/ERA5_LAND/DAILY_AGGR', 'v_component_of_wind_10m', geom, start_date, end_date)
    if not df_u.empty and not df_v.empty:
        df_uv = pd.merge(df_u, df_v, on='date', how='outer')
        df_uv['wind_speed'] = np.sqrt(df_uv.iloc[:, 1]**2 + df_uv.iloc[:, 2]**2)
        dfs.append(df_uv[['date', 'wind_speed']])

    # Merge all valid dataframes
    valid_dfs = [d for d in dfs if not d.empty]
    if len(valid_dfs) == 0:
        print(f"⚠️ No valid data found for {tile_row.tile_id}")
        return pd.DataFrame()

    df = valid_dfs[0]
    for d in valid_dfs[1:]:
        df = pd.merge(df, d, on='date', how='outer')

    # Interpolate and fill missing values using datetime index
    df = df.sort_values('date')
    df = df.set_index('date')
    df = df.interpolate(method='time').ffill().bfill().dropna()
    df = df.reset_index()  # Restore 'date' as a column

    return df


# %%
# ------------------------------------------------------------
# CELL 7: Extract for all tiles
# ------------------------------------------------------------


# Create a folder to store CSVs
output_dir = "tiles_data"
os.makedirs(output_dir, exist_ok=True)

all_tiles_data = {}
for idx, row in tiles_gdf.iterrows():
    print(f"🚀 Processing {row.tile_id}...")
    df_tile = extract_features_for_tile(row)
    if df_tile.empty:
        print(f"❌ No data extracted for {row.tile_id}")
        continue
    all_tiles_data[row.tile_id] = df_tile
    # Save CSV in the output folder
    df_tile.to_csv(f"{output_dir}/{row.tile_id}_timeseries.csv", index=False)

print(f"✅ Extracted data for {len(all_tiles_data)} tiles.")


# %%

# ------------------------------------------------------------
# CELL 8: Feature engineering & labeling
# ------------------------------------------------------------
def prepare_features(df, lookback=12):
    df = df.copy().set_index('date').resample('M').mean()
    df['ndvi_roll_mean_3'] = df.get('ndvi', pd.Series()).rolling(3).mean()
    df['evi_roll_mean_3'] = df.get('evi', pd.Series()).rolling(3).mean()
    df['precip_roll_sum_3'] = df.get('precip', pd.Series()).rolling(3).sum()
    for lag in range(1, lookback+1):
        for col in df.columns:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df.dropna()

def create_labels_from_df(df_monthly):
    labels = pd.Series(0, index=df_monthly.index)
    if 'ndvi' in df_monthly.columns:
        monthwise_q = df_monthly['ndvi'].groupby(df_monthly.index.month).quantile(0.9)
        for d in df_monthly.index:
            if df_monthly.loc[d,'ndvi'] > monthwise_q.loc[d.month]:
                labels.loc[d] = 1
    elif 'evi' in df_monthly.columns:
        monthwise_q = df_monthly['evi'].groupby(df_monthly.index.month).quantile(0.9)
        for d in df_monthly.index:
            if df_monthly.loc[d,'evi'] > monthwise_q.loc[d.month]:
                labels.loc[d] = 1
    return labels

# ------------------------------------------------------------
# CELL 9: Model builders
# ------------------------------------------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_xgb():
    return XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=4, learning_rate=0.05)


# %%

# ------------------------------------------------------------
# CELL 10: Train & forecast
# ------------------------------------------------------------
results = {}
for tile_id, df in all_tiles_data.items():
    print(f"📘 Training for {tile_id} ...")
    df_feat = prepare_features(df, lookback=6)
    if df_feat.shape[0] < 36:
        print(f"⏩ Not enough data for {tile_id}, skipping")
        continue
    labels = create_labels_from_df(df_feat)
    df_feat['label'] = labels
    df_feat = df_feat.dropna()
    X = df_feat.drop(columns=['label'])
    y = df_feat['label']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    xgb = build_xgb()
    xgb.fit(X_train, y_train)
    xgb_pred_proba = xgb.predict(X_test)
    timesteps = 3
    def to_lstm_input(X_arr, timesteps=timesteps):
        Xs = []
        for i in range(timesteps, X_arr.shape[0]):
            Xs.append(X_arr[i-timesteps:i, :])
        return np.array(Xs)
    Xs = to_lstm_input(X_scaled)
    ys = y.values[timesteps:]
    if Xs.shape[0] < 10:
        print(f"⚠️ Not enough points for LSTM on {tile_id}")
        lstm_proba = np.zeros_like(y_test)
    else:
        split_idx = int(0.8 * Xs.shape[0])
        Xs_train, Xs_test = Xs[:split_idx], Xs[split_idx:]
        ys_train, ys_test = ys[:split_idx], ys[split_idx:]
        lstm = build_lstm_model((Xs_train.shape[1], Xs_train.shape[2]))
        lstm.fit(Xs_train, ys_train, epochs=20, batch_size=8, verbose=0)
        lstm_proba = lstm.predict(Xs_test).flatten()
    min_len = min(len(xgb_pred_proba), len(lstm_proba)) if len(lstm_proba)>0 else len(xgb_pred_proba)
    if min_len == 0:
        ensemble_proba = xgb_pred_proba
    else:
        ensemble_proba = (xgb_pred_proba[-min_len:] + lstm_proba[-min_len:]) / 2.0
    last_known = X_scaled[-timesteps:,:] if X_scaled.shape[0] >= timesteps else X_scaled
    forecast_steps = FORECAST_YEARS * 12
    forecasts = []
    current_window = last_known.copy()
    for step in range(forecast_steps):
        xgb_feat = current_window[-1].reshape(1, -1)
        p_xgb = xgb.predict(xgb_feat)[0]
        try:
            p_lstm = lstm.predict(current_window.reshape(1, current_window.shape[0], current_window.shape[1]))[0,0]
        except Exception:
            p_lstm = p_xgb
        p_ens = float((p_xgb + p_lstm)/2.0)
        severity = 'high' if p_ens > BLOOM_PROB_THRESHOLD else ('moderate' if p_ens >= BLOOM_MODERATE_LOW else 'low')
        flag = 1 if p_ens > BLOOM_PROB_THRESHOLD else 0
        last_date = df_feat.index.max()
        forecast_date = (last_date + pd.DateOffset(months=step+1)).strftime('%Y-%m-%d')
        forecasts.append({'date': forecast_date, 'probability': p_ens, 'severity': severity, 'flag': flag})
        next_row = current_window[-1].copy()
        current_window = np.vstack([current_window[1:], next_row]) if current_window.shape[0] > 1 else np.vstack([current_window, next_row])
    results[tile_id] = {
        'tile_id': tile_id,
        'lon_min': float(tiles_gdf.loc[tiles_gdf['tile_id']==tile_id,'lon_min'].values[0]),
        'lat_min': float(tiles_gdf.loc[tiles_gdf['tile_id']==tile_id,'lat_min'].values[0]),
        'forecasts': forecasts
    }


# %%

# ------------------------------------------------------------
# CELL 11: Save output JSON
# ------------------------------------------------------------
with open("bloom_predictions_per_tile.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved forecasts to {OUTPUT_JSON}")
print("Notebook run complete.")


# %%
# ------------------------------------------------------------
# CELL 12: Predict NDVI and EVI for next year for each tile
# ------------------------------------------------------------
print("🔄 Generating NDVI and EVI predictions for next year...")

ndvi_evi_results = {}

for tile_id, df in all_tiles_data.items():
    if tile_id not in results:
        continue  # skip tiles that weren't modeled

    df_feat = prepare_features(df, lookback=6)
    if df_feat.empty or 'ndvi' not in df_feat.columns or 'evi' not in df_feat.columns:
        print(f"⚠️ Skipping {tile_id} (missing NDVI/EVI data).")
        continue

    # Use the most recent 12 months as input for NDVI and EVI trend prediction
    df_recent = df_feat.iloc[-12:].copy()
    if df_recent.empty:
        continue

    # Fit a simple XGB model for NDVI and EVI trends
    xgb_ndvi = build_xgb()
    xgb_evi = build_xgb()

    X_ndvi = np.arange(len(df_recent)).reshape(-1, 1)
    y_ndvi = df_recent['ndvi'].values
    y_evi = df_recent['evi'].values

    xgb_ndvi.fit(X_ndvi, y_ndvi)
    xgb_evi.fit(X_ndvi, y_evi)

    # Predict 12 months ahead (next year)
    future_X = np.arange(len(df_recent), len(df_recent) + 12).reshape(-1, 1)
    ndvi_pred = xgb_ndvi.predict(future_X)
    evi_pred = xgb_evi.predict(future_X)

    ndvi_next_year = float(np.mean(ndvi_pred))
    evi_next_year = float(np.mean(evi_pred))

    # Store in dictionary
    ndvi_evi_results[tile_id] = {
        "predicted_ndvi_next_year": ndvi_next_year,
        "predicted_evi_next_year": evi_next_year
    }

    # Update main results dict
    results[tile_id].update(ndvi_evi_results[tile_id])

# Print and save
print("✅ Added NDVI and EVI forecasts to each tile:")
for t, vals in ndvi_evi_results.items():
    print(f"  {t}: NDVI={vals['predicted_ndvi_next_year']:.3f}, EVI={vals['predicted_evi_next_year']:.3f}")

with open("bloom_predictions_per_tile_with_ndvi_evi.json", "w") as f:
    json.dump(results, f, indent=2)

print("💾 Saved updated predictions with NDVI and EVI.")


# %%



