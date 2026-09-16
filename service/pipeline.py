import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


LOW_LIQUIDITY = {"BANE", "BANEP"}

CLASS_MAP = {0: "падение", 1: "боковик", 2: "рост"}


def load_artifact(artifact_dir):
    with open(f"{artifact_dir}/manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(f"{artifact_dir}/features.json", encoding="utf-8") as f:
        features = json.load(f)
    scaler = joblib.load(f"{artifact_dir}/scaler.pkl")
    models = [load_model(f"{artifact_dir}/clf_lstm_seed{s}.keras") for s in manifest["seeds"]]
    return manifest, features, scaler, models


def prepare_data(df, cfg):
    data = df.copy().sort_values(["ticker", "Date"])
    h = cfg["horizon"]
    data["target_return"] = data.groupby("ticker")["close"].shift(-h) / data["close"] - 1
    data["oil_in_rub"] = data["brent"] * data["usd_rub"]
    data["oil_in_rub"] = data.groupby("ticker")["oil_in_rub"].pct_change()
    for col in cfg["pct_change_cols"]:
        if col in data.columns:
            data[col] = data.groupby("ticker")[col].pct_change()
    data["stock_ret_1d"] = data.groupby("ticker")["close"].pct_change()
    data["rel_to_imoex"] = data["stock_ret_1d"] - data["imoex"]
    data["stock_ret_5d"] = data.groupby("ticker")["stock_ret_1d"].transform(lambda x: x.rolling(5).mean())
    data["mkt_vol_20"] = data.groupby("ticker")["imoex"].transform(lambda x: x.rolling(20).std())
    data["volatility_20"] = data.groupby("ticker")["stock_ret_1d"].transform(lambda x: x.rolling(20).std())
    data["rel_volume"] = data.groupby("ticker")["volume"].transform(lambda x: x / x.rolling(20).mean())
    if {"high", "low"}.issubset(data.columns):
        data["range_pct"] = (data["high"] - data["low"]) / data["close"]
    data = pd.get_dummies(data, columns=["ticker"], prefix="ticker", dtype=float)
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def create_sequences(frame, feature_cols, window_size, target_col="target_return"):
    Xs, ys, ds, ts = [], [], [], []
    for tcol in [c for c in frame.columns if c.startswith("ticker_")]:
        sub = frame[frame[tcol] == 1.0].sort_values("Date")
        if len(sub) <= window_size:
            continue
        feats = sub[feature_cols].values.astype("float32")
        tgt = sub[target_col].values.astype("float32")
        dates = sub["Date"].values
        name = tcol.replace("ticker_", "")
        for i in range(window_size, len(sub)):
            Xs.append(feats[i - window_size:i]); ys.append(tgt[i])
            ds.append(dates[i]); ts.append(name)
    return np.array(Xs), np.array(ys), np.array(ds), np.array(ts)


def apply_scaler(frame, features, scaler):
    binary = [c for c in features if c.startswith("ticker_")
              or c.endswith("is_changed") or c.endswith("is_new")]
    scale_cols = [c for c in features if c not in binary]
    f = frame.copy()
    f[scale_cols] = scaler.transform(f[scale_cols])
    f[features] = f[features].astype("float32")
    return f


def make_3_classes(y, thr):
    return np.select([y < -thr, y > thr], [0, 2], default=1).astype("int64")


def ensemble_proba(models, X):
    return np.mean([m.predict(X, verbose=0) for m in models], axis=0)


def build_dataset(csv_path, manifest, features, scaler):
    cfg = manifest["config"]
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].between(cfg["date_start"], cfg["date_end"])]
    df = df[~df["ticker"].isin(cfg["exclude_tickers"])]
    df = df.drop(columns=[c for c in cfg["drop_columns"] if c in df.columns])

    data = prepare_data(df, cfg)
    data = data.dropna(subset=features + ["target_return"])   # те же строки, что в обучении
    data = apply_scaler(data, features, scaler)

    X, y_ret, ds, ts = create_sequences(data, features, cfg["window_size"])
    ds = pd.to_datetime(ds).normalize()
    return X, y_ret, ds, ts


def decide(probs, manifest, ticker):
    class_map = {int(k): v for k, v in manifest.get("class_map", CLASS_MAP).items()} \
        if isinstance(manifest.get("class_map"), dict) else CLASS_MAP
    pred = int(np.argmax(probs))
    return {
        "pred_class": pred,
        "pred_label": class_map[pred],
        "lead_proba": float(probs[pred]),
        "low_liquidity": ticker in LOW_LIQUIDITY,
        "probs": {class_map[c]: float(probs[c]) for c in range(len(probs))},
    }


def predict_one(X, y_ret, ds, ts, ticker, date, manifest, models):
    target_date = pd.Timestamp(date).normalize()
    idx = np.where((ts == ticker) & (ds == target_date))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    probs = ensemble_proba(models, X[i:i + 1])[0]
    out = decide(probs, manifest, ticker)
    out["ticker"] = ticker
    out["date"] = target_date.date().isoformat()
    out["horizon"] = manifest["horizon"]
    true_ret = float(y_ret[i])
    true_cls = int(make_3_classes(np.array([true_ret]), manifest["class_threshold"])[0])
    class_map = {int(k): v for k, v in manifest.get("class_map", CLASS_MAP).items()}
    out["actual_return"] = true_ret
    out["actual_label"] = class_map[true_cls]
    out["hit"] = (out["pred_class"] == true_cls)
    return out


def available_dates(ds, ts, ticker):
    d = sorted(pd.unique(ds[ts == ticker]))
    return [pd.Timestamp(x).date().isoformat() for x in d]


def raw_context_for(csv_path, manifest, ticker, date):
    cfg = manifest["config"]
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].between(cfg["date_start"], cfg["date_end"])]
    df = df[~df["ticker"].isin(cfg["exclude_tickers"])]
    df = df.drop(columns=[c for c in cfg["drop_columns"] if c in df.columns])
    data = prepare_data(df, cfg)
    d = pd.Timestamp(date).normalize()
    row = data[(data.get("ticker_" + ticker, 0) == 1.0) & (data["Date"].dt.normalize() == d)]
    if row.empty:
        return {}
    row = row.iloc[0]
    out = {}
    if "volatility_20" in data.columns:
        out["волатильность 20д"] = f"{row['volatility_20'] * 100:.1f}%"
    if "stock_ret_5d" in data.columns:
        out["ср. доходность 5д"] = f"{row['stock_ret_5d'] * 100:+.2f}%"
    if "range_pct" in data.columns:
        out["дневной диапазон"] = f"{row['range_pct'] * 100:.1f}%"
    return out
