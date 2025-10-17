# run_us_eod_liquidity_report.py
# -*- coding: utf-8 -*-
"""
EOD（前日終値）×上位流動性N銘柄の一括取得→分析→A〜F出力
- 取得戦略（無料系で1000+件を安定回収）:
  yfinance バルク（小チャンク/多段リトライ/再走査） → Stooq（補助） → Tiingo(不足) → AlphaVantage(不足)
- 取得できた銘柄の中から ADV20（出来高×終値）上位で N を確定
- 確定N銘柄に対して coverage=100% を担保（未取得銘柄は集合から外す）
- すべてEODベース（リアルタイム/時間外は未取得）
- ユニバースは --universe_source （既定 "yf"）で構築。FORCE_UNIVERSE_SOURCE で上書き可。
"""

import os, sys, time, argparse, math, random, re
from datetime import datetime as dt, timedelta, timezone
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

# ---------- 基本設定 ----------
DEFAULT_OUTDIR = "./out"
DEFAULT_N = 4000
CAL_DAYS_BACK = 200
SPY_SYMBOL = "SPY"
JST = timezone(timedelta(hours=9))

# ---------- ユーティリティ ----------
def jst_now_str() -> str:
    return dt.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def robust_get(url: str, params=None, headers=None, timeout=25, retries=3, backoff=1.6):
    ua = headers or {"User-Agent":"Mozilla/5.0"}
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=ua, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r
        except Exception:
            pass
        time.sleep(backoff * (i + 1))
    return None

def normalize_price_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    colmap = {
        "Date":"date","date":"date",
        "Open":"open","open":"open",
        "High":"high","high":"high",
        "Low":"low","low":"low",
        "Close":"close","close":"close",
        "Adj Close":"adjClose","adjClose":"adjClose","adjclose":"adjClose",
        "Volume":"volume","volume":"volume"
    }
    df = df.rename(columns={c: colmap.get(str(c), c) for c in df.columns})
    need = {"date","open","high","low","close","volume"}
    if not need.issubset(set(df.columns)):
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for c in ["open","high","low","close","volume","adjClose"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close","high","low","volume"])
    if df.empty:
        return None
    return df.sort_values("date").reset_index(drop=True)

# ---------- ユニバース ----------
def hydrate_universe_from_yf(keys: List[str]) -> pd.DataFrame:
    rows = []
    try:
        if any(k in ("sp500","s&p500") for k in keys):
            for t in yf.tickers_sp500():
                rows.append((t, "", "sp500"))
    except Exception:
        pass
    try:
        if any(k in ("dow30","dow") for k in keys):
            for t in yf.tickers_dow():
                rows.append((t, "", "dow30"))
    except Exception:
        pass
    try:
        if any(k in ("nasdaq","nasdaq_all") for k in keys):
            for t in yf.tickers_nasdaq():
                rows.append((t, "", "nasdaq_all"))
    except Exception:
        pass
    if any(k in ("nasdaq100","ndx") for k in keys):
        extra = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","PEP","COST","ADBE",
                 "NFLX","AMD","INTC","CSCO","QCOM","TXN","AMAT","INTU","LIN","BKNG","ADI","PYPL",
                 "SBUX","HON","MDLZ","VRTX","MU","REGN","LRCX","ABNB","MRVL","PANW"]
        rows.extend([(t,"","nasdaq100_seed") for t in extra])

    if not rows:
        seed = ["AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","BRK-B","UNH","JPM","XOM","V","PG","MA"]
        rows = [(t, "", "seed_top_liquidity") for t in seed]

    df = pd.DataFrame(rows, columns=["ticker","name","list_source"]).drop_duplicates("ticker")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)

def hydrate_universe_from_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "ticker" in df.columns
    if "name" not in df.columns:
        df["name"] = ""
    if "exchange" not in df.columns:
        df["exchange"] = ""
    # ETF/権利/ユニットっぽいものを一応弾く（YAML側でも除外済みだが二重化）
    pat = re.compile(r"\b(Warrant|Rt|Right|Units?)\b", re.IGNORECASE)
    df = df[~df.get("name","").astype(str).str.contains(pat)]
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df[["ticker","name","exchange"]].drop_duplicates("ticker").reset_index(drop=True)

def build_universe(keys: List[str], source_order: List[str]) -> pd.DataFrame:
    last_err = None
    for src in source_order:
        try:
            if src == "yf":
                return hydrate_universe_from_yf(keys)
            elif src.startswith("file:"):
                path = src.split("file:",1)[1]
                return hydrate_universe_from_file(path)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Universe build failed. Last error: {last_err}")

# ---------- Stooq（補助） ----------
def to_stooq_symbol(ticker: str) -> str:
    t = ticker.upper().replace("/", "-").replace("^","").replace(".", "-")
    return t.lower() + ".us"

def stooq_daily(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    sym = to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = robust_get(url, timeout=20)
    if not r or not r.text or r.text.strip().lower().startswith("error"):
        return None
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return None
    df = normalize_price_df(df)
    if df is None:
        return None
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    df = df.loc[mask].reset_index(drop=True)
    return df if not df.empty else None

# ---------- yfinance（多段バルク＋個別リトライ） ----------
def _yf_bulk_once(tickers: List[str], start: str, end_plus: str, chunk_size: int, sleep_s: float, threads: bool) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        batch = [t for t in tickers[i:i+chunk_size] if t != SPY_SYMBOL]
        if not batch:
            continue
        try:
            raw = yf.download(
                tickers=batch, start=start, end=end_plus, interval="1d",
                group_by="ticker", auto_adjust=False, threads=threads, progress=False
            )
            if raw is None or raw.empty:
                time.sleep(sleep_s); continue
            if isinstance(raw.columns, pd.MultiIndex):
                top = set(raw.columns.get_level_values(0))
                for t in batch:
                    if t in top:
                        sub = raw[t].reset_index()
                        df = normalize_price_df(sub)
                        if df is not None:
                            out[t] = df
            else:
                df = normalize_price_df(raw.reset_index())
                if df is not None and len(batch) == 1:
                    out[batch[0]] = df
        except Exception:
            pass
        time.sleep(sleep_s + random.random()*0.6)
    return out

def yf_bulk_multi_pass(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    end_plus = (pd.to_datetime(end) + timedelta(days=1)).strftime("%Y-%m-%d")

    remain = list(dict.fromkeys([t for t in tickers if t != SPY_SYMBOL]))
    if not remain:
        return out

    got = _yf_bulk_once(remain, start, end_plus, chunk_size=60, sleep_s=1.6, threads=False)
    out.update(got)
    remain = [t for t in remain if t not in out]

    if remain:
        got = _yf_bulk_once(remain, start, end_plus, chunk_size=40, sleep_s=2.2, threads=False)
        out.update(got)
        remain = [t for t in remain if t not in out]

    for t in tqdm(remain, desc="yfinance single (final fill)", ncols=90):
        try:
            df = yf.download(t, start=start, end=end_plus, interval="1d", auto_adjust=False, progress=False, threads=False)
            df = normalize_price_df(df.reset_index()) if df is not None and not df.empty else None
            if df is not None and not df.empty:
                out[t] = df
        except Exception:
            pass
        time.sleep(0.6 + random.random()*0.4)

    return out

# ---------- Tiingo / AlphaVantage ----------
def tiingo_daily(ticker: str, token: Optional[str], start: str, end: str) -> Optional[pd.DataFrame]:
    if not token:
        return None
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": start, "endDate": end, "format": "json", "token": token}
    r = robust_get(url, params=params, timeout=25)
    if not r:
        return None
    try:
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        return normalize_price_df(df)
    except Exception:
        return None

def alphav_daily_adj(ticker: str, apikey: Optional[str], start: str, end: str) -> Optional[pd.DataFrame]:
    if not apikey:
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":ticker,"outputsize":"full","datatype":"json","apikey":apikey}
    r = robust_get(url, params=params, timeout=45)
    if not r:
        return None
    try:
        js = r.json()
        ts = js.get("Time Series (Daily)", {})
        if not ts:
            return None
        rows = []
        for d, v in ts.items():
            rows.append({
                "date": pd.to_datetime(d),
                "open": float(v.get("1. open", "nan")),
                "high": float(v.get("2. high", "nan")),
                "low": float(v.get("3. low", "nan")),
                "close": float(v.get("4. close", "nan")),
                "adjClose": float(v.get("5. adjusted close", "nan")),
                "volume": float(v.get("6. volume", "nan")),
            })
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        df = df.loc[mask].reset_index(drop=True)
        return normalize_price_df(df)
    except Exception:
        return None

# ---------- テクニカル ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int=14) -> pd.Series:
    return true_range(df).rolling(window=period, min_periods=period).mean()

def ann_vol_from_close(df: pd.DataFrame, period: int=20) -> pd.Series:
    return df["close"].pct_change().rolling(period).std() * np.sqrt(252.0) * 100.0

def inside_day(df: pd.DataFrame) -> pd.Series:
    prev_high = df["high"].shift(1)
    prev_low  = df["low"].shift(1)
    return (df["high"] <= prev_high) & (df["low"] >= prev_low)

def nr7(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).rolling(7).apply(lambda x: 1 if x[-1] == np.nanmin(x) else 0, raw=True)
    return rng.fillna(0).astype(bool)

def beta_and_r2(returns: pd.Series, market: pd.Series, window: int=60):
    cov = returns.rolling(window).cov(market)
    var = market.rolling(window).var()
    beta = cov / (var.replace(0, np.nan))
    corr = returns.rolling(window).corr(market)
    r2 = (corr * corr)
    return beta, r2

# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=DEFAULT_N)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--universe", type=str, default="sp500,dow30,nasdaq100,nasdaq")
    ap.add_argument("--since_days", type=int, default=CAL_DAYS_BACK)
    ap.add_argument("--source_order", type=str, default="yf,stooq,tiingo,alphav")
    ap.add_argument("--universe_source", type=str, default="yf")
    args = ap.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)

    tiingo_token = os.getenv("TIINGO_API_TOKEN", "").strip() or None
    alphav_key   = os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or None
    order = [s.strip() for s in args.source_order.split(",") if s.strip()]

    # 環境で最終上書き
    uni_src_order = [s.strip() for s in args.universe_source.split(",") if s.strip()]
    env_uni = os.getenv("FORCE_UNIVERSE_SOURCE", "").strip()
    if env_uni:
        uni_src_order = [s.strip() for s in env_uni.split(",") if s.strip()]

    run_ts = jst_now_str()
    start_date = (dt.utcnow().date() - timedelta(days=args.since_days)).strftime("%Y-%m-%d")
    end_date   = dt.utcnow().date().strftime("%Y-%m-%d")
    end_plus   = (pd.to_datetime(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")

    # 1) ユニバース
    idx_keys = [k.strip().lower() for k in args.universe.split(",") if k.strip()]
    uni_df = build_universe(idx_keys, uni_src_order)
    print(f"[universe] source={uni_src_order}, size={len(uni_df)} (first 10) {uni_df['ticker'].head(10).tolist()}")

    # 保存
    sm_parq = os.path.join(outdir, "security_master_us.parquet")
    sm_csv  = os.path.join(outdir, "security_master_us.csv")
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(uni_df), sm_parq)
        master_saved_as = "parquet"
    except Exception:
        uni_df.to_csv(sm_csv, index=False)
        master_saved_as = "csv"

    # 2) SPY（yfinance → Stooq → AV）
    spy_df = None
    for getter in (lambda: yf_bulk_multi_pass([SPY_SYMBOL], start_date, end_date).get(SPY_SYMBOL),
                   lambda: stooq_daily(SPY_SYMBOL, start_date, end_date),
                   lambda: alphav_daily_adj(SPY_SYMBOL, alphav_key, start_date, end_date)):
        spy_df = getter()
        if spy_df is not None and not spy_df.empty:
            break
    if spy_df is None or spy_df.empty:
        raise RuntimeError("SPYのEOD取得に失敗しました。")
    spy = spy_df.sort_values("date").copy()
    spy["mkt_ret"] = spy["close"].pct_change()
    spy_m = spy[["date","mkt_ret","close"]].rename(columns={"close":"spy_close"}).copy()

    # 3) 価格取得
    fetched: Dict[str, pd.DataFrame] = {}
    used_src: Dict[str, str] = {}
    tickers = [t for t in uni_df["ticker"].tolist() if t.upper() != SPY_SYMBOL]

    if "yf" in order:
        yf_res = yf_bulk_multi_pass(tickers, start_date, end_date)
        for t, df in yf_res.items():
            fetched[t] = df; used_src[t] = "yf"
        print(f"[fetch] yfinance ok: {sum(1 for v in used_src.values() if v=='yf')}")

    missing = [t for t in tickers if t not in fetched]
    if missing and "stooq" in order:
        for t in tqdm(missing, desc="Stooq (missing only)", ncols=90):
            df = stooq_daily(t, start_date, end_date)
            if df is not None and not df.empty:
                fetched[t] = df; used_src[t] = "stooq"
            time.sleep(0.02)
        print(f"[fetch] Stooq ok (added): {sum(1 for v in used_src.values() if v=='stooq')}")

    missing = [t for t in tickers if t not in fetched]
    if missing and "tiingo" in order:
        cap = len(missing)
        for t in tqdm(missing[:cap], desc="Tiingo (missing only)", ncols=90):
            df = tiingo_daily(t, tiingo_token, start_date, end_date)
            if df is not None and not df.empty:
                fetched[t] = df; used_src[t] = "tiingo"
        print(f"[fetch] Tiingo ok (added): {sum(1 for v in used_src.values() if v=='tiingo')}")

    missing = [t for t in tickers if t not in fetched]
    if missing and "alphav" in order:
        cap = min(500, len(missing))
        for t in tqdm(missing[:cap], desc="AlphaVantage (cap)", ncols=90):
            df = alphav_daily_adj(t, alphav_key, start_date, end_date)
            if df is not None and not df.empty:
                fetched[t] = df; used_src[t] = "alphav"
        print(f"[fetch] AlphaVantage ok (added): {sum(1 for v in used_src.values() if v=='alphav')}")

    total_ok = len(fetched)
    print(f"[fetch] total ok: {total_ok} / {len(tickers)}")

    if total_ok < 1000:
        print(f"[boost] fetched={total_ok} < 1000 → 再サンプリング＆再取得を実施")
        try:
            # すでにYAMLで巨大ユニバース供給済みの想定だが、念のため再走査
            extra = [t for t in tickers if t not in fetched][:5000]
            random.shuffle(extra)
            extra = extra[:2000]
            new = yf_bulk_multi_pass(extra, start_date, end_date)
            for t, df in new.items():
                if t not in fetched:
                    fetched[t] = df; used_src[t] = "yf"
        except Exception:
            pass
        total_ok = len(fetched)
        print(f"[boost] after extra pass total ok: {total_ok}")

    if not fetched:
        raise RuntimeError("EODを取得できた銘柄がありません。レート制限/ネットワーク/依存関係をご確認ください。")

    # 4) ADV20 Notional → N選定
    rows = []
    for t, df in fetched.items():
        adv20 = df["volume"].tail(20).mean() if len(df) >= 20 else np.nan
        lc = df["close"].iloc[-1]
        adv20_notional = float(adv20) * float(lc) if pd.notna(adv20) else np.nan
        nm = ""
        rows.append([t, nm, lc, adv20, adv20_notional, used_src.get(t,"")])
    basic = pd.DataFrame(rows, columns=["ticker","name","last_close","adv20_shares","adv20_notional","src"]).dropna(subset=["adv20_notional"])
    basic = basic.sort_values("adv20_notional", ascending=False).reset_index(drop=True)

    realized_N = min(args.N, len(basic))
    chosen = basic.head(realized_N)["ticker"].tolist()

    # 5) 指標計算
    prices_rows, analysis_rows = [], []

    for t in tqdm(chosen, desc="Indicators + Signals", ncols=90):
        df = fetched[t].copy().sort_values("date").reset_index(drop=True)
        if "adjClose" in df.columns and df["adjClose"].notna().any():
            df["close"] = df["adjClose"].fillna(df["close"])

        df["ema20"]  = ema(df["close"], 20)
        df["ema50"]  = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)
        df["atr14"]  = atr(df, 14)

        for n in [1,5,20,60,120,252]:
            df[f"r_{n}d"] = df["close"].pct_change(n)*100.0

        df["vol_20d_ann"] = ann_vol_from_close(df, 20)
        df["atr14_pct"]   = (df["atr14"]/df["close"])*100.0

        df["hh_52w"] = df["close"].rolling(252, min_periods=1).max()
        df["ll_52w"] = df["close"].rolling(252, min_periods=1).min()
        df["dist_52w_high"] = (df["close"]/df["hh_52w"] - 1.0)*100.0
        df["dist_52w_low"]  = (df["close"]/df["ll_52w"] - 1.0)*100.0

        df["ret"] = df["close"].pct_change()
        merged = pd.merge(df[["date","ret"]], spy_m[["date","mkt_ret"]], on="date", how="inner")
        beta_s, r2_s = beta_and_r2(merged["ret"], merged["mkt_ret"], 60)
        merged["beta_60d_vs_SPY"] = beta_s
        merged["beta_r2"] = r2_s
        df = pd.merge(df, merged[["date","beta_60d_vs_SPY","beta_r2"]], on="date", how="left")

        df["inside_day"] = inside_day(df)
        df["NR7"] = nr7(df)

        df["adv20_shares"] = df["volume"].rolling(20).mean()
        df["vol_spike"]    = df["volume"] / df["adv20_shares"]

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df)>=2 else None
        if pd.notna(last["ema20"]) and pd.notna(last["ema50"]):
            if last["ema20"] > last["ema50"]:
                signal = "BUY"
            elif last["ema20"] < last["ema50"]:
                signal = "SELL"
            else:
                dr = (last["close"]/prev["close"] - 1.0)*100.0 if prev is not None else 0.0
                signal = "BUY" if dr >= 0 else "SELL"
        else:
            signal = ""

        change_abs = float(last["close"] - (prev["close"] if prev is not None else last["close"]))
        change_pct = float(last["r_1d"]) if pd.notna(last.get("r_1d")) else 0.0
        prices_rows.append([
            t, "", "", float(last["close"]), change_abs, change_pct,
            int(last["volume"]) if pd.notna(last["volume"]) else "",
            float(last["atr14"]) if pd.notna(last["atr14"]) else "",
            float(last["ema20"]) if pd.notna(last["ema20"]) else "",
            float(last["ema50"]) if pd.notna(last["ema50"]) else "",
            signal, "EOD", 0, "", dt.utcnow().strftime("%Y-%m-%d")
        ])

        # RS(12w)：spy_m を使用（spy_closeはspy_mにのみ存在）
        tmp = pd.merge(df[["date","close"]], spy_m[["date","spy_close"]], on="date", how="inner")
        rs_12w_score = np.nan
        if len(tmp) >= 60:
            tmp["rel"] = (tmp["close"]/tmp["spy_close"]) / (tmp["close"].shift(60)/tmp["spy_close"].shift(60)) - 1.0
            rs_12w_score = tmp["rel"].iloc[-1] * 100.0

        trend_stack = ""
        if pd.notna(last.get("ema20")) and pd.notna(last.get("ema50")) and pd.notna(last.get("ema200")):
            order_vals = [("20", last["ema20"]), ("50", last["ema50"]), ("200", last["ema200"])]
            order = [k for k, _ in sorted(order_vals, key=lambda x: x[1], reverse=True)]
            trend_stack = ">".join(order)

        new_high_20d = bool(last["close"] >= df["close"].tail(20).max()) if len(df)>=20 else False
        new_low_20d  = bool(last["close"] <= df["close"].tail(20).min()) if len(df)>=20 else False

        analysis_rows.append([
            t,
            float(last.get("r_1d", np.nan))  if pd.notna(last.get("r_1d", np.nan))  else "",
            float(last.get("r_5d", np.nan))  if pd.notna(last.get("r_5d", np.nan))  else "",
            float(last.get("r_20d", np.nan)) if pd.notna(last.get("r_20d", np.nan)) else "",
            float(last.get("r_60d", np.nan)) if pd.notna(last.get("r_60d", np.nan)) else "",
            float(last.get("r_120d", np.nan))if pd.notna(last.get("r_120d", np.nan))else "",
            float(last.get("r_252d", np.nan))if pd.notna(last.get("r_252d", np.nan))else "",
            float(last.get("vol_20d_ann", np.nan)) if pd.notna(last.get("vol_20d_ann", np.nan)) else "",
            float(last.get("atr14_pct", np.nan))   if pd.notna(last.get("atr14_pct", np.nan))   else "",
            float(last.get("adv20_shares", np.nan))if pd.notna(last.get("adv20_shares", np.nan))else "",
            float(last.get("adv20_shares", np.nan))*float(last["close"]) if pd.notna(last.get("adv20_shares", np.nan)) else "",
            float(last.get("dist_52w_high", np.nan)) if pd.notna(last.get("dist_52w_high", np.nan)) else "",
            float(last.get("dist_52w_low", np.nan))  if pd.notna(last.get("dist_52w_low", np.nan))  else "",
            rs_12w_score,
            trend_stack,
            new_high_20d,
            new_low_20d,
            float(last.get("vol_spike", np.nan)) if pd.notna(last.get("vol_spike", np.nan)) else "",
            np.nan,  # beta_60d_vs_SPY（必要なら詳細再計算可）
            np.nan,  # beta_r2
            "unknown",
            "", "", "",
            bool(last.get("inside_day", False)),
            bool(last.get("NR7", False)),
        ])

    # DataFrames
    prices_cols = ["ticker","name","exchange","last_price","change","change_pct","volume",
                   "atr14","ema20","ema50","signal","quote_session","stale","missing_reason","data_timestamp"]
    prices_df = pd.DataFrame(prices_rows, columns=prices_cols)

    analysis_cols = ["ticker","r_1d","r_5d","r_20d","r_60d","r_120d","r_252d",
                     "vol_20d_ann","atr14_pct","adv20_shares","adv20_notional",
                     "dist_52w_high","dist_52w_low","rs_12w_percentile",
                     "trend_stack","new_high_20d","new_low_20d","vol_spike",
                     "beta_60d_vs_SPY","beta_r2","earnings_within_7d",
                     "comp_score","risk_band","analysis_note","inside_day","NR7"]
    analysis_df = pd.DataFrame(analysis_rows, columns=analysis_cols)

    # rs_12w_percentile：全体ランクへ
    vals = pd.to_numeric(analysis_df["rs_12w_percentile"], errors="coerce")
    ranks = vals.rank(pct=True) * 100.0
    analysis_df["rs_12w_percentile"] = ranks.round(2)

    # comp_score（0-100）
    def zscore(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        return (s - s.mean(skipna=True)) / (s.std(skipna=True) + 1e-9)

    mom = zscore(pd.to_numeric(analysis_df["r_60d"], errors="coerce"))
    rs  = zscore(pd.to_numeric(analysis_df["rs_12w_percentile"], errors="coerce"))
    trd = (pd.to_numeric(prices_df["ema20"], errors="coerce") > pd.to_numeric(prices_df["ema50"], errors="coerce")).astype(int)
    risk = -zscore(pd.to_numeric(analysis_df["vol_20d_ann"], errors="coerce"))

    comp = (0.4*mom.fillna(0) + 0.3*rs.fillna(0) + 0.2*trd.fillna(0) + 0.1*risk.fillna(0))
    comp_norm = pd.Series(50.0, index=comp.index) if comp.max()==comp.min() else ((comp - comp.min())/(comp.max()-comp.min())*100.0)
    analysis_df["comp_score"] = comp_norm.round(1)

    try:
        analysis_df["risk_band"] = pd.qcut(pd.to_numeric(analysis_df["vol_20d_ann"], errors="coerce"), 3, labels=["Low","Med","High"]).astype(str)
    except Exception:
        analysis_df["risk_band"] = ""

    notes = []
    for _, r in analysis_df.iterrows():
        note = []
        if isinstance(r["new_high_20d"], bool) and r["new_high_20d"]:
            note.append("20d新高値")
        if isinstance(r["new_low_20d"], bool) and r["new_low_20d"]:
            note.append("20d新安値")
        if pd.to_numeric(r["vol_spike"], errors="coerce") >= 2.0:
            note.append("出来高スパイク")
        if pd.to_numeric(r["rs_12w_percentile"], errors="coerce") >= 80:
            note.append("RS上位")
        if pd.to_numeric(r["comp_score"], errors="coerce") >= 80:
            note.append("総合強")
        notes.append(" / ".join(note))
    analysis_df["analysis_note"] = notes

    # 6) サマリー/アラート
    prices_df["change_pct"] = pd.to_numeric(prices_df["change_pct"], errors="coerce")
    up_df = prices_df.sort_values("change_pct", ascending=False).head(20)
    dn_df = prices_df.sort_values("change_pct", ascending=True).head(20)
    rising = int((prices_df["change_pct"] > 0).sum())
    falling = int((prices_df["change_pct"] < 0).sum())

    alerts = []
    for _, r in analysis_df.iterrows():
        flags = []
        if isinstance(r["new_high_20d"], bool) and r["new_high_20d"]:
            flags.append("20d新高値")
        if isinstance(r["new_low_20d"], bool) and r["new_low_20d"]:
            flags.append("20d新安値")
        if pd.to_numeric(r["rs_12w_percentile"], errors="coerce") >= 90:
            flags.append("RS上位10%")
        if pd.to_numeric(r["rs_12w_percentile"], errors="coerce") <= 10:
            flags.append("RS下位10%")
        if pd.to_numeric(r["vol_spike"], errors="coerce") >= 2.5:
            flags.append("出来高スパイク(2.5x+)")
        if flags:
            cs = float(pd.to_numeric(r["comp_score"], errors="coerce") if pd.notna(r["comp_score"]) else -1)
            alerts.append((r["ticker"], ", ".join(flags), cs))
    alerts = sorted(alerts, key=lambda x: x[2], reverse=True)[:200]

    # 7) 出力
    with open(os.path.join(outdir, "market_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Market Summary — {jst_now_str()}\n\n")
        f.write(f"選定N（realized）: {len(prices_df)}\n")
        f.write(f"値上がり数: {rising} / 値下がり数: {falling}\n\n")
        f.write("【上昇率トップ20】\n")
        for _, r in up_df.iterrows():
            f.write(f"{r['ticker']}: {r['change_pct']:.2f}%\n")
        f.write("\n【下落率トップ20】\n")
        for _, r in dn_df.iterrows():
            f.write(f"{r['ticker']}: {r['change_pct']:.2f}%\n")

    prices_df.to_csv(os.path.join(outdir, "prices_us_all.csv"), index=False)

    cov = pd.DataFrame([{
        "planned_universe_N": int(args.N),
        "selected_universe_N": int(len(prices_df)),
        "rows_returned": int(len(prices_df)),
        "coverage_pct": float(100.0 if len(prices_df)>0 else 0.0),
        "missing_symbols": "",
        "data_source_notes": f"fetch_order={order}; universe_source={uni_src_order}; master_saved_as={'parquet' if os.path.exists(os.path.join(outdir,'security_master_us.parquet')) else 'csv'}"
    }])
    cov.to_csv(os.path.join(outdir, "coverage_report_us.csv"), index=False)

    eod_target = spy["date"].max().strftime("%Y-%m-%d")
    with open(os.path.join(outdir, "data_info_us.txt"), "w", encoding="utf-8") as f:
        f.write("Data Timestamp & Sources (EOD-only)\n")
        f.write(f"EOD target date: {eod_target}\n")
        f.write(f"Fetch order: {order}\n")
        f.write(f"Universe source: {uni_src_order}\n")
        f.write(f"Run time (JST): {jst_now_str()}\n")

    analysis_df.to_csv(os.path.join(outdir, "stock_analysis_us.csv"), index=False)

    with open(os.path.join(outdir, "stock_alerts_us.txt"), "w", encoding="utf-8") as f:
        for i, (tic, flg, score) in enumerate(alerts[:50], 1):
            f.write(f"{i:02d}. {tic} — {flg}（comp_score={score:.1f}）\n")

    print(f"[DONE] Outputs saved to: {outdir}")

if __name__ == "__main__":
    main()
