# run_us_eod_liquidity_report.py
# -*- coding: utf-8 -*-
"""
EOD（前日終値）×上位流動性N銘柄の一括取得→分析→A〜F出力
- フリー優先のフェイルオーバー: yfinance → Tiingo → Alpha Vantage
- 取得できた銘柄の中から ADV20（出来高×終値）上位で N を確定
- 確定N銘柄に対して coverage=100% を担保（未取得銘柄は集合から外す）
- すべてEODベース（リアルタイム/時間外は未取得）

出力（--outdir 配下）:
 A) market_summary.txt
 B) prices_us_all.csv
 C) coverage_report_us.csv
 D) data_info_us.txt
 E) stock_analysis_us.csv
 F) stock_alerts_us.txt
 参考) security_master_us.parquet (pyarrowあり) / .csv (代替)

依存: pandas numpy requests tqdm pyarrow(任意) yfinance
"""

import os, sys, time, math, argparse
from datetime import datetime as dt, timedelta, timezone
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

# ---------- 基本設定 ----------
DEFAULT_OUTDIR = "./out"
DEFAULT_N = 1500
CAL_DAYS_BACK = 560          # 約400営業日相当をカバー
SPY_SYMBOL = "SPY"          # RS/βのベンチマーク
JST = timezone(timedelta(hours=9))

# Wikipedia（候補プール用）
WIKI_LISTS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "dow30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "r1000": "https://en.wikipedia.org/wiki/Russell_1000_Index",
    "sp400": "https://en.wikipedia.org/wiki/S%26P_400",
    "sp600": "https://en.wikipedia.org/wiki/S%26P_600",
}

# ---------- ユーティリティ ----------
def jst_now_str() -> str:
    return dt.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def robust_get(url: str, params=None, headers=None, timeout=30, retries=3, backoff=1.5):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(backoff * (i + 1))
    return None

def parse_wiki_tickers(html: str) -> List[Tuple[str, str]]:
    """Symbol/Company/Nameのテーブルをざっくり抽出→[(ticker, name)]"""
    try:
        tables = pd.read_html(html)
    except ValueError:
        return []
    results = []
    for df in tables:
        cols = [str(c).lower() for c in df.columns]
        if not any("symbol" in c for c in cols): 
            continue
        name_idx = None
        for key in ["security", "company", "name"]:
            arr = [i for i, c in enumerate(cols) if key in c]
            if arr:
                name_idx = arr[0]; break
        sym_idx = [i for i, c in enumerate(cols) if "symbol" in c]
        if not sym_idx: 
            continue
        sidx = sym_idx[0]
        for _, row in df.iterrows():
            try:
                sym = str(row.iloc[sidx]).strip().replace(" ", "").upper()
                if not sym or sym.lower() == "nan":
                    continue
                nm = ""
                if name_idx is not None:
                    nm = str(row.iloc[name_idx]).strip()
                results.append((sym, nm))
            except Exception:
                continue
    return results

def hydrate_universe_from_wiki(keys: List[str]) -> pd.DataFrame:
    rows = []
    for k in keys:
        url = WIKI_LISTS.get(k)
        if not url: 
            continue
        r = robust_get(url, timeout=40)
        if not r:
            continue
        pairs = parse_wiki_tickers(r.text)
        for t, n in pairs:
            rows.append((t, n, k))
        time.sleep(1.0)  # 優しめウェイト
    if not rows:
        raise RuntimeError("Index lists fetch failed (Wikipedia).")
    df = pd.DataFrame(rows, columns=["ticker","name","list_source"]).drop_duplicates("ticker")
    return df.reset_index(drop=True)

# ---------- データ取得（yfinance / Tiingo / AlphaVantage） ----------
def yf_daily(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns={
            "Date":"date","Open":"open","High":"high","Low":"low",
            "Close":"close","Adj Close":"adjClose","Volume":"volume"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None

def tiingo_daily(ticker: str, token: Optional[str], start: str, end: str) -> Optional[pd.DataFrame]:
    if not token: 
        return None
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": start, "endDate": end, "format": "json", "token": token}
    r = robust_get(url, params=params, timeout=30)
    if not r:
        return None
    try:
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        # 正規化
        for c_from, c_to in [("adjClose","adjClose"),("close","close"),("open","open"),("high","high"),("low","low"),("volume","volume")]:
            if c_from not in df.columns:
                df[c_from] = np.nan
        return df.sort_values("date").reset_index(drop=True)
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
        return df if not df.empty else None
    except Exception:
        return None

def fetch_prices_history(ticker: str, start: str, end: str,
                         tiingo_token: Optional[str],
                         alphav_key: Optional[str],
                         order: List[str]) -> Tuple[Optional[pd.DataFrame], str]:
    """order例: ['yf','tiingo','alphav']"""
    for src in order:
        if src == "yf":
            df = yf_daily(ticker, start, end)
        elif src == "tiingo":
            df = tiingo_daily(ticker, tiingo_token, start, end)
        elif src == "alphav":
            df = alphav_daily_adj(ticker, alphav_key, start, end)
        else:
            df = None
        if df is not None and not df.empty:
            return df, src
    return None, "missing"

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

def beta_and_r2(returns: pd.Series, market: pd.Series, window: int=60) -> Tuple[pd.Series, pd.Series]:
    cov = returns.rolling(window).cov(market)
    var = market.rolling(window).var()
    beta = cov / (var.replace(0, np.nan))
    corr = returns.rolling(window).corr(market)
    r2 = (corr * corr)
    return beta, r2

# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=DEFAULT_N, help="目標の上位流動性銘柄数（取得できた銘柄の中から確定）")
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="出力先ディレクトリ")
    ap.add_argument("--universe", type=str, default="sp500,nasdaq100,dow30,r1000,sp400,sp600",
                    help="候補プールに使う指数（カンマ区切り）")
    ap.add_argument("--since_days", type=int, default=CAL_DAYS_BACK, help="何日前までのEODを取るか（カレンダー日）")
    ap.add_argument("--source_order", type=str, default="yf,tiingo,alphav",
                    help="取得優先順（例: 'yf,tiingo,alphav'）")
    args = ap.parse_args()

    outdir = args.outdir
    ensure_dir(outdir)

    tiingo_token = os.getenv("TIINGO_API_TOKEN", "").strip() or None
    alphav_key   = os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or None
    order = [s.strip() for s in args.source_order.split(",") if s.strip()]

    run_ts = jst_now_str()
    start_date = (dt.utcnow().date() - timedelta(days=args.since_days)).strftime("%Y-%m-%d")
    end_date   = dt.utcnow().date().strftime("%Y-%m-%d")

    # 1) 候補プール（Wikipediaの指数ユニオン）
    idx_keys = [k.strip().lower() for k in args.universe.split(",") if k.strip()]
    uni_df = hydrate_universe_from_wiki(idx_keys)  # ticker, name, list_source

    # 保存（parquet優先）
    sm_parq = os.path.join(outdir, "security_master_us.parquet")
    sm_csv  = os.path.join(outdir, "security_master_us.csv")
    master_saved_as = "parquet"
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(uni_df), sm_parq)
    except Exception:
        uni_df.to_csv(sm_csv, index=False)
        master_saved_as = "csv"

    # 2) SPY（ベンチマーク）
    spy_df, spy_src = fetch_prices_history(SPY_SYMBOL, start_date, end_date, tiingo_token, alphav_key, order)
    if spy_df is None or spy_df.empty:
        raise RuntimeError("SPYのEOD取得に失敗しました。ネットワーク/データ源をご確認ください。")
    spy = spy_df.sort_values("date").copy()
    spy["mkt_ret"] = spy["close"].pct_change()

    # 3) 各銘柄のEOD取得
    fetched: Dict[str, pd.DataFrame] = {}
    used_src: Dict[str, str] = {}
    tickers = [t for t in uni_df["ticker"].tolist() if t.upper() != SPY_SYMBOL]

    for t in tqdm(tickers, desc="Fetching EOD", ncols=90):
        df, src = fetch_prices_history(t, start_date, end_date, tiingo_token, alphav_key, order)
        if df is None or df.empty:
            continue
        need = {"date","open","high","low","close","volume"}
        if not need.issubset(df.columns):
            continue
        df = df.dropna(subset=["close","high","low","volume"]).sort_values("date")
        if df.empty:
            continue
        fetched[t] = df.reset_index(drop=True)
        used_src[t] = src

    if not fetched:
        raise RuntimeError("EODを取得できた銘柄がありません。データ源やレート制限をご確認ください。")

    # 4) ADV20 Notionalを算出して上位Nを確定
    rows = []
    for t, df in fetched.items():
        adv20 = df["volume"].tail(20).mean() if len(df) >= 20 else np.nan
        lc = df["close"].iloc[-1]
        adv20_notional = float(adv20) * float(lc) if pd.notna(adv20) else np.nan
        nm = uni_df.loc[uni_df["ticker"]==t, "name"]
        nm = nm.iloc[0] if len(nm) else ""
        rows.append([t, nm, lc, adv20, adv20_notional, used_src.get(t,"")])
    basic = pd.DataFrame(rows, columns=["ticker","name","last_close","adv20_shares","adv20_notional","src"]).dropna(subset=["adv20_notional"])
    basic = basic.sort_values("adv20_notional", ascending=False).reset_index(drop=True)

    realized_N = min(args.N, len(basic))
    chosen = basic.head(realized_N)["ticker"].tolist()
    chosen_set = set(chosen)

    # 5) 指標計算（選定N）
    prices_rows = []
    analysis_rows = []

    # SPYと日付でマージ用に、軽く辞書化
    spy_m = spy[["date","mkt_ret","close"]].rename(columns={"close":"spy_close"}).copy()

    for t in tqdm(chosen, desc="Indicators + Signals", ncols=90):
        df = fetched[t].copy().sort_values("date").reset_index(drop=True)

        # 調整終値があれば優先
        if "adjClose" in df.columns and df["adjClose"].notna().any():
            df["close"] = df["adjClose"].fillna(df["close"])

        # EMA/ATR
        df["ema20"]  = ema(df["close"], 20)
        df["ema50"]  = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)
        df["atr14"]  = atr(df, 14)

        # リターン系列
        for n in [1,5,20,60,120,252]:
            df[f"r_{n}d"] = df["close"].pct_change(n)*100.0

        # 年率ボラ、ATR%
        df["vol_20d_ann"] = ann_vol_from_close(df, 20)
        df["atr14_pct"]   = (df["atr14"]/df["close"])*100.0

        # 52週距離
        df["hh_52w"] = df["close"].rolling(252, min_periods=1).max()
        df["ll_52w"] = df["close"].rolling(252, min_periods=1).min()
        df["dist_52w_high"] = (df["close"]/df["hh_52w"] - 1.0)*100.0
        df["dist_52w_low"]  = (df["close"]/df["ll_52w"] - 1.0)*100.0

        # βとR^2（60d）
        df["ret"] = df["close"].pct_change()
        merged = pd.merge(df[["date","ret"]], spy_m[["date","mkt_ret"]], on="date", how="inner")
        beta_s, r2_s = beta_and_r2(merged["ret"], merged["mkt_ret"], 60)
        merged["beta_60d_vs_SPY"] = beta_s
        merged["beta_r2"] = r2_s
        df = pd.merge(df, merged[["date","beta_60d_vs_SPY","beta_r2"]], on="date", how="left")

        # パターン
        df["inside_day"] = inside_day(df)
        df["NR7"] = nr7(df)

        # 出来高異常
        df["adv20_shares"] = df["volume"].rolling(20).mean()
        df["vol_spike"]    = df["volume"] / df["adv20_shares"]

        # シグナル
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

        # Prices（B）
        change_abs = float(last["close"] - (prev["close"] if prev is not None else last["close"]))
        change_pct = float(last["r_1d"]) if pd.notna(last.get("r_1d")) else 0.0
        prices_rows.append([
            t,
            (basic.loc[basic["ticker"]==t, "name"].iloc[0] if (basic["ticker"]==t).any() else ""),
            "",  # exchange 不明：空
            float(last["close"]),
            change_abs,
            change_pct,
            int(last["volume"]) if pd.notna(last["volume"]) else "",
            float(last["atr14"]) if pd.notna(last["atr14"]) else "",
            float(last["ema20"]) if pd.notna(last["ema20"]) else "",
            float(last["ema50"]) if pd.notna(last["ema50"]) else "",
            signal,
            "EOD",
            0,    # stale
            "",   # missing_reason
            dt.utcnow().strftime("%Y-%m-%d")
        ])

        # Analysis（E）
        # RS（12週=60d）をベンチマーク比で厳密化：過去60営業日の相対価格比
        rs_12w_percentile = np.nan
        if len(df) >= 60:
            # SPYと日付で揃え、比率の60日リターンを比較
            tmp = pd.merge(df[["date","close"]], spy[["date","close"]].rename(columns={"close":"spy_close"}), on="date", how="inner")
            if len(tmp) >= 60:
                tmp["rel"] = (tmp["close"]/tmp["spy_close"]) / (tmp["close"].shift(60)/tmp["spy_close"].shift(60)) - 1.0
                rs_12w = tmp["rel"].iloc[-1] * 100.0
                # 後で全体パーセンタイルにするため一旦保存
                rs_12w_percentile = rs_12w  # ここは一時値、後でrankに差し替え

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
            rs_12w_percentile,  # 後で全体パーセンタイルに変換
            trend_stack,
            new_high_20d,
            new_low_20d,
            float(last.get("vol_spike", np.nan)) if pd.notna(last.get("vol_spike", np.nan)) else "",
            float(last.get("beta_60d_vs_SPY", np.nan)) if pd.notna(last.get("beta_60d_vs_SPY", np.nan)) else "",
            float(last.get("beta_r2", np.nan))         if pd.notna(last.get("beta_r2", np.nan))         else "",
            "unknown",  # earnings_within_7d（無料での網羅は困難)
            "", "", "",  # comp_score, risk_band, analysis_note（あとで埋める）
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

    # rs_12w_percentile を全体ランクに変換（現状は「相対12週リターン%」の生値）
    if "rs_12w_percentile" in analysis_df.columns:
        vals = pd.to_numeric(analysis_df["rs_12w_percentile"], errors="coerce")
        ranks = vals.rank(pct=True) * 100.0
        analysis_df["rs_12w_percentile"] = ranks.round(2)

    # comp_score（0-100）: モメンタム/相対力/トレンド/低ボラを合成
    def zscore(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        return (s - s.mean(skipna=True)) / (s.std(skipna=True) + 1e-9)

    mom = zscore(pd.to_numeric(analysis_df["r_60d"], errors="coerce"))
    rs  = zscore(pd.to_numeric(analysis_df["rs_12w_percentile"], errors="coerce"))
    trd = (pd.to_numeric(prices_df["ema20"], errors="coerce") > pd.to_numeric(prices_df["ema50"], errors="coerce")).astype(int)
    risk = -zscore(pd.to_numeric(analysis_df["vol_20d_ann"], errors="coerce"))

    comp = (0.4*mom.fillna(0) + 0.3*rs.fillna(0) + 0.2*trd.fillna(0) + 0.1*risk.fillna(0))
    if comp.max() == comp.min():
        comp_norm = pd.Series(50.0, index=comp.index)  # 全部同値のとき
    else:
        comp_norm = ((comp - comp.min()) / (comp.max() - comp.min()) * 100.0)
    analysis_df["comp_score"] = comp_norm.round(1)

    # risk_band（三分位）
    try:
        analysis_df["risk_band"] = pd.qcut(pd.to_numeric(analysis_df["vol_20d_ann"], errors="coerce"), 3, labels=["Low","Med","High"]).astype(str)
    except Exception:
        analysis_df["risk_band"] = ""

    # analysis_note（簡易）
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
            alerts.append((r["ticker"], ", ".join(flags), float(pd.to_numeric(r["comp_score"], errors="coerce") if pd.notna(r["comp_score"]) else -1)))
    alerts = sorted(alerts, key=lambda x: x[2], reverse=True)[:200]

    # 7) 出力
    # A
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

    # B
    prices_df.to_csv(os.path.join(outdir, "prices_us_all.csv"), index=False)

    # C
    cov = pd.DataFrame([{
        "planned_universe_N": int(args.N),
        "selected_universe_N": int(len(prices_df)),
        "rows_returned": int(len(prices_df)),
        "coverage_pct": 100.0 if len(prices_df)>0 else 0.0,
        "missing_symbols": "",
        "data_source_notes": f"fetch_order={order}; universe=Wikipedia index union; master_saved_as={master_saved_as}"
    }])
    cov.to_csv(os.path.join(outdir, "coverage_report_us.csv"), index=False)

    # D
    eod_target = spy["date"].max().strftime("%Y-%m-%d")
    with open(os.path.join(outdir, "data_info_us.txt"), "w", encoding="utf-8") as f:
        f.write("Data Timestamp & Sources (EOD-only)\n")
        f.write(f"EOD target date: {eod_target}\n")
        f.write(f"Fetch order: {order}\n")
        f.write(f"Run time (JST): {run_ts}\n")

    # E
    analysis_df.to_csv(os.path.join(outdir, "stock_analysis_us.csv"), index=False)

    # F
    with open(os.path.join(outdir, "stock_alerts_us.txt"), "w", encoding="utf-8") as f:
        for i, (tic, flg, score) in enumerate(alerts[:50], 1):
            f.write(f"{i:02d}. {tic} — {flg}（comp_score={score:.1f}）\n")

    print(f"[DONE] Outputs saved to: {outdir}")

if __name__ == "__main__":
    main()
