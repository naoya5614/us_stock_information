# scripts/build_universe.py
# -*- coding: utf-8 -*-
"""
Nasdaq Trader の公式ディレクトリ（nasdaqlisted.txt / otherlisted.txt）から
NYSE / NYSE MKT(AMEX) / NYSE ARCA / NASDAQ の ACTIVE 普通株ユニバースを構築し、
data/universe.csv を出力する。
前段のワークフローで data/*.txt を取得済みであることが前提。
"""

import sys
import re
from pathlib import Path
import pandas as pd

NAS_PATH = Path("data/nasdaqlisted.txt")
OTH_PATH = Path("data/otherlisted.txt")
OUT_CSV  = Path("data/universe.csv")

if not NAS_PATH.exists() or not OTH_PATH.exists():
    print("[error] missing nasdaqlisted.txt or otherlisted.txt", file=sys.stderr)
    sys.exit(1)

NAS_COLS = ["Symbol", "Security Name", "Test Issue", "ETF", "NextShares"]
OTH_COLS = ["ACT Symbol", "Security Name", "Exchange", "Test Issue", "ETF"]

# NASDAQ 上場（ETF/NextShares/テスト除外）
try:
    nas = pd.read_csv(
        NAS_PATH,
        sep="|",
        usecols=NAS_COLS,
        dtype=str,
        engine="python",
        skipfooter=1,  # "File Creation Time" のフッター行を除去
    ).rename(
        columns={
            "Symbol": "ticker",
            "Security Name": "name",
            "Test Issue": "test",
            "ETF": "etf",
            "NextShares": "next",
        }
    )
except Exception as e:
    print(f"[error] reading {NAS_PATH}: {e}", file=sys.stderr)
    raise

nas = nas[(nas["etf"] == "N") & (nas["next"] == "N") & (nas["test"] == "N")]
nas["exchange"] = "NASDAQ"

# NYSE/ARCA/AMEX（ETF/テスト除外）
try:
    oth = pd.read_csv(
        OTH_PATH,
        sep="|",
        usecols=OTH_COLS,
        dtype=str,
        engine="python",
        skipfooter=1,
    ).rename(
        columns={
            "ACT Symbol": "ticker",
            "Security Name": "name",
            "Exchange": "exch",
            "Test Issue": "test",
            "ETF": "etf",
        }
    )
except Exception as e:
    print(f"[error] reading {OTH_PATH}: {e}", file=sys.stderr)
    raise

valid_ex = {"N", "A", "P"}  # N:NYSE, A:NYSE MKT(AMEX), P:NYSE ARCA
oth = oth[(oth["etf"] == "N") & (oth["test"] == "N") & (oth["exch"].isin(valid_ex))]
oth["exchange"] = oth["exch"]

df = pd.concat(
    [nas[["ticker", "name", "exchange"]], oth[["ticker", "name", "exchange"]]],
    ignore_index=True,
)

# ワラント/権利/ユニット等を名称で除外（簡易）
pat = re.compile(r"\b(Warrant|Warrants|Rt|Right|Rights|Units?)\b", re.IGNORECASE)
df = df[~df["name"].fillna("").str.contains(pat, na=False)]

# TICKER 正規化 & 重複排除
df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

OUT_CSV.write_text(df.to_csv(index=False), encoding="utf-8")
print(f"[universe-csv] rows={len(df)}  sample={df['ticker'].head(10).tolist()}")
