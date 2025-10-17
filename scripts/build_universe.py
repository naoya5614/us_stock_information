#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import pathlib
import sys

# 入力ファイル（scripts/fetch_symdirs.sh が保存）
NAS_FILE = "data/nasdaqlisted.txt"
OTH_FILE = "data/otherlisted.txt"

# 読み取り列
NAS_COLS = ["Symbol", "Security Name", "Test Issue", "ETF", "NextShares"]
OTH_COLS = ["ACT Symbol", "Security Name", "Exchange", "Test Issue", "ETF"]

def read_nasdaq(path: str) -> pd.DataFrame:
    try:
        df = (
            pd.read_csv(
                path,
                sep="|",
                usecols=NAS_COLS,
                dtype=str,
                engine="python",
                skipfooter=1,  # 末尾の "File Creation Time" 行を除去
            )
            .rename(
                columns={
                    "Symbol": "ticker",
                    "Security Name": "name",
                    "Test Issue": "test",
                    "ETF": "etf",
                    "NextShares": "next",
                }
            )
        )
    except Exception as e:
        print(f"[error] reading nasdaqlisted.txt: {e}", file=sys.stderr)
        raise
    # ETF/Next/テスト除外
    df = df[(df["etf"] == "N") & (df["next"] == "N") & (df["test"] == "N")]
    df["exchange"] = "NASDAQ"
    return df[["ticker", "name", "exchange"]]

def read_otherlisted(path: str) -> pd.DataFrame:
    try:
        df = (
            pd.read_csv(
                path,
                sep="|",
                usecols=OTH_COLS,
                dtype=str,
                engine="python",
                skipfooter=1,
            )
            .rename(
                columns={
                    "ACT Symbol": "ticker",
                    "Security Name": "name",
                    "Exchange": "exch",
                    "Test Issue": "test",
                    "ETF": "etf",
                }
            )
        )
    except Exception as e:
        print(f"[error] reading otherlisted.txt: {e}", file=sys.stderr)
        raise
    # N:NYSE, A:NYSE MKT(AMEX), P:NYSE ARCA（Cboe/BATS "Z" などは除外）
    valid_ex = {"N", "A", "P"}
    df = df[(df["etf"] == "N") & (df["test"] == "N") & (df["exch"].isin(valid_ex))]
    df["exchange"] = df["exch"]
    return df[["ticker", "name", "exchange"]]

def main():
    nas = read_nasdaq(NAS_FILE)
    oth = read_otherlisted(OTH_FILE)
    df = pd.concat([nas, oth], ignore_index=True)

    # ワラント/権利/ユニット等を名称で除外（簡易）
    pat = re.compile(r"\b(Warrant|Warrants|Rt|Right|Rights|Units?)\b", re.IGNORECASE)
    df = df[~df["name"].fillna("").str.contains(pat, na=False)]

    # TICKER 正規化 & 重複排除
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    out = pathlib.Path("data/universe.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"[universe-csv] rows={len(df)}  sample={df['ticker'].head(10).tolist()}")

if __name__ == "__main__":
    main()
