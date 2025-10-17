#!/usr/bin/env bash
set -euo pipefail

mkdir -p data

# ミラー優先（www → apex → ftp）、HTTP/HTTPS 両方、短いリトライ＋ハードタイムアウト
fetch_file() {
  local out="$1" ; shift
  local urls=("$@")
  local ok=0
  for u in "${urls[@]}"; do
    echo "[fetch] $u -> $out"
    if timeout 40s curl -fsS \
         --connect-timeout 8 --max-time 25 \
         --retry 2 --retry-delay 2 --retry-all-errors \
         -o "$out" "$u"; then
      dos2unix "$out" >/dev/null 2>&1 || true
      local sz
      sz=$(wc -c < "$out" || echo 0)
      if [ "$sz" -ge 2048 ]; then
        echo "[fetch] ok size=${sz} bytes"
        ok=1
        break
      else
        echo "[fetch] too small (${sz} bytes), will try next URL"
      fi
    else
      echo "[fetch] curl failed, trying next URL"
    fi
  done
  if [ "$ok" -ne 1 ]; then
    echo "::error::Failed to fetch $out from all mirrors"
    exit 1
  fi
}

NAS_URLS=(
  "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
  "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
  "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
  "http://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
  "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
  "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)
OTH_URLS=(
  "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
  "http://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
  "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
  "http://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
  "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
  "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)

fetch_file "data/nasdaqlisted.txt" "${NAS_URLS[@]}"
fetch_file "data/otherlisted.txt"  "${OTH_URLS[@]}"

echo "[head] nasdaqlisted.txt:"
head -n 3 data/nasdaqlisted.txt || true
echo "[head] otherlisted.txt:"
head -n 3 data/otherlisted.txt || true
