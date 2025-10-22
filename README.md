# Quant Crypto MVP — FIL/USDT 5m (Binance)

This is a minimal, end-to-end crypto trading MVP tailored for **Filecoin (FIL/USDT)** on a **5-minute** timeframe, with:
- Trend model (LightGBM) using technical features
- Lightweight order-book proxies (can be upgraded to real depth later)
- Signal fusion
- Dynamic stop (ATR-based) & trailing stop
- Simple slippage model
- Vectorbt backtest
- FastAPI paper-trading service

> **Note:** This repo defaults to **Binance spot** via `ccxt` and targets **FIL/USDT 5m**.  
> If you trade futures/perpetual, switch to `binanceusdm` in config and add funding/fee settings accordingly.

## Quickstart

### 1) Python env

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure

Edit **conf/config.yml** as needed. Copy `.env.example` to `.env` if you plan to use authenticated calls.

```bash
cp .env.example .env
```

### 3) Fetch data & build features

```bash
python data/prepare.py
```

### 4) Train model

```bash
python models/train_lgb.py
```

### 5) Backtest

```bash
python backtest/run_vectorbt.py
```

### 6) Run paper-trading API

```bash
uvicorn live.app:app --host 0.0.0.0 --port 8000 --reload
# GET http://localhost:8000/health
# GET http://localhost:8000/signal
```

### 7) Docker (optional)

```bash
docker build -t quant-crypto-fil5m .
docker run --rm -p 8000:8000 quant-crypto-fil5m
```

## Project layout

```
quant-crypto-fil5m/
├─ data/               # Raw & feature files (Parquet)
├─ models/             # Trained model files
├─ backtest/           # Backtesting scripts
├─ live/               # FastAPI paper trading
├─ conf/               # Config files
├─ utils/              # Helper functions
└─ scripts/            # Helper scripts
```

## Upgrades

- Replace order-book proxies with real Binance depth subscription (`python-binance` / websockets).
- Add regime detection (HMM/LSTM) to scale exposure.
- Execution splitting (TWAP/VWAP), refine slippage by trade size vs. depth.
- Monitoring with Prometheus & Grafana; experiment tracking with MLflow.

**Disclaimer:** For research only. Crypto trading involves substantial risk.
