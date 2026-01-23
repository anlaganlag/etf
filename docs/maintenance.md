# Project Maintenance Guide

This document provides technical details for maintaining and extending the A-Share ETF Selection System.

## 🏗️ Core Logic Overview

### ETF Scoring Model
The system uses a multi-factor momentum score. The score is calculated by checking if an ETF's performance over various windows (1d, 3d, 5d, 10d, 20d) ranks within the top `N` of the entire market.
Weights are defined in `config.py`:
- `r1`: 100 points
- `r3`: 70 points
- `r5`: 50 points
- `r10`: 30 points
- `r20`: 20 points

### Data Pipeline
1. **Source**: Primarily use `AkShare` (Sina) for real-time and historical ETF data.
2. **Fallback**: `Baostock` is used for basic list fetching if Sina fails.
3. **Caching**: Data is cached in `data_cache/` in CSV format. ETF daily data is refreshed if older than 12 hours.

## 🔧 Maintenance Tasks

### Adding a New Data Source
Modify `src/data_fetcher.py`. Implement a new method or update `_safe_fetch` to handle new API providers.

### Modifying Strategy Factors
Update `src/etf_ranker.py`. The `_calculate_momentum_score` method is where you can add new calculation logic (e.g., volume profile, RSI).

### Running Backtests
Use the scripts in `scripts/backtest/`. 
- `backtest_strategy.py`: Basic weekly rotation simulation.
- Ensure that any script you run uses `from config import config` to correctly resolve paths.

## 📁 Output Management
- **Reports**: Always direct generated reports to `output/reports/`.
- **Data**: Intermediary or final CSV files go to `output/data/`.
- **Cleanup**: The `output/` directory and `data_cache/` can be deleted safely; they will be regenerated on the next run.

## 🛡️ Best Practices
1. **Avoid Hardcoded Paths**: Always use `config.BASE_DIR` or other defined paths in `config.py`.
2. **Serial Execution**: The current system is optimized for stability over speed using serial fetching to avoid IP blocks from Sina.
3. **Check Logs**: If an ETF list is missing names, it usually means the Sina fetch failed and it fell back to Baostock. Check your internet connection or Sina API availability.
