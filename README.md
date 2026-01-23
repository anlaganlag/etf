# A-Share ETF Selection System (V2.0)

A modular, automated system for selecting and ranking A-Share market ETFs based on multi-period momentum and sector relative strength.

## 📁 Project Structure

```text
├── main.py                 # Main entry point for ETF analysis
├── config.py               # Central configuration and path management
├── src/                    # Core logic and modules
│   ├── data_fetcher.py     # Market data wrappers (AkShare, Baostock)
│   ├── etf_ranker.py       # ETF scoring and ranking logic
│   ├── sector_ranker.py    # Sector identification and ranking
│   └── etf_mapper.py       # (Optional) Sector-to-ETF mapping logic
├── scripts/                # Utility and research scripts
│   ├── backtest/           # Historical performance simulation
│   ├── analysis/           # Strategy optimization and insight generation
│   └── data/               # Data maintenance and preprocessing
├── output/                 # Generated results (not in git)
│   ├── data/               # CSV, Excel files
│   ├── reports/            # Markdown, HTML analysis reports
│   └── charts/             # Visualization curves and plots
├── docs/                   # Documentation and legacy records
│   └── legacy_reports/     # Previous analysis results
└── data_cache/             # Local CSV cache for market data
```

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running Daily Analysis
```bash
python main.py
```
This will:
1. Fetch latest ETF list.
2. Calculate scores based on r1, r3, r5, r10, r20 momentum.
3. Select the top 10 strongest ETFs (one per sector).
4. Save results to `output/data/top_10_etfs.csv`.

## 🛠️ Configuration

Edit `config.py` to adjust:
- `SECTOR_TOP_N_THRESHOLD`: Ranking cut-off for receiving score.
- `SECTOR_PERIOD_SCORES`: Weights for different reward periods.
- `ETF_SECTOR_LIMIT`: Diversification constraint.

## 📊 Documentation
For a deep dive into the system logic and maintenance procedures, see [docs/maintenance.md](file:///Users/randy/Documents/code/akshare/docs/maintenance.md).
