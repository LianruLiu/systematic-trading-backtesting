# Systematic Trading Strategy Development and Backtesting

Two strategies backtested on real daily data (Yahoo Finance, adjusted closes),
with transaction costs and an in-sample / out-of-sample split. Benchmark: SPY buy-and-hold.

## Strategies

- **MA crossover (SPY):** long SPY when the 50-day moving average is above the
  200-day moving average, otherwise flat in cash. Signal applied next day (no look-ahead).
- **Cross-sectional momentum:** at each month-end, rank a fixed 20-stock universe by
  12-1 month return, hold the top 5 equal-weighted through the next month, rebalance monthly.

Universe (20 large caps): AAPL, MSFT, NVDA, AMZN, META, GOOGL, AVGO, JPM, UNH, JNJ,
XOM, WMT, MA, V, PG, HD, CVX, LLY, MRK, ABBV.

## Costs and assumptions

- 10 bps one-way on traded notional, charged on each rebalance / signal-change day.
- Risk-free rate assumed 0 for Sharpe ratios (stated explicitly).
- No borrow costs, no market impact, no intraday data.

## Results (data 2015-01-02 to 2026-10-30; run `backtest.py` to reproduce)

| Strategy | Period | Ann. return | Ann. vol | Sharpe (rf=0) | Max drawdown | Monthly win rate |
|---|---|---|---|---|---|---|
| SPY buy-and-hold | 2015–2019 | 11.60% | 13.43% | 0.864 | -19.35% | 76.7% |
| SPY buy-and-hold | 2020–2026 (OOS) | 14.94% | 19.96% | 0.749 | -33.72% | 65.9% |
| MA crossover | 2015–2019 | 7.13% | 10.24% | 0.696 | -13.70% | 58.3% |
| MA crossover | 2020–2026 (OOS) | 9.67% | 16.75% | 0.577 | -33.72% | 53.7% |
| Momentum top-5 | 2015–2019 | 23.39% | 17.68% | 1.323 | -31.67% | 60.0% |
| Momentum top-5 | 2020–2026 (OOS) | 35.75% | 27.41% | 1.304 | -33.03% | 65.9% |

## Read the momentum number honestly

The momentum out-of-sample Sharpe of 1.30 is **inflated by survivorship bias**:
the universe is today's large-cap winners, selected with hindsight, and 2020–2026
was a mega-cap momentum regime. A point-in-time universe (historical index
constituents) would give a lower, more credible number. Do not quote 1.30 as a
strategy's expected Sharpe. The MA crossover result is the cleaner lesson: it
cut volatility versus buy-and-hold but gave up return, and it did not avoid the
2020 drawdown.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backtest.py
```

Outputs go to `results/`: `results.json`, `equity_curves.csv`, `equity_curves.png`,
and the downloaded `prices.csv`.

## Limitations

- Survivorship bias in the momentum universe (see above).
- Flat 10 bps cost is an assumption, not a measured slippage model.
- Past performance does not predict future returns.
