"""
Project 1: Systematic Trading Strategy Development and Backtesting.

Strategies (daily data, adjusted closes):
  A) MA crossover on SPY: long SPY when 50-day MA > 200-day MA, else flat (cash).
  B) Cross-sectional momentum: at each month-end, rank a fixed universe by
     12-1 month return, hold the top 5 equal-weighted through the next month,
     rebalance monthly.

Costs: 10 bps one-way on traded notional, charged in full on each rebalance /
signal-change day. Risk-free rate assumed 0 for Sharpe (stated, not hidden).

Split: in-sample 2015-2019, out-of-sample 2020-2026-10. Benchmark: SPY buy-and-hold.

Known limitations (read before quoting any number):
  - The stock universe is today's large caps, so momentum results carry
    survivorship bias. This is a learning backtest, not a production signal.
  - No intraday data, no borrow costs, no market impact; costs are a flat assumption.
  - Past performance does not predict future returns.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "JPM", "UNH",
    "JNJ", "XOM", "WMT", "MA", "V", "PG", "HD", "CVX", "LLY", "MRK", "ABBV",
]
BENCH = "SPY"
START, END = "2015-01-01", "2026-10-31"
COST_BPS = 10  # one-way, on traded notional
TOP_N = 5


def fetch() -> pd.DataFrame:
    px = yf.download(UNIVERSE + [BENCH], start=START, end=END,
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all").ffill().dropna()
    px.to_csv(OUT / "prices.csv")
    return px


def metrics(rets: pd.Series, turnover_annual: float) -> dict:
    rets = rets.dropna()
    n_years = len(rets) / 252
    ann_ret = (1 + rets).prod() ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    win = float((rets.resample("ME").sum() > 0).mean())  # fraction of positive months
    return {
        "annualized_return": round(float(ann_ret), 4),
        "annualized_volatility": round(float(ann_vol), 4),
        "sharpe_rf0": round(float(sharpe), 3),
        "max_drawdown": round(float(dd.min()), 4),
        "monthly_win_rate": round(win, 3),
        "annualized_turnover": round(float(turnover_annual), 3),
        "n_days": int(len(rets)),
    }


def ma_crossover(spy: pd.Series):
    ma50 = spy.rolling(50).mean()
    ma200 = spy.rolling(200).mean()
    pos = (ma50 > ma200).astype(float).shift(1).fillna(0.0)  # trade next day, no look-ahead
    strat_rets = pos * spy.pct_change()
    trades = pos.diff().abs().fillna(0.0)
    cost = trades * COST_BPS / 10_000
    turnover_annual = float(trades.sum() / (len(spy) / 252))
    return strat_rets - cost, turnover_annual


def momentum(px: pd.DataFrame):
    stocks = px[UNIVERSE]
    daily = stocks.pct_change()
    monthly = stocks.resample("ME").last()
    # 12-1 month momentum: 12m return skipping the most recent month
    mom = monthly.pct_change(12).shift(1)
    periods = daily.index.to_period("M").unique()
    strat = pd.Series(0.0, index=daily.index)
    cost = pd.Series(0.0, index=daily.index)
    prev_w = pd.Series(0.0, index=UNIVERSE)
    turnover_total = 0.0
    for i, per in enumerate(periods):
        # picks decided at the end of the previous month
        if i == 0:
            continue
        prev_month_end = monthly.index[monthly.index.to_period("M") == periods[i - 1]]
        if len(prev_month_end) == 0:
            continue
        rank = mom.loc[prev_month_end[0]].dropna()
        if len(rank) < TOP_N:
            continue
        tickers = rank.nlargest(TOP_N).index.tolist()
        w = pd.Series(0.0, index=UNIVERSE)
        w[tickers] = 1.0 / len(tickers)
        days = daily.index[daily.index.to_period("M") == per]
        if len(days) == 0:
            continue
        turn = float((w - prev_w).abs().sum())
        turnover_total += turn
        cost.loc[days[0]] = turn * COST_BPS / 10_000  # charged on rebalance day
        strat.loc[days] = daily.loc[days, UNIVERSE].mul(w, axis=1).sum(axis=1)
        prev_w = w
    turnover_annual = turnover_total / (len(daily) / 252)
    return strat - cost, turnover_annual


def main() -> None:
    px = fetch()
    spy_rets = px[BENCH].pct_change()
    ma_rets, ma_turn = ma_crossover(px[BENCH])
    mom_rets, mom_turn = momentum(px)

    def split(s: pd.Series, turn: float):
        return {
            "in_sample_2015_2019": metrics(s.loc["2015":"2019"], turn),
            "out_of_sample_2020_2026": metrics(s.loc["2020":], turn),
            "full": metrics(s, turn),
        }

    results = {
        "universe": UNIVERSE,
        "benchmark": BENCH,
        "data_range": [str(px.index[0].date()), str(px.index[-1].date())],
        "cost_bps_one_way": COST_BPS,
        "risk_free_rate_assumed": 0.0,
        "SPY_buy_hold": split(spy_rets, 0.0),
        "MA_crossover_SPY_50_200": split(ma_rets, ma_turn),
        "Momentum_top5_12minus1": split(mom_rets, mom_turn),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    equity = pd.DataFrame({
        "SPY": (1 + spy_rets.fillna(0)).cumprod(),
        "MA_crossover": (1 + ma_rets.fillna(0)).cumprod(),
        "Momentum_top5": (1 + mom_rets.fillna(0)).cumprod(),
    })
    equity.to_csv(OUT / "equity_curves.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax)
    ax.set_title("Growth of $1 — strategies vs SPY buy-and-hold (after 10bps costs)")
    ax.set_ylabel("Growth of $1")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=150)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
