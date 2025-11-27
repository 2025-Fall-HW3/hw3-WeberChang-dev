"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # Parameters for the enhanced strategy
        mom_lookback = 252                    # 12-month momentum
        cov_lookback = max(self.lookback, 60) # at least 60 days for covariance
        shrink_alpha = 0.2                    # shrinkage intensity (0=no shrink, 1=full shrink)
        target_vol = 0.10                     # target annual vol (10%)
        max_leverage = 2.0                    # cap on leverage
        max_weight = 0.30                     # per-asset maximum weight (before leverage)
        smoothing = 0.85                      # weight smoothing (0=no smoothing, 1=full hold)
        
        # initialize rows as zeros
        self.portfolio_weights.loc[:, :] = 0.0

        prev_w = pd.Series(0.0, index=self.price.columns)

        n = len(assets)
        for t in range(cov_lookback, len(self.price)):
            current_date = self.price.index[t]

            # rolling windows for returns used in cov / vol
            window_returns = self.returns.iloc[t-cov_lookback : t][assets]

            # sample cov and vol
            cov = window_returns.cov().values
            vol = window_returns.std()

            # shrink covariance toward avg variance * I (Ledoit-Wolf style simple shrink)
            avg_var = np.mean(np.diag(cov))
            cov_shrunk = (1 - shrink_alpha) * cov + shrink_alpha * (np.eye(n) * avg_var)

            # momentum signal (12-month simple return). If too short, use available.
            if t >= mom_lookback:
                past_price = self.price[assets].iloc[t - mom_lookback]
                recent_price = self.price[assets].iloc[t]
                mom = (recent_price.values / past_price.values) - 1.0
            else:
                # short series: use lookback returns as proxy
                mom = window_returns.mean().values * self.lookback

            # respect NaNs
            mom = np.nan_to_num(mom, nan=0.0)

            # zero-out negative momentum (long-only momentum filter)
            mom_signal = np.maximum(mom, 0.0)

            # if all signals are zero (no positive momentum), fallback to inverse-vol weighting
            if np.all(mom_signal == 0):
                raw_w = 1.0 / np.maximum(vol.values, 1e-8)
                raw_w = np.nan_to_num(raw_w)
            else:
                # Tangency direction: solve Cov^{-1} * signal
                try:
                    x = np.linalg.solve(cov_shrunk, mom_signal)
                except np.linalg.LinAlgError:
                    # fallback to pseudo-inverse
                    x = np.dot(np.linalg.pinv(cov_shrunk), mom_signal)

                # force negatives to zero (long-only)
                x = np.maximum(x, 0.0)
                raw_w = x

                # if numerical issues lead to all zeros, fallback again
                if np.all(raw_w == 0.0):
                    raw_w = 1.0 / np.maximum(vol.values, 1e-8)

            # normalize (pre-cap)
            if raw_w.sum() <= 0:
                w = np.ones_like(raw_w) / len(raw_w)
            else:
                w = raw_w / raw_w.sum()

            # enforce max weight cap and renormalize (iterate simple clipping)
            w = np.minimum(w, max_weight)
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()

            # compute current portfolio volatility (annualized)
            port_var = float(np.dot(w, np.dot(cov_shrunk, w)))
            port_vol = np.sqrt(port_var) * np.sqrt(252)  # annualize: sqrt(252)

            # volatility targeting (apply leverage)
            if port_vol > 0:
                lev = target_vol / port_vol
            else:
                lev = 1.0
            lev = min(lev, max_leverage)

            w = w * lev

            # map to DataFrame index order and apply smoothing with previous weights
            w_series = pd.Series(0.0, index=self.price.columns)
            w_series[assets] = w

            # smoothing to reduce turnover
            w_series = smoothing * prev_w + (1 - smoothing) * w_series

            # final safety: set negatives to zero and renormalize (shouldn't be negative)
            w_series[w_series < 0] = 0.0
            s = w_series.sum()
            if s > 0:
                w_series = w_series / s * min(1.0, s)  # preserve leverage proportionally if >1

            # assign and update prev
            self.portfolio_weights.loc[current_date, :] = w_series
            prev_w = w_series.copy()

        # ensure excluded ETF has zero weight
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights[self.exclude] = 0.0

        # forward fill and fill NA
        # (original code will ffill + fillna after this block)

        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
