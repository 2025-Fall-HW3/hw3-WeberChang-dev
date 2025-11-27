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
        SIMPLE BUT EFFECTIVE: MOMENTUM + LOW VOLATILITY + TREND FILTER
        Stop overcomplicating. Focus on:
        1. Strong 6-month momentum
        2. Confirmed uptrend (3M also positive)
        3. Reasonable concentration
        4. NO LEVERAGE (it's hurting us)
        """
        
        mom_primary = 126        # 6-month momentum (sweet spot)
        mom_confirm = 63         # 3-month trend confirmation
        vol_window = 126         # 6-month volatility
        
        top_n = 5                # Hold top 5 sectors
        max_weight = 0.35        # Allow concentration
        min_weight = 0.10        # Meaningful positions only
        
        self.portfolio_weights.loc[:, :] = 0.0
        n = len(assets)
        
        for t in range(mom_primary, len(self.price)):
            current_date = self.price.index[t]
            
            # 6-month momentum (PRIMARY SIGNAL)
            p_6m = self.price[assets].iloc[t - mom_primary]
            p_now = self.price[assets].iloc[t]
            mom_6m = (p_now / p_6m - 1.0).values
            
            # 3-month momentum (CONFIRMATION)
            p_3m = self.price[assets].iloc[t - mom_confirm]
            mom_3m = (p_now / p_3m - 1.0).values
            
            # BOTH must be positive (trend filter)
            valid_mask = (mom_6m > 0) & (mom_3m > 0)
            
            # Volatility
            ret_window = self.returns.iloc[t-vol_window:t][assets]
            vols = ret_window.std().values * np.sqrt(252)
            vols = np.maximum(vols, 0.05)
            
            # Sharpe-like score
            sharpe_score = mom_6m / vols
            sharpe_score = sharpe_score * valid_mask  # Zero out invalid
            
            # Need at least 3 positive, otherwise take best available
            if np.sum(sharpe_score > 0) >= 3:
                top_idx = np.argsort(sharpe_score)[-top_n:]
                top_idx = top_idx[sharpe_score[top_idx] > 0]
            else:
                # Take top 5 by momentum regardless
                top_idx = np.argsort(mom_6m)[-top_n:]
            
            # WEIGHTING: Proportional to signal strength
            scores = sharpe_score[top_idx]
            
            if np.all(scores <= 0):
                # All negative, use momentum directly
                scores = mom_6m[top_idx]
                scores = np.maximum(scores, 0.01)
            else:
                scores = np.maximum(scores, 0.01)
            
            # Linear weighting (not quadratic - too aggressive)
            weights = scores / scores.sum()
            
            # Apply limits
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
            # Build portfolio
            w = np.zeros(n)
            w[top_idx] = weights
            
            # NO LEVERAGE - just normalize to sum = 1.0
            if w.sum() > 0:
                w = w / w.sum()
            
            # Store
            w_series = pd.Series(0.0, index=self.price.columns)
            w_series[assets] = w
            self.portfolio_weights.loc[current_date, :] = w_series
        
        if self.exclude in self.portfolio_weights.columns:
            self.portfolio_weights[self.exclude] = 0.0
        
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
