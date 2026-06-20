#!/usr/bin/env python3
"""
Backtest an EPA-based betting strategy using nflreadpy.

Corrected Data Conventions:
- spread_line > 0: Home Team is a Favorite
- spread_line < 0: Home Team is an Underdog
- Scoring Margin = Home Score - Away Score
- Home Covers if: Scoring Margin > Closing Spread Line

Usage:
    python backtest_epa_strategy.py
"""

from typing import List, Optional
import pandas as pd
import nflreadpy as nfl

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data(seasons: List[int]):
    print(f"Loading weekly team stats & schedule data for seasons: {seasons} via nflreadpy")
    # Pull data as Polars DataFrames and convert immediately to Pandas
    weekly = nfl.load_team_stats(seasons, summary_level="week").to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()
    return weekly, schedules


def compute_team_point_in_time_epa(weekly_df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    epa_candidates = ["offense_epa", "off_epa", "team_epa", "epa", "offensive_epa"]
    team_candidates = ["team", "team_abbr", "team_name", "posteam"]

    epa_col = find_column(weekly_df, epa_candidates)
    team_col = find_column(weekly_df, team_candidates)

    if epa_col and team_col:
        print(f"Using weekly data EPA column '{epa_col}' and team column '{team_col}'.")
        team_epa = weekly_df[["season", "week", team_col, epa_col]].rename(
            columns={team_col: "team", epa_col: "team_epa"}
        )
    else:
        print("No EPA column found in weekly data — calculating from play-by-play aggregation...")
        pbp = nfl.load_pbp(seasons).to_pandas()
        pbp = pbp.dropna(subset=["posteam", "epa"])  
        team_epa = (
            pbp.groupby(["season", "week", "posteam"], as_index=False)["epa"].sum()
            .rename(columns={"posteam": "team", "epa": "team_epa"})
        )

    team_epa["team"] = team_epa["team"].astype(str).str.upper().str.strip()
    team_epa["season"] = team_epa["season"].astype(int)
    team_epa["week"] = team_epa["week"].astype(int)

    # Rolling 4-week mean shifted by 1 week to avoid look-ahead bias
    team_epa = team_epa.sort_values(["team", "season", "week"])  
    team_epa["epa_rolling_4"] = team_epa.groupby(["team", "season"])["team_epa"].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    team_epa["epa_pt"] = team_epa.groupby(["team", "season"])["epa_rolling_4"].shift(1)

    return team_epa[["season", "week", "team", "epa_pt"]]


def run_backtest(seasons: List[int]):
    weekly, schedules = load_data(seasons)
    schedules = schedules.copy()
    
    schedules["season"] = schedules["season"].astype(int)
    schedules["week"] = schedules["week"].astype(int)

    home_col = find_column(schedules, ["home_team", "home"])
    away_col = find_column(schedules, ["away_team", "away"])
    spread_col = find_column(schedules, ["spread_line", "spread"]) 
    home_score_col = find_column(schedules, ["home_score", "home_points"])
    away_score_col = find_column(schedules, ["away_score", "away_points"])

    schedules[home_col] = schedules[home_col].astype(str).str.upper().str.strip()
    schedules[away_col] = schedules[away_col].astype(str).str.upper().str.strip()

    schedules_clean = schedules.dropna(subset=[home_score_col, away_score_col, spread_col])
    schedules_clean[home_score_col] = pd.to_numeric(schedules_clean[home_score_col], errors="coerce")
    schedules_clean[away_score_col] = pd.to_numeric(schedules_clean[away_score_col], errors="coerce")
    schedules_clean = schedules_clean.dropna(subset=[home_score_col, away_score_col])

    team_epa_pt = compute_team_point_in_time_epa(weekly, seasons)
    
    home_stats = team_epa_pt.rename(columns={"team": home_col, "epa_pt": "epa_home"})
    away_stats = team_epa_pt.rename(columns={"team": away_col, "epa_pt": "epa_away"})

    merged = schedules_clean.merge(
        home_stats[["season", "week", home_col, "epa_home"]], on=["season", "week", home_col], how="left"
    ).merge(
        away_stats[["season", "week", away_col, "epa_away"]], on=["season", "week", away_col], how="left"
    )

    merged["epa_diff"] = merged["epa_home"] - merged["epa_away"]

    # -------------------------------------------------------------------------
    # STRATEGY TOGGLES (Uncomment the one you want to test!)
    # -------------------------------------------------------------------------
    # System A: Home UNDERDOGS with an EPA edge (spread_line < 0)
    # bets = merged[(merged["epa_diff"] > 5.0) & (merged[spread_col] < 0)].copy()
    
    # System B: Home FAVORITES with an EPA edge (spread_line > 0)
    bets = merged[(merged["epa_diff"] > 5.0) & (merged[spread_col] > 0)].copy()
    # -------------------------------------------------------------------------

    if bets.empty:
        print("No historical matches found for these parameters.")
        return

    bets["home_score"] = bets[home_score_col].astype(float)
    bets["away_score"] = bets[away_score_col].astype(float)
    bets["spread"] = bets[spread_col].astype(float)

    # Elegant nflverse margin calculation
    bets["scoring_margin"] = bets["home_score"] - bets["away_score"]
    bets["cover"] = bets["scoring_margin"] > bets["spread"]
    bets["push"] = bets["scoring_margin"] == bets["spread"]

    total_bets = len(bets)
    pushes = int(bets["push"].sum())
    wins = int(bets["cover"].sum())
    losses = total_bets - wins - pushes

    profit = wins * 100 - losses * 110
    ats_win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    print("--- Corrected Backtest Results ---")
    print(f"Seasons: {seasons}")
    print(f"Total Bets Placed: {total_bets}")
    print(f"Record (W-L-P): {wins}-{losses}-{pushes}")
    print(f"ATS Win Rate (ex-pushes): {ats_win_rate:.2f}%")
    print(f"Net Profit/Loss (Assuming -110): ${profit:,}")


if __name__ == "__main__":
    run_backtest([2022, 2023])