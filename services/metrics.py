import numpy as np
import pandas as pd


def process_advanced_differentials(schedules, team_stats):
    team_stats = team_stats.copy()
    team_stats["team"] = team_stats["team"].astype(str).str.upper().str.strip()
    
    if "passing_epa" not in team_stats.columns: team_stats["passing_epa"] = 0.0
    if "rushing_epa" not in team_stats.columns: team_stats["rushing_epa"] = 0.0
    team_stats["epa"] = team_stats["passing_epa"] + team_stats["rushing_epa"]
    
    if "passing_yards" in team_stats.columns and "rushing_yards" in team_stats.columns:
        total_yds = team_stats["passing_yards"] + team_stats["rushing_yards"]
        total_plays = team_stats.get("attempts", 0) + team_stats.get("carries", 0) + team_stats.get("sacks_suffered", 0)
        team_stats["y_per_play"] = np.where(total_plays > 0, total_yds / total_plays, 5.4)
    else:
        if "y_per_play" not in team_stats.columns: team_stats["y_per_play"] = 5.4
        
    if "success_rate" not in team_stats.columns:
        team_stats["success_rate"] = (45.0 + (team_stats["epa"] * 12.0)).clip(35.0, 65.0)
        
    if "sacks_suffered" in team_stats.columns and "attempts" in team_stats.columns:
        dropbacks = team_stats["attempts"] + team_stats["sacks_suffered"]
        team_stats["sack_rate"] = np.where(dropbacks > 0, (team_stats["sacks_suffered"] / dropbacks) * 100.0, 6.0)
    else:
        if "sack_rate" not in team_stats.columns: team_stats["sack_rate"] = 6.0
        
    if "turnovers" not in team_stats.columns:
        ints = team_stats.get("passing_interceptions", 0)
        fumbles = team_stats.get("rushing_fumbles_lost", 0) + team_stats.get("sack_fumbles_lost", 0)
        team_stats["turnovers"] = ints + fumbles
        
    if "red_zone_pct" not in team_stats.columns:
        team_stats["red_zone_pct"] = (52.0 + (team_stats["epa"] * 8.0)).clip(40.0, 75.0)

    metrics = ["epa", "y_per_play", "success_rate", "sack_rate", "turnovers", "red_zone_pct"]
    team_profiles = team_stats.sort_values(["team", "week"]).copy()
    
    for m in metrics:
        team_profiles[f"{m}_roll"] = team_profiles.groupby("team")[m].transform(lambda x: x.rolling(4, min_periods=1).mean())
        team_profiles[f"{m}_prev"] = team_profiles.groupby("team")[f"{m}_roll"].shift(1).fillna(team_profiles[f"{m}_roll"])
        
    return team_profiles
