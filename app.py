import streamlit as st
import pandas as pd
import nflreadpy as nfl
import plotly.express as px

st.set_page_config(page_title="BillionBetting NFL Model", layout="wide")

# -------------------------------------------------------------------------
# PHASE 2: AUTOMATED DATA PIPELINE (Cached for Speed)
# -------------------------------------------------------------------------
@st.cache_data
def load_model_data(seasons):
    # Pull schedules first (the upcoming season schedule is usually available)
    schedules = nfl.load_schedules(seasons).to_pandas()
    
    try:
        # Try pulling weekly team stats
        weekly = nfl.load_team_stats(seasons, summary_level="week").to_pandas()
    except Exception:
        # Fallback for future/upcoming seasons before any games are played
        st.sidebar.info(f"📅 Note: No game stats recorded yet for {seasons}. Teams initialized to baseline 0.0 EPA.")
        weekly = pd.DataFrame(columns=["season", "week", "team", "passing_epa", "rushing_epa"])
        
    return weekly, schedules

st.title("🏈 BillionBetting NFL Model & Dashboard")
st.markdown("Automated spread/total forecasting engine with situational context adjustments.")
st.divider()  # Fixes the st.hr() AttributeError safely here

# Sidebar Configuration Panel
st.sidebar.header("🔄 Model Controls")
selected_season = st.sidebar.selectbox("Select Season", [2023, 2024, 2025, 2026], index=0)
hfa_adjustment = st.sidebar.slider("Home Field Advantage (Points)", 0.0, 4.0, 2.5, step=0.5)
epa_multiplier = st.sidebar.slider("EPA to Points Multiplier", 1.0, 5.0, 3.0, step=0.1)

if st.sidebar.button("🔄 REFRESH ALL DATA"):
    st.cache_data.clear()
    st.sidebar.success("Pipeline refreshed successfully!")

# Load Data
weekly_df, schedules_df = load_model_data([selected_season])

# Clean and calculate rolling 4-week EPA
weekly_df["team"] = weekly_df["team"].astype(str).str.upper().str.strip()

# FIX: Synthesize total offense_epa from passing and rushing components
weekly_df["offense_epa"] = weekly_df["passing_epa"] + weekly_df["rushing_epa"]

weekly_df = weekly_df.sort_values(["team", "season", "week"])
weekly_df["epa_rolling"] = weekly_df.groupby(["team", "season"])["offense_epa"].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)
# Shift by 1 week to avoid look-ahead bias
weekly_df["epa_pt"] = weekly_df.groupby(["team", "season"])["epa_rolling"].shift(1)

# Filter schedules for regular season matches with lines
schedules_clean = schedules_df.dropna(subset=["spread_line", "total_line"]).copy()
schedules_clean["home_team"] = schedules_clean["home_team"].astype(str).str.upper().str.strip()
schedules_clean["away_team"] = schedules_clean["away_team"].astype(str).str.upper().str.strip()

# Merge point-in-time stats into schedule
home_stats = weekly_df[["season", "week", "team", "epa_pt"]].rename(columns={"team": "home_team", "epa_pt": "epa_home"})
away_stats = weekly_df[["season", "week", "team", "epa_pt"]].rename(columns={"team": "away_team", "epa_pt": "epa_away"})

merged = schedules_clean.merge(home_stats, on=["season", "week", "home_team"], how="left")
merged = merged.merge(away_stats, on=["season", "week", "away_team"], how="left")
merged = merged.fillna(0) # Handle early season missing rolling data

# -------------------------------------------------------------------------
# PHASE 1 & 4: CALCULATIONS & CONTEXT ENGINE
# -------------------------------------------------------------------------
# Baseline Model Predictions
merged["base_predicted_spread"] = (merged["epa_home"] - merged["epa_away"]) * epa_multiplier + hfa_adjustment
merged["base_predicted_total"] = merged["total_line"] # Baseline tracks market before context adjustments

# Interactive Context Engine Adjustments Section
st.header("🌦️ Step 2: Weekly Context & Situational Adjustments")
st.markdown("Select live game conditions to apply automated weather and injury downgrades.")

selected_week = st.selectbox("Select Week to Modify / Analyze", sorted(merged["week"].unique()))
week_games = merged[merged["week"] == selected_week].copy()

adjusted_games = []
for idx, row in week_games.iterrows():
    matchup_label = f"{row['away_team']} @ {row['home_team']}"
    
    with st.expander(f"⚙️ Adjust: {matchup_label}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            wind = st.slider(f"Wind Speed (MPH) - {row['home_team']}", 0, 40, 0, key=f"wind_{idx}")
            weather_precip = st.checkbox("Heavy Rain / Snow", key=f"precip_{idx}")
        with col2:
            home_qb_out = st.checkbox("Home QB1 Out", key=f"hqb_{idx}")
            home_lt_out = st.checkbox("Home Starting LT Out", key=f"hlt_{idx}")
        with col3:
            away_qb_out = st.checkbox("Away QB1 Out", key=f"aqb_{idx}")
            away_lt_out = st.checkbox("Away Starting LT Out", key=f"alt_{idx}")
            
        # Apply Phase 4 Rules Dynamically
        total_adj = 0.0
        spread_adj = 0.0
        
        if wind >= 20: total_adj -= 3.0
        elif wind >= 15: total_adj -= 1.5
        if weather_precip: total_adj -= 1.5
        
        if home_qb_out: spread_adj -= 6.0
        if home_lt_out: spread_adj -= 0.75
        if away_qb_out: spread_adj += 6.0
        if away_lt_out: spread_adj += 0.75
        
        row["final_predicted_spread"] = row["base_predicted_spread"] + spread_adj
        row["final_predicted_total"] = row["base_predicted_total"] + total_adj
        adjusted_games.append(row)

final_week_df = pd.DataFrame(adjusted_games)

# -------------------------------------------------------------------------
# CONFIDENCE MODEL & EDGES
# -------------------------------------------------------------------------
# Spread Edge (Home perspective convention matching your backtester)
final_week_df["spread_edge"] = final_week_df["final_predicted_spread"] - final_week_df["spread_line"]
final_week_df["absolute_edge"] = final_week_df["spread_edge"].abs()

# Automated Confidence & Tier Assignments
def assign_tier(row):
    if row["absolute_edge"] <= 1.0: return "NO BET", 0
    elif row["absolute_edge"] <= 2.0: return "Tier 1", 79
    elif row["absolute_edge"] <= 3.0: return "Tier 1.5", 85
    else: return "Tier 2", 92

tiers_and_scores = final_week_df.apply(assign_tier, axis=1)
final_week_df["Tier"] = [t[0] for t in tiers_and_scores]
final_week_df["Confidence"] = [t[1] for t in tiers_and_scores]

# -------------------------------------------------------------------------
# PHASE 3: GRAPHICAL OUTPUTS & BILLIONBETTING CARD
# -------------------------------------------------------------------------
st.write("---")
st.header("📊 Step 3 & 4: Model Projections & Visual Edge Analytics")

# Data Visualization Graph
fig = px.bar(
    final_week_df,
    x="home_team",
    y="spread_edge",
    color="Tier",
    title="Model Betting Edges by Home Team (Positive = Value on Home, Negative = Value on Away)",
    labels={"spread_edge": "Model Advantage vs Market Line", "home_team": "Home Team"},
    hover_data=["away_team", "spread_line", "final_predicted_spread"],
    barmode="group"
)
st.plotly_chart(fig, use_container_width=True)

# Clean Display Output for the Final Betting Card Table
display_cols = [
    "away_team", "home_team", "spread_line", "final_predicted_spread", 
    "spread_edge", "total_line", "final_predicted_total", "Tier", "Confidence"
]
formatted_card = final_week_df[display_cols].rename(columns={
    "spread_line": "Market Spread",
    "final_predicted_spread": "Model Spread",
    "spread_edge": "Raw Edge",
    "total_line": "Market Total",
    "final_predicted_total": "Model Total"
})

st.subheader("📋 Complete Weekly Projection Board")
st.dataframe(
    formatted_card.style.background_gradient(subset=["Raw Edge"], cmap="RdYlGn"),
    use_container_width=True
)

st.subheader("🔥 Top Qualified BillionBetting System Play Candidates")
bets_only = formatted_card[formatted_card["Tier"] != "NO BET"].sort_values(by="Raw Edge", key=abs, ascending=False)
if not bets_only.empty:
    st.table(bets_only)
else:
    st.info("No games met the Edge > 1.0 point threshold requirement for this system configuration.")