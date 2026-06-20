import streamlit as st
import pandas as pd
import numpy as np
import nflreadpy as nfl
import plotly.express as px
import os
from datetime import datetime
from services.pipeline import fetch_and_build_pipeline
from services.metrics import process_advanced_differentials
from storage.tracker import TRACKER_FILE, init_tracker
from ui.style import apply_styles

# -------------------------------------------------------------------------
# STYLING & CONFIGURATION (moved to ui/style.py)
# -------------------------------------------------------------------------
apply_styles()

# -------------------------------------------------------------------------
# PERSISTENT STORAGE INITIALIZATION
# -------------------------------------------------------------------------
if "model_runs" not in st.session_state:
    st.session_state.model_runs = {}
if "selected_week" not in st.session_state:
    st.session_state.selected_week = 1

# initialize tracker file (creates CSV if missing)
init_tracker()

# Pipeline implementation moved to pipeline.py (imported as fetch_and_build_pipeline)

# Advanced metric engine moved to metrics.py (imported as process_advanced_differentials)

# -------------------------------------------------------------------------
# SIDEBAR CONTROL DECK
# -------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Engine Controls & Settings")
    st.caption("Configure mode, season, and model-level power settings used throughout the workspace.")

    # MASTER MODE SWITCH
    app_mode = st.radio(
        "Application Mode",
        ["🔬 Research & Backtesting", "🏈 Live Weekly Operations"],
        index=0,
        help="Choose 'Research & Backtesting' for historical validation, or 'Live Weekly Operations' for the active weekly workflow."
    )
    
    st.divider()
    target_season = st.selectbox("Target Season", [2023, 2024, 2025, 2026], index=3, help="Season to pull or simulate schedules and stats for.")
    
    with st.container(border=True):
        st.markdown("**Global Model Power Settings**")
        st.caption("Adjust global parameters that influence model sensitivity and home-field bias.")
        hfa_input = st.slider("Home Field Advantage (pts)", 0.0, 4.0, 2.5, step=0.25, help="Estimate (in points) added to home team advantage when computing predicted spreads.")
        scale_input = st.slider("Model Intensity Scale Factor", 1.0, 5.0, 3.2, step=0.1, help="Multiplier that scales the composite differential to produce the model spread. Higher values make the model more aggressive.")
        
    if st.button("🔄 UPDATE NFL DATA / REFRESH PIPELINES", use_container_width=True):
        st.cache_data.clear()
        st.toast("Power Query Automations Completed. All Lines Re-Indexed!", icon="🔄")
    st.caption("Click to clear cached data and refresh the schedules/stats pipelines. This may take a moment for large seasons.")

# Pipeline Processing Execution
raw_sched, raw_stats = fetch_and_build_pipeline(target_season)
processed_profiles = process_advanced_differentials(raw_sched, raw_stats)


# =========================================================================
# ENVIRONMENT ENVIRONMENT CONTROLLER
# =========================================================================

# -------------------------------------------------------------------------
# MODE 1: HISTORICAL RESEARCH & BACKTESTING ENGINE
# -------------------------------------------------------------------------
if app_mode == "🔬 Research & Backtesting":
    st.title("🔬 Automated Historical Backtesting & Validation Lab")
    st.caption("Use this lab to run bulk historical backtests, validate model thresholds, and inspect raw outcomes vs. model predictions.")
    st.markdown("This workspace evaluates pure statistical baseline metrics across historical data to isolate predictive performance without any manual intervention.")

    # Macro Execution Form
    with st.form("backtest_form"):
        st.markdown("### 🏃‍♂️ Configure Bulk Backtest Run")
        st.caption("Select an inclusive week range to evaluate historical model alignment vs. closing lines.")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            start_wk = st.number_input("Start Evaluation Week", min_value=1, max_value=18, value=5, help="First week (inclusive) for the backtest range.")
        with col_b2:
            end_wk = st.number_input("End Evaluation Week", min_value=1, max_value=18, value=18, help="Last week (inclusive) for the backtest range.")
        submit_backtest = st.form_submit_button("Execute Automated Backtest Profile", use_container_width=True)

    if submit_backtest:
        backtest_records = []
        eval_sched = raw_sched[(raw_sched["week"] >= start_wk) & (raw_sched["week"] <= end_wk)]
        
        for idx, row in eval_sched.iterrows():
            wk = row["week"]
            h_team = row["home_team"]
            a_team = row["away_team"]
            
            h_prof = processed_profiles[(processed_profiles["team"] == h_team) & (processed_profiles["week"] == wk)]
            a_prof = processed_profiles[(processed_profiles["team"] == a_team) & (processed_profiles["week"] == wk)]
            
            if h_prof.empty or a_prof.empty:
                continue
                
            # Ingest Baselines
            h_epa = h_prof["epa_prev"].values[0]
            a_epa = a_prof["epa_prev"].values[0]
            h_ypp = h_prof["y_per_play_prev"].values[0]
            a_ypp = a_prof["y_per_play_prev"].values[0]
            h_sr = h_prof["success_rate_prev"].values[0]
            a_sr = a_prof["success_rate_prev"].values[0]
            h_sack = h_prof["sack_rate_prev"].values[0]
            a_sack = a_prof["sack_rate_prev"].values[0]
            h_to = h_prof["turnovers_prev"].values[0]
            a_to = a_prof["turnovers_prev"].values[0]
            h_rz = h_prof["red_zone_pct_prev"].values[0]
            a_rz = a_prof["red_zone_pct_prev"].values[0]
            
            # Blueprint Formula Math Delta
            epa_diff = h_epa - a_epa
            ypp_diff = h_ypp - a_ypp
            sr_diff = h_sr - a_sr
            sack_diff = a_sack - h_sack
            to_diff = a_to - h_to
            rz_diff = h_rz - a_rz
            
            composite_delta = (epa_diff * 4.5) + (ypp_diff * 1.5) + (sr_diff * 0.12) + (sack_diff * 0.08) + (to_diff * 0.40) + (rz_diff * 0.04)
            predicted_spread = -(composite_delta * (scale_input / 3.2)) - hfa_input
            
            # Evaluation Math Against True Outcomes
            market_spread = row["spread_line"]
            home_edge = market_spread - predicted_spread
            abs_edge = abs(home_edge)
            
            # Confidence Weight Allocator
            stat_score = min(100.0, (abs_edge / 3.0) * 100.0)
            confidence_score = round((0.40 * stat_score) + (0.30 * 85.0) + (0.20 * (75.0 + abs(market_spread) * 2.5)) + (0.10 * 100.0))
            
            # Map Out Pick Directives
            chosen_side = h_team if home_edge > 0 else a_team
            
            # Determine True Wins Against Spread
            actual_margin = row["away_score"] - row["home_score"]  # Positive means away won or lost by less than spread
            
            # Grading System Evaluator
            if home_edge > 0:  # Backing Home Team
                covered = 1 if (row["home_score"] + market_spread > row["away_score"]) else (0 if (row["home_score"] + market_spread < row["away_score"]) else 0.5)
            else:              # Backing Away Team
                covered = 1 if (row["home_score"] + market_spread < row["away_score"]) else (0 if (row["home_score"] + market_spread > row["away_score"]) else 0.5)
                
            backtest_records.append({
                "Week": wk, "Game": f"{a_team} @ {h_team}", "Market Line": market_spread, "Model Spread": round(predicted_spread, 2),
                "Edge": round(home_edge, 2), "Abs_Edge": abs_edge, "Confidence": confidence_score, "Pick": chosen_side, "Covered": covered
            })
            
        bt_df = pd.DataFrame(backtest_records)
        
        if not bt_df.empty:
            st.success(f"Successfully evaluated {len(bt_df)} games from Week {start_wk} to {end_wk}!")
            
            # --- VALIDATION GRAPHICS ENGINE ---
            st.markdown("---")
            st.subheader("📊 Strategic Model Performance Audit Diagnostic")
            st.caption("High-level win rate metrics computed versus the spread. Use these to assess threshold viability.")
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                overall_wr = bt_df[bt_df["Covered"] != 0.5]["Covered"].mean() * 100
                st.metric("Overall System Win Rate (vs. Spread)", f"{overall_wr:.1f}%")
            with c_m2:
                edge_1_wr = bt_df[(bt_df["Abs_Edge"] >= 1.0) & (bt_df["Covered"] != 0.5)]["Covered"].mean() * 100
                st.metric("Win Rate — Edge ≥ 1.0 pts", f"{edge_1_wr:.1f}%")
            with c_m3:
                edge_2_wr = bt_df[(bt_df["Abs_Edge"] >= 2.0) & (bt_df["Covered"] != 0.5)]["Covered"].mean() * 100
                st.metric("Win Rate — Edge ≥ 2.0 pts", f"{edge_2_wr:.1f}%")
                
            # Plotly Visualization Edge Bucket Check
            st.markdown("### Threshold Viability Profile")
            st.caption("Bar chart showing win percentage by calculated model edge buckets — useful for selecting edge cutoffs.")
            bt_df["Edge Bucket"] = pd.cut(bt_df["Abs_Edge"], bins=[0, 1, 2, 3, 10], labels=["0-1 Pts Edge", "1-2 Pts Edge", "2-3 Pts Edge", "3+ Pts Edge"])
            chart_data = bt_df[bt_df["Covered"] != 0.5].groupby("Edge Bucket", observed=False)["Covered"].mean().reset_index()
            chart_data["Win %"] = chart_data["Covered"] * 100
            
            fig_buckets = px.bar(chart_data, x="Edge Bucket", y="Win %", title="Win Rate Alignment by Calculated Model Edge Thresholds")
            st.plotly_chart(fig_buckets, use_container_width=True)
            
            st.markdown("### Raw Analytical Ledger")
            st.dataframe(bt_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No data found matching evaluation metrics filters.")

# -------------------------------------------------------------------------
# MODE 2: PRODUCTION RUN LIVE OPERATIONAL SYSTEM
# -------------------------------------------------------------------------
else:
    # Render operational selectors
    all_weeks = [int(w) for w in sorted(raw_sched["week"].unique())]
    current_index = int(st.session_state.selected_week - 1)
    current_index = max(0, min(current_index, len(all_weeks) - 1))

    st.session_state.selected_week = st.sidebar.selectbox(
        "Active Operational Week", 
        all_weeks, 
        index=current_index
    )
    active_week_sched = raw_sched[raw_sched["week"] == st.session_state.selected_week].copy()

    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
        "📋 Dashboard", "📅 NFL Schedule", "📊 Team Stats", "🧠 Advanced Stats", 
        "💰 Betting Lines", "🏥 Injuries", "👑 Power Ratings", "🎯 Model Output", 
        "🔥 BillionBetting Card", "📈 Results Tracker"
    ])

    # [REST OF YOUR ORIGINAL 10 TABS CODE REMAINS EXACTLY THE SAME BELOW]
    # TAB 1: DASHBOARD (rendered via ui.dashboard)
    from ui.dashboard import render_dashboard
    with t1:
        render_dashboard(processed_profiles)

    # TAB 2: NFL SCHEDULE
    with t2:
        from ui.schedule import render_schedule
        render_schedule(active_week_sched, st.session_state.selected_week)

    # TAB 3: TEAM STATS
    with t3:
        from ui.team_stats import render_team_stats
        render_team_stats(processed_profiles, st.session_state.selected_week)

    # TAB 4: ADVANCED STATS
    with t4:
        from ui.advanced_stats import render_advanced_stats
        render_advanced_stats(processed_profiles, st.session_state.selected_week)

    # TAB 5: BETTING LINES
    with t5:
        from ui.betting_lines import render_betting_lines
        render_betting_lines(active_week_sched)

    # TAB 6: INJURIES
    with t6:
        from ui.injuries import render_injuries
        render_injuries()

    # TAB 7: POWER RATINGS
    with t7:
        from ui.power_ratings import render_power_ratings
        render_power_ratings(processed_profiles, st.session_state.selected_week)

    # TAB 8: MODEL OUTPUT
    with t8:
        from ui.model_output import render_model_output
        render_model_output(active_week_sched, processed_profiles, scale_input, hfa_input)

    # TAB 9: BILLIONBETTING CARD
    with t9:
        from ui.card import render_card
        render_card(st.session_state.model_runs, st.session_state.selected_week)

    # TAB 10: RESULTS TRACKER
    with t10:
        from ui.results_tracker import render_results_tracker
        render_results_tracker()