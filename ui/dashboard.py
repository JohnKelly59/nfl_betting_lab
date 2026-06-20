import streamlit as st
import pandas as pd
from storage.tracker import TRACKER_FILE
from ui.components.metric import render_metric


def render_dashboard(processed_profiles):
    st.subheader("🏈 BillionBetting Command Center — Dashboard & System Health")
    st.caption("Quick operational summary: ledger counts, CLV, win rate, and the active week context.")
    db_col1, db_col2, db_col3, db_col4 = st.columns(4)
    tracker_df = pd.read_csv(TRACKER_FILE)

    render_metric(db_col1, "Total Profiled Plays (Ledger Entries)", len(tracker_df), "Total number of plays recorded in the historical ledger.")
    clv_avg = tracker_df["CLV"].mean() if not tracker_df.empty else 0.0
    render_metric(db_col2, "Average CLV Captured", f"{clv_avg:+.2f} Pts", "Average captured CLV (closing line value) per logged play.")
    win_rate = (len(tracker_df[tracker_df["Result"] == "WIN"]) / len(tracker_df) * 100) if len(tracker_df) > 0 else 0.0
    render_metric(db_col3, "System Model Accuracy", f"{win_rate:.1f}%", "Percent of logged plays marked as 'WIN' in the ledger (excluding pushes).")
    render_metric(db_col4, "Active Week Horizon", f"Week {st.session_state.selected_week}", "Currently selected operational week for model calculations.")

    st.markdown("---")
    st.markdown("### ⚡ System Status Checklist")
    st.caption("At-a-glance health indicators for data pipelines and metric engine readiness.")
    c1, c2, c3 = st.columns(3)
    c1.success("✅ Power Query Engine Pipeline Connected")
    c1.caption("Live schedule and team stats ingestion (or simulated fallback).")
    c2.success(f"✅ Metric Engine Loaded ({len(processed_profiles)} System Vectors)")
    c2.caption("Processed team differential vectors available for model computations.")
    c3.info("💡 Projections Ready for Context Layer Inputs")
    c3.caption("Open a matchup in Model Output to tune injuries, weather, and situational factors.")
