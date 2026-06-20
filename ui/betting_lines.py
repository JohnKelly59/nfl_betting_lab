import streamlit as st


def render_betting_lines(active_week_sched):
    st.subheader("💰 Live Vegas Sportsbook Aggregations")
    st.caption("Marketplace consensus lines (spread/total) used to compute edge vs. the model's predicted numbers.")
    st.dataframe(active_week_sched[["away_team", "home_team", "spread_line", "total_line"]].rename(
        columns={"spread_line": "Consensus Spread", "total_line": "Consensus Total"}
    ), use_container_width=True, hide_index=True)
