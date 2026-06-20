import streamlit as st
import pandas as pd


def render_model_output(active_week_sched, processed_profiles, scale_input, hfa_input):
    st.subheader("🎯 Context Intelligence & Math Modeling Workspace")
    st.caption("Open a matchup below to tweak weather, injuries, and situational overrides — the model will update predicted spread, total, confidence, and tier.")
    adjusted_game_records = []

    from ui.components.game_expander import render_game_expander

    for idx, row in active_week_sched.iterrows():
        rec = render_game_expander(row, processed_profiles, st.session_state.selected_week, scale_input, hfa_input)
        if rec:
            adjusted_game_records.append(rec)

    st.session_state.model_runs[st.session_state.selected_week] = pd.DataFrame(adjusted_game_records)
    st.success("✅ Calculations executed and finalized.")
