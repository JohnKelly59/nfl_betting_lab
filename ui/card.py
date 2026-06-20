import streamlit as st
import pandas as pd
from datetime import datetime
from storage.tracker import TRACKER_FILE


def render_card(model_runs, selected_week):
    st.subheader("🔥 BillionBetting Qualified Execution Card — Official")
    st.caption("The official execution card shows qualified plays (Tiered recommendations). Export selected plays to the historical ledger below.")
    if selected_week in model_runs:
        current_card_df = model_runs[selected_week]
        qualified_plays = current_card_df[current_card_df["Tier"] != "NO BET"].sort_values(by="Abs Edge", ascending=False)

        if not qualified_plays.empty:
            from ui.components.tier_card import render_tier_card
            for c_idx, p_row in qualified_plays.iterrows():
                render_tier_card(p_row)

            st.markdown("---")
            if st.button("📥 LOG AND EXPORT ENTIRE CARD PLAYS TO HISTORICAL DB", use_container_width=True):
                current_tracker = pd.read_csv(TRACKER_FILE)
                rows_to_append = []
                for _, p_r in qualified_plays.iterrows():
                    rows_to_append.append({
                        "Date": datetime.now().strftime("%Y-%m-%d"), "Game": p_r["Matchup"],
                        "Closing Line": p_r["Market Spread"], "Model Line": p_r["Model Spread"],
                        "Edge": p_r["Raw Edge"], "Confidence": p_r["Confidence"], "Tier": p_r["Tier"],
                        "Result": "PENDING", "ATS Result": "PENDING", "CLV": 0.0
                    })
                updated_tracker = pd.concat([current_tracker, pd.DataFrame(rows_to_append)], ignore_index=True)
                updated_tracker.to_csv(TRACKER_FILE, index=False)
                st.toast("Card registered cleanly into persistent tracker!", icon="💾")
                st.caption(f"Exported {len(rows_to_append)} plays to persistent tracker.")
        else:
            st.info("No qualified plays this week. Adjust model inputs or widen thresholds in Model Output if desired.")
    else:
        st.warning("Please run model calculations in the Model Output tab first to initialize the execution card.")
