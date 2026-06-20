import streamlit as st
import pandas as pd
from storage.tracker import TRACKER_FILE


def render_results_tracker():
    st.subheader("📈 Historical Results Tracker — Ledger & Performance Audits")
    st.caption("Edit ledger rows to record final results and CLV (closing line value). Save changes to persist historical performance.")
    historical_ledger = pd.read_csv(TRACKER_FILE)

    if not historical_ledger.empty:
        st.markdown("### 🖊️ Finalize Ledger Results & CLV Metrics")
        st.caption("Hint: set 'Result' to WIN/LOSS/PENDING and update 'CLV' to reflect net captured units. Use 'Update and Secure' to save.")
        edited_ledger = st.data_editor(historical_ledger, use_container_width=True, num_rows="dynamic")

        if st.button("💾 UPDATE AND SECURE HISTORICAL RESULTS LEDGER", use_container_width=True):
            edited_ledger.to_csv(TRACKER_FILE, index=False)
            st.success("Ledger database changes saved successfully.")
            st.rerun()

        st.markdown("---")
        st.markdown("### 🧠 Strategic Model Performance Audit Diagnostic")
        q_edge1 = edited_ledger[edited_ledger["Edge"].abs() > 1.0]
        q_edge2 = edited_ledger[edited_ledger["Edge"].abs() > 2.0]

        c_an1, c_an2 = st.columns(2)
        with c_an1:
            st.markdown("**Roadmap Question Evaluation Portfolio:**")
            st.write(f"1. Are Edge > 1 plays profitable? **Total Logged Plays:** {len(q_edge1)}")
            st.write(f"2. Are Edge > 2 plays more profitable? **Total Logged Plays:** {len(q_edge2)}")
        with c_an2:
            st.write(f"3. Does Confidence 77+ outperform? **Avg Logged Conf:** {edited_ledger['Confidence'].mean():.1f}%")
            clv_captured = edited_ledger["CLV"].sum()
            st.write(f"4. Net Verified CLV Units Captured: **{clv_captured:+.2f} Pts**")
    else:
        st.info("Ledger is empty. To populate: 1) Visit 'Model Output' and execute calculations, 2) Go to 'BillionBetting Card' and export qualified plays to the ledger, then return here to finalize results.")
