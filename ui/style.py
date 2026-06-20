import streamlit as st


def apply_styles():
    st.set_page_config(
        page_title="BillionBetting — NFL Model & Ledger",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
            .block-container { padding-top: 5.5rem; padding-bottom: 1.5rem; }
            .tier-card {
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 10px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)
