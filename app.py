
import streamlit as st

st.set_page_config(
    page_title="AI Trading Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Analyst")

st.success("Your Streamlit application is running!")

st.header("Market Analysis")

symbol = st.text_input(
    "Enter Stock Symbol",
    value="NIFTY"
)

if st.button("Analyze"):
    st.write("Selected symbol:", symbol)
    st.info("Your trading analyst is ready for the next development step.")
