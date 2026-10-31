import streamlit as st

st.title("AI-Driven Climate Risk Forecasting")

st.write("Welcome to my Final Year Project!")

st.header("Climate Risk Assessment")

district = st.selectbox(
    "Select District",
    ["Coimbatore", "Chennai", "Madurai", "Salem"]
)

st.write("Selected District:", district)