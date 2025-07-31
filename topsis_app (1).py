import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="EcoTOPSIS Ranking App", layout="centered")

# ---------- Title & Description ----------
st.title("EcoTOPSIS: Sustainable Ranking App")
st.markdown("""
This app applies the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) method to rank alternatives based on multiple criteria.

How It Works:
1. Upload a dataset (CSV/Excel) with alternatives and numerical criteria.
2. Enter weights and impacts for each criterion using sliders and dropdowns.
3. View the ranking and download the result.
""")

# ---------- Example Dataset ----------
def get_example_data():
    data = {
        "Alternative": ["A1", "A2", "A3", "A4"],
        "Cost": [250, 200, 300, 275],
        "Efficiency": [60, 70, 50, 65],
        "Sustainability": [80, 90, 70, 85],
