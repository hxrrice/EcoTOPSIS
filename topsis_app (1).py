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
    }
    return pd.DataFrame(data)

# ---------- File Upload ----------
st.header("Upload Your Data")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file. Ensure it is a valid CSV or Excel file. Error: {str(e)}")
else:
    st.info("No file uploaded. Using example data.")
    df = get_example_data()

# ---------- Input Weights and Impacts (Sidebar) ----------
st.sidebar.subheader("Input Weights and Impacts")

if df.shape[1] < 3:
    st.error("Please upload a dataset with at least one alternative column and two numeric criteria.")
    st.stop()

criteria = df.columns[1:]  # All columns except 'Alternative'

# Sidebar sliders for each criterion's weight
weights = []
total_weight = 0
impacts = []

for i, criterion in enumerate(criteria):
    weight = st.sidebar.slider(f"Weight for {criterion}", 0.0, 1.0, 1/len(criteria), step=0.01)
    weights.append(weight)
    total_weight += weight
    
    #
