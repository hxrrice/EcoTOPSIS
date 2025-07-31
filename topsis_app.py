
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="EcoTOPSIS Ranking App", layout="centered")

# ---------- Title & Description ----------
st.title("🌱 EcoTOPSIS: Sustainable Ranking App")
st.markdown("""
This app applies the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method to rank alternatives based on multiple criteria.

🧾 **How It Works**:
1. Upload a dataset (CSV/Excel) with alternatives and numerical criteria.
2. Enter weights and impacts for each criterion.
3. View the ranking and download the result!
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
st.header("📂 Upload Your Data")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except:
        st.error("❌ Failed to read file. Ensure it is a valid CSV or Excel file.")
else:
    st.info("No file uploaded. Using example data.")
    df = get_example_data()

# ---------- Input Weights and Impacts ----------
st.subheader("⚙️ Input Weights and Impacts")

if df.shape[1] < 3:
    st.error("❌ Please upload a dataset with at least one alternative column and two numeric criteria.")
    st.stop()

criteria = df.columns[1:]
default_weights = [round(1 / len(criteria), 2)] * len(criteria)
weights = st.text_input("Enter weights (comma-separated):", ",".join(map(str, default_weights)))
impacts = st.text_input("Enter impacts for each criterion (+ for benefit, - for cost):", ",".join(["-"] + ["+"] * (len(criteria) - 1)))

try:
    weights = list(map(float, weights.strip().split(',')))
    impacts = list(map(str.strip, impacts.strip().split(',')))

    if len(weights) != len(criteria) or len(impacts) != len(criteria):
        st.error("❌ Number of weights and impacts must match number of criteria.")
        st.stop()

    if not all(impact in ["+", "-"] for impact in impacts):
        st.error("❌ Impacts must be '+' or '-'.")
        st.stop()
except:
    st.error("❌ Invalid format for weights or impacts.")
    st.stop()

# ---------- Normalize Matrix ----------
matrix = df.iloc[:, 1:].astype(float)
norm_matrix = matrix / np.sqrt((matrix ** 2).sum())

# ---------- Weighted Normalized Matrix ----------
weighted_matrix = norm_matrix * weights

# ---------- Ideal & Anti-Ideal Solutions ----------
ideal = np.where(np.array(impacts) == "+", weighted_matrix.max(), weighted_matrix.min())
anti_ideal = np.where(np.array(impacts) == "+", weighted_matrix.min(), weighted_matrix.max())

# ---------- Distances ----------
dist_ideal = np.sqrt(((weighted_matrix - ideal) ** 2).sum(axis=1))
dist_anti = np.sqrt(((weighted_matrix - anti_ideal) ** 2).sum(axis=1))

# ---------- Relative Closeness & Ranking ----------
closeness = dist_anti / (dist_ideal + dist_anti)
df_result = df.copy()
df_result["Closeness"] = closeness
df_result["Rank"] = df_result["Closeness"].rank(method='max', ascending=False).astype(int)
df_result.sort_values("Rank", inplace=True)

# ---------- Display Results ----------
st.subheader("📊 Results")
st.dataframe(df_result.style.highlight_max("Closeness", axis=0, color="lightgreen"))

# ---------- Show Intermediate Matrices ----------
with st.expander("🔍 Show Normalized and Weighted Matrices"):
    st.write("**Normalized Matrix:**")
    st.dataframe(pd.DataFrame(norm_matrix, columns=criteria, index=df["Alternative"]))

    st.write("**Weighted Normalized Matrix:**")
    st.dataframe(pd.DataFrame(weighted_matrix, columns=criteria, index=df["Alternative"]))

# ---------- Downloadable Output ----------
def convert_df(df):
    output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:

        df.to_excel(writer, index=False, sheet_name='TOPSIS Result')
    def convert_df(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='TOPSIS Result')
    return output.getvalue()


st.download_button(
    label="📥 Download Result as Excel",
    data=convert_df(df_result),
    file_name="topsis_result.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
