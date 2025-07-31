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

        # Check and rename the first column to "Alternative" if necessary
        if df.columns[0] != "Alternative":
            df.columns = ["Alternative"] + list(df.columns[1:])
            st.warning("The first column has been renamed to 'Alternative'.")
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
    
    # Sidebar dropdown for impact type (+ for benefit, - for cost)
    impact = st.sidebar.selectbox(f"Impact for {criterion}", options=["+", "-"], key=criterion)
    impacts.append(impact)

# Adjust the last slider if the total weight exceeds 1
if total_weight > 1.0:
    st.sidebar.warning(f"Warning: The sum of weights is {total_weight:.2f}. Adjusting the last weight to ensure the sum is 1.")
    weights[-1] = 1.0 - sum(weights[:-1])

# Display weights for the user
st.sidebar.write(f"Total weight: {sum(weights)} (adjusted to 1 if necessary)")

# ---------- Step 1: Display Uploaded Data ----------
st.subheader("Step 1: Uploaded Data")
st.write(df)

# ---------- Step 2: Normalize the Matrix ----------
st.subheader("Step 2: Normalize the Decision Matrix")
matrix = df.iloc[:, 1:].astype(float)
norm_matrix = matrix / np.sqrt((matrix ** 2).sum())
st.write(norm_matrix)

# ---------- Step 3: Compute the Weighted Normalized Matrix ----------
st.subheader("Step 3: Compute the Weighted Normalized Matrix")
weighted_matrix = norm_matrix * weights
st.write(weighted_matrix)

# ---------- Step 4: Calculate Ideal (PIS) and Anti-Ideal (NIS) Solutions ----------
st.subheader("Step 4: Calculate Ideal (PIS) and Anti-Ideal (NIS) Solutions")
ideal = np.where(np.array(impacts) == "+", weighted_matrix.max(), weighted_matrix.min())
anti_ideal = np.where(np.array(impacts) == "+", weighted_matrix.min(), weighted_matrix.max())

st.write("Ideal (PIS) Solution:")
st.write(ideal)

st.write("Anti-Ideal (NIS) Solution:")
st.write(anti_ideal)

# ---------- Step 5: Compute Euclidean Distances ----------
st.subheader("Step 5: Compute the Euclidean Distances")
dist_ideal = np.sqrt(((weighted_matrix - ideal) ** 2).sum(axis=1))
dist_anti = np.sqrt(((weighted_matrix - anti_ideal) ** 2).sum(axis=1))

st.write("Euclidean Distance from PIS:")
st.write(dist_ideal)

st.write("Euclidean Distance from NIS:")
st.write(dist_anti)

# ---------- Step 6: Calculate Relative Closeness to Ideal Solution ----------
st.subheader("Step 6: Calculate Relative Closeness to Ideal Solution")
closeness = dist_anti / (dist_ideal + dist_anti)
st.write("Relative Closeness to Ideal Solution:")
st.write(closeness)

# ---------- Step 7: Display Final Ranking ----------
st.subheader("Step 7: Final Ranking")
df_result = df.copy()
df_result["Closeness"] = closeness
df_result["Rank"] = df_result["Closeness"].rank(method='max', ascending=False).astype(int)
df_result.sort_values("Rank", inplace=True)

st.write(df_result.style.highlight_max("Closeness", axis=0, color="lightgreen"))

# ---------- Show Intermediate Matrices ----------
with st.expander("Show Normalized and Weighted Matrices"):
    st.write("**Normalized Matrix:**")
    st.dataframe(pd.DataFrame(norm_matrix, columns=criteria, index=df["Alternative"]))

    st.write("**Weighted Normalized Matrix:**")
    st.dataframe(pd.DataFrame(weighted_matrix, columns=criteria, index=df["Alternative"]))

# ---------- Downloadable Output ----------
def convert_df(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='TOPSIS Result')
    return output.getvalue()

st.download_button(
    label="Download Result as Excel",
    data=convert_df(df_result),
    file_name="topsis_result.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
