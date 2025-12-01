import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import (
    norm, gamma, expon, beta, lognorm, weibull_min, 
    chi2, uniform, triang, pareto
)

# ------------------------------------------------------------
# Helper function to compute fit error
# ------------------------------------------------------------
def compute_error(y_true, y_fit):
    # Average absolute error between histogram and fitted curve
    return np.mean(np.abs(y_true - y_fit))

# ------------------------------------------------------------
# Create dictionary of distribution objects
# ------------------------------------------------------------
DISTRIBUTIONS = {
    "Normal": norm,
    "Gamma": gamma,
    "Exponential": expon,
    "Beta": beta,
    "Lognormal": lognorm,
    "Weibull": weibull_min,
    "Chi-squared": chi2,
    "Uniform": uniform,
    "Triangular": triang,
    "Pareto": pareto
}

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("📊 Histogram Fitting Tool (SciPy Stats)")

st.write("""
This app allows you to upload or enter data, choose a statistical 
distribution, fit parameters using SciPy, and visualize the result.
""")

# ------------------------------------------------------------
# DATA INPUT SECTION
# ------------------------------------------------------------
st.header("1. Upload or Enter Data")

option = st.radio(
    "Choose data input method:",
    ("Enter manually", "Upload CSV file")
)

# --- Manual Entry ---
data = None
if option == "Enter manually":
    manual = st.text_area("Enter numbers separated by commas:", "1,2,3,4,5,6")
    try:
        data = np.array([float(x) for x in manual.split(",")])
    except:
        st.warning("Please enter valid comma-separated numbers.")

# --- Upload CSV ---
else:
    file = st.file_uploader("Upload CSV with a single column of numbers")
    if file:
        df = pd.read_csv(file)
        data = df.iloc[:, 0].values

# Stop if no data yet
if data is None:
    st.stop()

# ------------------------------------------------------------
# DISTRIBUTION SELECTION AND FITTING
# ------------------------------------------------------------
st.header("2. Choose a Distribution to Fit")

dist_name = st.selectbox("Select distribution:", list(DISTRIBUTIONS.keys()))
Dist = DISTRIBUTIONS[dist_name]

# Fit parameters
params = Dist.fit(data)

st.subheader("Fitted Parameters")
st.write(params)

# ------------------------------------------------------------
# MANUAL FITTING MODE
# ------------------------------------------------------------
st.header("3. Manual Fit (Adjust Parameters)")

manual_fit = st.checkbox("Enable manual fitting")

if manual_fit:
    sliders = []
    st.write("Adjust parameters:")
    for i, p in enumerate(params):
        sliders.append(
            st.slider(f"Parameter {i}", p - abs(p)*2 - 1, p + abs(p)*2 + 1, p)
        )
    params = sliders

# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------
st.header("4. Visualization")

fig, ax = plt.subplots(figsize=[6, 4])

# Histogram
counts, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.4)

# Fitted curve
x = np.linspace(min(data), max(data), 300)
dist = Dist(*params)
pdf = dist.pdf(x)

ax.plot(x, pdf, linewidth=2)

ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title(f"Data Histogram with Fitted {dist_name} Distribution")

st.pyplot(fig)

# ------------------------------------------------------------
# FIT ERROR
# ------------------------------------------------------------
st.header("5. Fit Error")

# Convert histogram to midpoints
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Compute fitted curve at each histogram bin
hist_fit = dist.pdf(bin_centers)
error = compute_error(counts, hist_fit)

st.write(f"**Average error:** {error:.5f}")
