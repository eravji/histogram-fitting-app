import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import (
    norm, gamma, expon, beta, lognorm, weibull_min,
    chi2, uniform, triang, pareto
)

# quick helper to compare histogram + model curve
def compute_error(actual, fitted):
    return np.mean(np.abs(actual - fitted))


# list of distributions we let the user try out
# (easier to work with them in a dict)
DISTS = {
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

# ---- Streamlit UI ----

st.title("Histogram Fitting Tool")

st.write(
    "Upload some data or type it in, pick a distribution, "
    "and the app will fit the parameters for you."
)

# ---- Get the data ----
st.header("1. Data Input")

choice = st.radio("How do you want to provide data?", ["Manual", "CSV upload"])

data = None

if choice == "Manual":
    raw = st.text_area("Numbers (comma-separated):", "1,2,3,4,5")
    try:
        data = np.array([float(x.strip()) for x in raw.split(",")])
    except:
        st.warning("Something's off. Make sure everything is numbers.")
else:
    file = st.file_uploader("Upload a CSV with one column of numbers")
    if file:
        df = pd.read_csv(file)
        data = df.iloc[:, 0].to_numpy()

if data is None:
    st.stop()

# ---- Distribution selection + fitting ----
st.header("2. Pick a Distribution")

dist_name = st.selectbox("Distribution:", list(DISTS.keys()))
dist_class = DISTS[dist_name]

# SciPy does the parameter fitting here
params = dist_class.fit(data)

st.write("Fitted parameters:", params)

# ---- Manual tweaking (optional) ----
st.header("3. Manual Adjustment")
use_manual = st.checkbox("Let me tweak the parameters")

if use_manual:
    new_params = []
    for i, p in enumerate(params):
        # just giving a reasonably wide range around the fitted value
        rng = abs(p) + 1
        val = st.slider(f"param {i}", p - rng, p + rng, p)
        new_params.append(val)
    params = new_params  # override

# ---- Plotting ----
st.header("4. Plot")

fig, ax = plt.subplots(figsize=(6, 4))

# histogram
counts, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.4)

# model curve
x = np.linspace(data.min(), data.max(), 300)
dist_obj = dist_class(*params)
pdf_vals = dist_obj.pdf(x)

ax.plot(x, pdf_vals, linewidth=2)
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title(f"{dist_name} Fit")

st.pyplot(fig)

# ---- error calculation ----
st.header("5. Fit Error")

bin_centers = (bins[:-1] + bins[1:]) / 2
fit_at_bins = dist_obj.pdf(bin_centers)

err = compute_error(counts, fit_at_bins)
st.write(f"Average Absolute Error: {err:.5f}")
