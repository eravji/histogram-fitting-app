# histogram_fitting_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------
# Helper Functions
# ------------------------------

def fit_distribution(data, dist_name):
    # Select the distribution based on name
    if dist_name == 'Normal':
        dist = stats.norm
    elif dist_name == 'Gamma':
        dist = stats.gamma
    elif dist_name == 'Weibull':
        dist = stats.weibull_min
    elif dist_name == 'Exponential':
        dist = stats.expon
    elif dist_name == 'Beta':
        dist = stats.beta
    elif dist_name == 'Uniform':
        dist = stats.uniform
    elif dist_name == 'Lognormal':
        dist = stats.lognorm
    elif dist_name == 'Pareto':
        dist = stats.pareto
    elif dist_name == 'Chi-Square':
        dist = stats.chi2
    elif dist_name == 'Student-t':
        dist = stats.t
    else:
        st.error("Distribution not recognized!")
        return None, None

    # Fit the distribution to the data
    params = dist.fit(data)

    # Calculate fitted PDF values
    x = np.linspace(min(data), max(data), 100)
    pdf = dist.pdf(x, *params)

    # Compute error between histogram and PDF
    hist_values, bin_edges = np.histogram(data, bins=25, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    error = np.mean(np.abs(hist_values - dist.pdf(bin_centers, *params)))

    return x, pdf, params, error

# ------------------------------
# Streamlit App Layout
# ------------------------------

st.title("Histogram Fitting App")

st.write("This app allows you to fit different statistical distributions to your data.")

# --- Data Input ---
st.header("Data Input")

data_option = st.radio("How would you like to input your data?", ("Manual Entry", "Upload CSV"))

data = []

if data_option == "Manual Entry":
    raw_data = st.text_area("Enter your data separated by commas:")
    if raw_data:
        data = [float(i) for i in raw_data.split(',')]
elif data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        # Assume first column contains data
        data = df.iloc[:, 0].values.tolist()

# --- Distribution Selection ---
if len(data) > 0:
    st.header("Select Distribution to Fit")
    dist_name = st.selectbox(
        "Choose a distribution",
        ["Normal", "Gamma", "Weibull", "Exponential", "Beta",
         "Uniform", "Lognormal", "Pareto", "Chi-Square", "Student-t"]
    )

    # Fit distribution and show results
    x, pdf, params, error = fit_distribution(data, dist_name)

    if x is not None:
        # Plot histogram and fitted curve
        fig, ax = plt.subplots(figsize=[6,4])
        ax.hist(data, bins=25, density=True, alpha=0.5, label='Data')
        ax.plot(x, pdf, 'r-', lw=2, label=f'{dist_name} Fit')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        st.pyplot(fig)

        # Display parameters and error
        st.subheader("Fitted Parameters")
        st.write(params)
        st.subheader("Fit Quality")
        st.write("Mean absolute error between histogram and PDF:", error)

