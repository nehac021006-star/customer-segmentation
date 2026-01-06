# app.py
# Customer Segmentation using RFM + KMeans (WITH WORKING PREDICTION)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import BytesIO

sns.set(style="whitegrid")

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation using RFM & KMeans")
st.markdown("Segment customers and predict cluster for new customers.")

# ---------------- Helper Functions ----------------
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def clean_data(df):
    df = df.dropna(subset=['CustomerID']).copy()
    df['CustomerID'] = df['CustomerID'].astype(int)

    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

def compute_rfm(df):
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def run_kmeans(rfm, k):
    X = rfm[['Recency', 'Frequency', 'Monetary']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, rfm['Cluster'])
    return rfm, sil, scaler, kmeans

def profile_clusters(rfm):
    profile = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Size'}).round(2)

    segment_map = {
        0: 'Champions',
        1: 'Loyal',
        2: 'Potential',
        3: 'At Risk'
    }

    profile['Segment'] = profile.index.map(lambda x: segment_map.get(x, 'Other'))
    return profile

def to_csv_bytes(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- Sidebar ----------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Online Retail CSV", type=['csv'])
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)

if st.sidebar.button("Run Segmentation"):
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df_clean = clean_data(df)
        rfm = compute_rfm(df_clean)

        rfm_clustered, sil, scaler, kmeans = run_kmeans(rfm, k)

        # SAVE MODEL IN SESSION
        st.session_state["rfm"] = rfm_clustered
        st.session_state["scaler"] = scaler
        st.session_state["kmeans"] = kmeans
        st.session_state["silhouette"] = sil

        st.success("Segmentation completed successfully!")
    else:
        st.warning("Please upload a CSV file.")

# ---------------- Display Results ----------------
if "rfm" in st.session_state:
    rfm_clustered = st.session_state["rfm"]

    st.subheader("RFM Table")
    st.dataframe(rfm_clustered.head())

    st.success(f"Silhouette Score: {st.session_state['silhouette']:.3f}")

    st.subheader("Customers per Cluster")
    st.bar_chart(rfm_clustered['Cluster'].value_counts().sort_index())

    st.subheader("Cluster Profile")
    profile = profile_clusters(rfm_clustered)
    st.dataframe(profile)

    st.subheader("Recency vs Monetary")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=rfm_clustered,
        x='Recency',
        y='Monetary',
        hue='Cluster',
        palette='viridis',
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("3D RFM View")
    fig3d = px.scatter_3d(
        rfm_clustered,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster'
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ---------------- Prediction Section ----------------
    st.markdown("---")
    st.subheader("🔮 Predict Segment for New Customer")

    r = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    f = st.number_input("Frequency (number of purchases)", min_value=0, value=3)
    m = st.number_input("Monetary (total spend)", min_value=0.0, value=200.0)

    if st.button("Predict Customer Segment"):
        new_data = np.array([[r, f, m]])
        new_scaled = st.session_state["scaler"].transform(new_data)
        cluster = st.session_state["kmeans"].predict(new_scaled)[0]

        segment_map = {
            0: "Champions",
            1: "Loyal",
            2: "Potential",
            3: "At Risk"
        }

        st.success(f"Predicted Cluster: {cluster}")
        st.success(f"Customer Segment: {segment_map.get(cluster, 'Unknown')}")

    st.download_button(
        "Download RFM with Clusters",
        to_csv_bytes(rfm_clustered),
        "rfm_with_clusters.csv"
    )

else:
    st.info("Upload data and click **Run Segmentation** to begin.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Customer Segmentation App using RFM Analysis & KMeans Clustering")
