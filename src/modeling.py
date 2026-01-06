# src/modeling.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def fit_kmeans(X_scaled, n_clusters: int = 4, random_state: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia = kmeans.inertia_
    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = np.nan
    return labels, kmeans, inertia, sil
