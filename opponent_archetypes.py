# opponent_archetypes.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_opponent_features(df, min_hands=50):
    grouped = df.groupby("player_id").filter(lambda x: len(x) >= min_hands)
    features = grouped.groupby("player_id").agg({
        "player_agg_freq": "mean",
        "player_fold_freq": "mean",
        "player_pass_freq_roll": "mean",
        "hand_strength": "mean",
        "stack_to_pot_ratio": "mean"
    }).fillna(0.0)
    return features

def cluster_opponents(features, n_clusters=5):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    features["archetype"] = clusters

    # Optional visualization
    plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='viridis')
    plt.title("Opponent Archetypes via KMeans")
    plt.show()

    return features.reset_index(), kmeans, scaler