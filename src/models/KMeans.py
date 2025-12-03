import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.data.load import load_movielens_processed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = load_movielens_processed("movielens_100k_interactions.csv")
# print(df.head())
# print(df.columns)

features = [
    "year",
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

X = df[features].copy()
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

df["cluster"] = kmeans.labels_


pca = PCA(n_components=2)
points_2d = pca.fit_transform(X_scaled)


plt.figure(figsize=(8,6))
plt.scatter(points_2d[:,0], points_2d[:,1], c=df["cluster"], cmap="viridis", alpha=0.6)
plt.title("Clusters de filmes (KMeans + PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()

# Tratamento das colunas não numéricas:

# numeric_cols = []
