import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import matplotlib.cm as cm  # Import colormap

# Load data from a CSV file
file_path ="C:\\Users\\KIIT\\Desktop\\segmentation\\restaurant_customer_satisfaction.csv"
df = pd.read_csv(file_path)

# Automatically detect numerical columns (features)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Optionally, filter columns based on certain keywords (if available)
possible_keywords = ['income', 'spending', 'score', 'amount', 'balance', 'purchase']
selected_features = [col for col in numeric_cols if any(keyword in col.lower() for keyword in possible_keywords)]

# If no columns are selected by keywords, fallback to using all numeric columns
if not selected_features:
    selected_features = numeric_cols

# Print selected features for debugging
print("Selected Features:", selected_features)

# Check if enough features were selected for a 2D scatter plot
if len(selected_features) < 2:
    raise ValueError("Not enough numeric features selected for clustering. At least two numeric features are required.")

X = df[selected_features]

# Proceed with clustering as before
wcss = []  # Within-cluster sum of squares
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph
plt.figure(figsize=(10, 6)) 
plt.plot(range(1, 6), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Assuming the optimal number of clusters is 3 (adjust based on the Elbow method graph)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Add the cluster information to the original dataframe
df['Cluster'] = y_kmeans

# Visualize the clusters only if at least two features were selected
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=selected_features[0], y=selected_features[1], hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.legend()
plt.show()

# Silhouette Plot
plt.figure(figsize=(10, 6))
silhouette_vals = silhouette_samples(X, y_kmeans)
y_lower = 10
for i in range(3):  # Assuming 3 clusters
    ith_cluster_silhouette_values = silhouette_vals[y_kmeans == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.viridis(float(i) / 3)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    y_lower = y_upper + 10

plt.axvline(x=silhouette_score(X, y_kmeans), color="red", linestyle="--")
plt.title("Silhouette Plot")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster Label")
plt.show()

# Cluster Centroid Plot
plt.figure(figsize=(10, 6))
centroids = kmeans.cluster_centers_
sns.scatterplot(data=df, x=selected_features[0], y=selected_features[1], hue='Cluster', palette='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')  # Mark centroids
plt.title('Cluster Centroids')
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.legend()
plt.show()

# Pair Plot
sns.pairplot(df, hue='Cluster', palette='viridis', vars=selected_features)
plt.show()

# Heatmap of the Cluster Centers
plt.figure(figsize=(10, 6))
sns.heatmap(centroids, annot=True, cmap='viridis', cbar=True, xticklabels=selected_features)
plt.title('Heatmap of Cluster Centers')
plt.show()

# Box Plot
for feature in selected_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Cluster', y=feature, palette='viridis')
    plt.title(f'Box Plot of {feature} by Cluster')
    plt.show()

# Parallel Coordinates Plot
plt.figure(figsize=(10, 6))
parallel_coordinates(df[['Cluster'] + selected_features], class_column='Cluster', color=('#556270', '#4ECDC4', '#C7F464'))
plt.title('Parallel Coordinates Plot')
plt.show()
