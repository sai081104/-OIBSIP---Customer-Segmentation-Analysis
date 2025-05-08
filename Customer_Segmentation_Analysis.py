# Customer Segmentation Analysis - OIBSIP Data Analytics Task

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("Customer_Segmentation.csv")

# Display first 5 rows
print("Dataset Preview:\n", data.head())

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Drop duplicates (if any)
data.drop_duplicates(inplace=True)

# Descriptive Statistics
print("\nDescriptive Statistics:\n", data.describe())

# Data Preprocessing
# Selecting relevant columns for segmentation
segmentation_data = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize the data
scaler = StandardScaler()
segmentation_data_scaled = scaler.fit_transform(segmentation_data)

# Elbow Method to Determine Optimal Clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(segmentation_data_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(segmentation_data_scaled)

# Visualize the Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=data["Annual Income (k$)"], y=data["Spending Score (1-100)"],
                hue=data["Cluster"], palette="viridis", s=100, alpha=0.7)
plt.title("Customer Segmentation Based on Income and Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Cluster Analysis
print("\nCluster Analysis:\n", data.groupby("Cluster").mean())
