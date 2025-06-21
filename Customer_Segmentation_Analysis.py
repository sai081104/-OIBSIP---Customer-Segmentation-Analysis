import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("customer_segmentation_dataset.csv")

# Display basic info
print("\nFirst 5 records:")
print(df.head())
print("\nData Summary:")
print(df.describe())
print("\nData Info:")
print(df.info())

# Encode Gender to numeric
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Select features for clustering
features = df[['Age', 'Annual Income (USD)', 'Spending Score (1-100)', 'Purchase Frequency']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

# Show cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualization 1: Pairplot by Segment
sns.pairplot(df, vars=['Age', 'Annual Income (USD)', 'Spending Score (1-100)'], hue='Segment', palette='Set2')
plt.suptitle("Customer Segments by Features", y=1.02)
plt.savefig("pairplot_segments.png")
plt.close()

# Visualization 2: Average Income by Segment
plt.figure(figsize=(8, 5))
sns.barplot(x='Segment', y='Annual Income (USD)', data=df, palette='pastel')
plt.title("Average Income per Customer Segment")
plt.savefig("income_by_segment.png")
plt.close()

# Visualization 3: Customer Count per Segment
plt.figure(figsize=(6, 4))
sns.countplot(x='Segment', data=df, palette='Set3')
plt.title("Customer Count per Segment")
plt.savefig("customer_count_segment.png")
plt.close()

print("\n✅ All visualizations saved successfully.")

