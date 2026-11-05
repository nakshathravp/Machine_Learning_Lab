
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Step 1: Generate network traffic dataset

np.random.seed(42)

# Normal network traffic
normal = np.column_stack([
    np.random.normal(2, 0.5, 200),       # Duration
    np.random.normal(20, 5, 200),        # Packets
    np.random.normal(1500, 300, 200),    # Bytes
    np.random.poisson(1, 200),           # Failed connections
    np.random.normal(75, 10, 200)        # Average packet size
])

# Abnormal network traffic
abnormal = np.column_stack([
    np.random.normal(8, 2, 50),
    np.random.normal(150, 30, 50),
    np.random.normal(15000, 3000, 50),
    np.random.poisson(12, 50),
    np.random.normal(120, 20, 50)
])

# Combine datasets
X = np.vstack((normal, abnormal))

# Labels are used only for evaluating the synthetic example
actual_labels = np.array([0] * 200 + [1] * 50)

# Remove negative values
X = np.maximum(X, 0.1)

df = pd.DataFrame(X, columns=[
    'Duration',
    'Packets',
    'Bytes',
    'Failed_Connections',
    'Avg_Packet_Size'
])

print("NETWORK TRAFFIC DATASET")
print(df.head())

# Step 2: Normalize features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means clustering

model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X_scaled)

# Step 4: Identify abnormal cluster

scores = model.cluster_centers_.mean(axis=1)
abnormal_cluster = np.argmax(scores)

predicted_labels = (
    clusters == abnormal_cluster
).astype(int)

# Step 5: Display classification results

df['Cluster'] = clusters

df['Result'] = np.where(
    predicted_labels == 1,
    'Abnormal',
    'Normal'
)

print("\nNETWORK EVENT CLASSIFICATION")
print(df[['Duration', 'Packets', 'Bytes',
          'Failed_Connections', 'Result']].head(15))

# Step 6: Calculate accuracy for sample dataset

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

print("\nMODEL RESULTS")
print("Total Network Events:", len(df))
print("Normal Events:", sum(predicted_labels == 0))
print("Abnormal Events:", sum(predicted_labels == 1))
print("Accuracy:", accuracy * 100, "%")

# Step 7: Visualize clusters

plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 1],
    X_scaled[:, 2],
    c=predicted_labels,
    cmap='coolwarm'
)

plt.xlabel("Number of Packets (Scaled)")
plt.ylabel("Number of Bytes (Scaled)")
plt.title("Network Traffic Clustering using K-Means")

plt.show()

print("\nNetwork Event Classification Completed!")
