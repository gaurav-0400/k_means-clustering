# ============================================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

# Pandas is used for loading and manipulating tabular data
import pandas as pd

# NumPy is used for numerical operations
import numpy as np

# Matplotlib is used for creating graphs and visualizations
import matplotlib.pyplot as plt

# Seaborn is used for better-looking statistical visualizations
import seaborn as sns

# StandardScaler is used to scale numerical features
from sklearn.preprocessing import StandardScaler

# KMeans is the K-Means clustering algorithm
from sklearn.cluster import KMeans

# Silhouette Score is used to evaluate the quality of clusters
from sklearn.metrics import silhouette_score

# PCA is used later to reduce multiple dimensions to 2 dimensions
# so that we can visualize the clusters
from sklearn.decomposition import PCA


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

# Change this path to the location of your CSV file.
#
# Example:
# If your Python file and CSV are in the same folder:
#
# df = pd.read_csv("Mall_Customers.csv")
#
# If the CSV has another name, use that name here.

df = pd.read_csv("Mall_Customers.csv")


# Display the first 5 rows of the dataset
# This helps us understand what the data looks like.
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. UNDERSTAND THE DATASET
# ============================================================

# Display the number of rows and columns
#
# Example:
# (200, 5)
#
# means:
# 200 rows
# 5 columns

print("\nDataset Shape:")
print(df.shape)


# Display all column names
print("\nColumn Names:")
print(df.columns)


# Display information about each column
#
# It shows:
# - Column names
# - Number of non-null values
# - Data types
#
# This is useful before preprocessing the dataset.

print("\nDataset Information:")
df.info()


# Display statistical information about numerical columns
#
# It shows:
# - count
# - mean
# - standard deviation
# - minimum
# - maximum
# - quartiles

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

# isnull() checks whether a value is missing.
#
# sum() counts the missing values in every column.

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 5. CHECK DUPLICATE ROWS
# ============================================================

# duplicated() checks whether rows are repeated.
#
# sum() tells us how many duplicate rows exist.

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 6. CHECK UNIQUE VALUES
# ============================================================

# unique() shows different values present in a column.
#
# We can use this to understand categorical columns.

print("\nUnique Gender Values:")

# NOTE:
# Change "Gender" if your dataset uses another column name.

print(df["Gender"].unique())


# ============================================================
# 7. BASIC DATA CLEANING
# ============================================================

# If duplicate rows exist, remove them.
#
# inplace=True means the changes are made directly to df.

df.drop_duplicates(inplace=True)


# ============================================================
# 8. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Before applying K-Means, we should understand the data visually.


# ------------------------------------------------------------
# 8.1 Gender Distribution
# ------------------------------------------------------------

# countplot counts how many customers belong to each gender.

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Gender")

plt.title("Customer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.show()


# ------------------------------------------------------------
# 8.2 Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.show()


# ------------------------------------------------------------
# 8.3 Annual Income Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Annual Income (k$)",
    bins=20,
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")

plt.show()


# ------------------------------------------------------------
# 8.4 Spending Score Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Spending Score (1-100)",
    bins=20,
    kde=True
)

plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# 9. UNDERSTAND RELATIONSHIP BETWEEN FEATURES
# ============================================================

# A scatter plot helps us visually check whether customers
# naturally form groups.

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"]
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.show()


# ============================================================
# 10. SELECT FEATURES FOR K-MEANS
# ============================================================

# K-Means needs numerical features.
#
# For our first project, we will use:
#
# Annual Income
# +
# Spending Score
#
# We don't use CustomerID because it is just an identifier.
#
# We also don't use Gender initially because we want to keep
# our first K-Means implementation simple.

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]


# Display the selected features
print("\nSelected Features:")
print(X.head())


# ============================================================
# 11. FEATURE SCALING
# ============================================================

# K-Means uses distance to determine which points belong
# to which cluster.
#
# If one feature has a much larger numerical range than
# another feature, it can dominate the distance calculation.
#
# StandardScaler transforms the features so that they are
# on a comparable scale.

scaler = StandardScaler()


# fit_transform():
#
# fit() -> calculates mean and standard deviation
#
# transform() -> scales the values
#
# The result is stored in X_scaled.

X_scaled = scaler.fit_transform(X)


# Convert the scaled NumPy array back into a DataFrame
# so that it is easier to read.

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)


# Display scaled data
print("\nScaled Data:")
print(X_scaled_df.head())


# ============================================================
# 12. UNDERSTAND ELBOW METHOD
# ============================================================

# We need to decide how many clusters K-Means should create.
#
# Should K be:
#
# 2?
# 3?
# 4?
# 5?
# 6?
#
# We use the Elbow Method to help determine a suitable K.


# Create an empty list to store inertia values.

inertia = []


# Try K values from 1 to 10.

for k in range(1, 11):

    # Create K-Means model.
    #
    # n_clusters=k means the current number of clusters is k.
    #
    # random_state=42 makes our result reproducible.
    #
    # n_init=10 means K-Means will run multiple initializations
    # and choose a better solution.

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )


    # Train K-Means on our scaled data.

    kmeans.fit(X_scaled)


    # inertia_ gives the WCSS value.
    #
    # WCSS = Within Cluster Sum of Squares
    #
    # It measures how close the data points are to their
    # cluster centers.

    inertia.append(kmeans.inertia_)


# ============================================================
# 13. PLOT ELBOW CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia / WCSS")

plt.xticks(range(1, 11))

plt.show()


# ============================================================
# 14. TRAIN FINAL K-MEANS MODEL
# ============================================================

# After looking at the Elbow graph, choose an appropriate K.
#
# For this dataset, 5 is commonly used as a starting point.
#
# IMPORTANT:
# Don't blindly use 5.
# Look at your Elbow graph and also check the Silhouette Score.

optimal_k = 5


# Create the final K-Means model.

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)


# fit_predict():
#
# fit() -> trains the K-Means model
#
# predict() -> assigns every customer to a cluster
#
# It returns cluster labels such as:
#
# 0
# 1
# 2
# 3
# 4

clusters = kmeans.fit_predict(X_scaled)


# Display cluster labels

print("\nCluster Labels:")
print(clusters)


# ============================================================
# 15. ADD CLUSTER LABELS TO ORIGINAL DATASET
# ============================================================

# Add a new column called Cluster.

df["Cluster"] = clusters


# Display the updated dataset.

print("\nDataset with Cluster:")
print(df.head())


# ============================================================
# 16. CHECK NUMBER OF CUSTOMERS IN EACH CLUSTER
# ============================================================

# value_counts() counts how many customers belong
# to each cluster.

print("\nCustomers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())


# ============================================================
# 17. VISUALIZE THE CLUSTERS
# ============================================================

# We use the two features:
#
# X_scaled[:, 0] -> Annual Income
# X_scaled[:, 1] -> Spending Score
#
# c=clusters means points are displayed according
# to their cluster label.

plt.figure(figsize=(10, 7))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)

plt.title("Customer Segmentation using K-Means")

plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")

plt.show()


# ============================================================
# 18. GET CLUSTER CENTERS
# ============================================================

# K-Means calculates a center point for every cluster.
#
# These are called centroids.

centers = kmeans.cluster_centers_


# Display centroids

print("\nCluster Centers:")
print(centers)


# ============================================================
# 19. VISUALIZE CLUSTERS + CENTROIDS
# ============================================================

plt.figure(figsize=(10, 7))


# Plot customers

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)


# Plot cluster centers
#
# marker="X" makes the centroid appear as an X.
#
# s=200 makes the centroid larger.

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200
)


plt.title("K-Means Clusters with Centroids")

plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")

plt.show()


# ============================================================
# 20. ANALYZE EACH CLUSTER
# ============================================================

# groupby("Cluster") groups customers according to
# their cluster.
#
# mean() calculates the average value of each feature
# within every cluster.

cluster_analysis = df.groupby("Cluster").mean(
    numeric_only=True
)


print("\nCluster Analysis:")
print(cluster_analysis)


# ============================================================
# 21. CLUSTER ANALYSIS ONLY FOR IMPORTANT FEATURES
# ============================================================

# We can specifically look at:
#
# Annual Income
# Spending Score

cluster_summary = df.groupby("Cluster")[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()


print("\nCluster Summary:")
print(cluster_summary)


# ============================================================
# 22. SILHOUETTE SCORE
# ============================================================

# Silhouette Score measures how well-separated the clusters are.
#
# Higher values generally indicate better-separated clusters.
#
# The score normally ranges from -1 to 1.

silhouette = silhouette_score(
    X_scaled,
    clusters
)


print("\nSilhouette Score:")
print(silhouette)


# ============================================================
# 23. TEST DIFFERENT K VALUES USING SILHOUETTE SCORE
# ============================================================

# Instead of checking only the Elbow Method,
# we can calculate the Silhouette Score for different K values.

silhouette_scores = []


# Test K from 2 to 10.
#
# We start from 2 because silhouette score isn't meaningful
# for a single cluster.

for k in range(2, 11):

    # Create K-Means model.

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )


    # Generate cluster labels.

    labels = model.fit_predict(X_scaled)


    # Calculate silhouette score.

    score = silhouette_score(
        X_scaled,
        labels
    )


    # Store the score.

    silhouette_scores.append(score)


# ============================================================
# 24. PLOT SILHOUETTE SCORES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Silhouette Score")

plt.xticks(range(2, 11))

plt.show()


# ============================================================
# 25. PCA FOR 2D VISUALIZATION
# ============================================================

# PCA stands for Principal Component Analysis.
#
# If we have many features, visualizing them directly is difficult.
#
# PCA can reduce many dimensions to 2 dimensions.
#
# In our current example we only have two features,
# so PCA isn't necessary.
#
# We are including it here so you understand how it can
# be used in larger unsupervised projects.

pca = PCA(
    n_components=2
)


# Transform the scaled data into two principal components.

X_pca = pca.fit_transform(X_scaled)


# Display the shape of PCA data.

print("\nPCA Shape:")
print(X_pca.shape)


# ============================================================
# 26. VISUALIZE PCA + CLUSTERS
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.title("Customer Clusters using PCA")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.show()


# ============================================================
# 27. EXPLAIN CLUSTERS
# ============================================================

# Let's print the average income and spending score
# for each cluster.

print("\n========== CUSTOMER SEGMENTS ==========")


for cluster in sorted(df["Cluster"].unique()):

    # Get customers belonging to this cluster.

    cluster_data = df[
        df["Cluster"] == cluster
    ]


    # Calculate average income.

    average_income = cluster_data[
        "Annual Income (k$)"
    ].mean()


    # Calculate average spending score.

    average_spending = cluster_data[
        "Spending Score (1-100)"
    ].mean()


    # Calculate number of customers.

    customer_count = len(cluster_data)


    print("\nCluster:", cluster)

    print(
        "Number of Customers:",
        customer_count
    )

    print(
        "Average Annual Income:",
        round(average_income, 2)
    )

    print(
        "Average Spending Score:",
        round(average_spending, 2)
    )


# ============================================================
# 28. SAVE FINAL DATASET
# ============================================================

# Save the original dataset along with the newly created
# Cluster column.

df.to_csv(
    "customer_segments.csv",
    index=False
)


print("\nFinal dataset saved as customer_segments.csv")


# ============================================================
# 29. END
# ============================================================

print("\nK-Means Customer Segmentation Completed!")