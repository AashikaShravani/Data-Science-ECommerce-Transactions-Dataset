import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from fpdf import FPDF

# Load the data
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")
products = pd.read_csv("Products.csv")

# Merge customer profile with transaction data
data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Feature Engineering: Aggregating transaction data per customer
customer_features = data.groupby("CustomerID").agg(
    total_spend=pd.NamedAgg(column="TotalValue", aggfunc="sum"),
    transaction_count=pd.NamedAgg(column="TransactionID", aggfunc="count"),
    avg_transaction_value=pd.NamedAgg(column="TotalValue", aggfunc="mean"),
    favorite_category=pd.NamedAgg(column="Category", aggfunc=lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
).reset_index()

# Convert categorical features (like 'Region' and 'Favorite Category') to one-hot encoding
region_encoded = pd.get_dummies(customers[["CustomerID", "Region"]], columns=["Region"])
category_encoded = pd.get_dummies(customer_features[["CustomerID", "favorite_category"]], columns=["favorite_category"])

# Merge encoded features with aggregated transaction data
customer_features = customer_features.merge(region_encoded, on="CustomerID", how="left").merge(category_encoded, on="CustomerID", how="left")

# Drop non-numeric columns (such as 'CustomerID' and 'favorite_category') and fill missing values
final_features = customer_features.drop(columns=["CustomerID", "favorite_category"]).fillna(0)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(final_features)

# Perform KMeans Clustering (Choose an appropriate number of clusters, e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
customer_features["Cluster"] = kmeans.fit_predict(normalized_features)

# Calculate clustering metrics: Davies-Bouldin Index
db_index = davies_bouldin_score(normalized_features, customer_features["Cluster"])
print(f"Davies-Bouldin Index (DB Index): {db_index}")

# Visualize the clusters
# Use PCA to reduce dimensionality for visualization (2D space)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(normalized_features)

# Create a DataFrame with the 2D PCA components and cluster labels
pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
pca_df["Cluster"] = customer_features["Cluster"]

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette="viridis", s=100, edgecolor='black', legend="full")
plt.title("Customer Segmentation Clusters (PCA Reduced Dimensions)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()

# Save the plot to a file
plt.savefig("cluster_visualization.png")
plt.close()

# Additional metrics: You can also compute other metrics such as inertia and silhouette score
inertia = kmeans.inertia_
print(f"KMeans Inertia: {inertia}")

# Save the clustering results to a CSV file
customer_features["Cluster"] = customer_features["Cluster"].astype(str)
customer_features.to_csv("Customer_Segmentation_Clusters.csv", index=False)

# Create PDF report using FPDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Add title
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Customer Segmentation Report", ln=True, align="C")
pdf.ln(10)

# Add clustering metrics
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, f"Number of Clusters: {len(customer_features['Cluster'].unique())}", ln=True)
pdf.cell(200, 10, f"Davies-Bouldin Index (DB Index): {db_index:.3f}", ln=True)
pdf.cell(200, 10, f"KMeans Inertia: {inertia:.2f}", ln=True)
pdf.ln(10)

# Add the clustering visualization image to the PDF
pdf.image("cluster_visualization.png", x=20, w=170)
pdf.ln(80)

# Add conclusion
pdf.multi_cell(0, 10, "The clustering results show the distribution of customers into different segments based on their transaction history and profile information. The DB Index value indicates the quality of the clustering, with lower values representing better clustering quality.")

# Output the PDF
pdf_output_path = "Aashika_Shravani_Clustering.pdf"
pdf.output(pdf_output_path)

print(f"Clustering report saved as '{pdf_output_path}'.")