import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load Data
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge Data
data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Feature Engineering
# Aggregate transaction data per customer
customer_features = data.groupby("CustomerID").agg(
    total_spend=pd.NamedAgg(column="TotalValue", aggfunc="sum"),
    transaction_count=pd.NamedAgg(column="TransactionID", aggfunc="count"),
    avg_transaction_value=pd.NamedAgg(column="TotalValue", aggfunc="mean"),
    favorite_category=pd.NamedAgg(column="Category", aggfunc=lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
).reset_index()

# Convert categorical features to one-hot encoding (e.g., Region and Favorite Category)
region_encoded = pd.get_dummies(customers[["CustomerID", "Region"]], columns=["Region"])
category_encoded = pd.get_dummies(customer_features[["CustomerID", "favorite_category"]], columns=["favorite_category"])

# Merge with numerical features
customer_features = customer_features.merge(region_encoded, on="CustomerID", how="left").merge(category_encoded, on="CustomerID", how="left")

# Drop non-numeric columns
final_features = customer_features.drop(columns=["CustomerID", "favorite_category"]).fillna(0)

# Normalize the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(final_features)

# Calculate Similarity
similarity_matrix = cosine_similarity(normalized_features)

# Align Customer IDs with Features
aligned_customer_ids = customer_features["CustomerID"].tolist()

# Create Recommendations
recommendations = {}

for idx, customer_id in enumerate(aligned_customer_ids):
    # Get similarity scores for the customer
    similarities = list(enumerate(similarity_matrix[idx]))
    # Sort by similarity score in descending order (exclude self)
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_matches = [(aligned_customer_ids[i], round(score, 2)) for i, score in similarities if i != idx][:3]
    recommendations[customer_id] = top_matches

# Filter Recommendations for CustomerIDs C0001 to C0020
filtered_recommendations = {
    cust_id: recommendations[cust_id]
    for cust_id in aligned_customer_ids
    if cust_id.startswith("C000") and int(cust_id[1:]) <= 20
}

# Save to CSV
output = pd.DataFrame(
    {
        "cust_id": filtered_recommendations.keys(),
        "recommendations": [str(recommendations) for recommendations in filtered_recommendations.values()],
    }
)
output.to_csv("Lookalike.csv", index=False)

print("Lookalike model and recommendations saved as 'FirstName_LastName_Lookalike.csv'.")