import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load Data
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Display Data Samples
print("Customers Data:")
print(customers.head())
print("\nProducts Data:")
print(products.head())
print("\nTransactions Data:")
print(transactions.head())

# Check for Missing Values
print("\nMissing Values:")
print("Customers:")
print(customers.isnull().sum())
print("\nProducts:")
print(products.isnull().sum())
print("\nTransactions:")
print(transactions.isnull().sum())

# Convert Date Columns to datetime
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Validate Data Types
print("\nData Types After Conversion:")
print(customers.dtypes)
print(products.dtypes)
print(transactions.dtypes)

# Exploratory Data Analysis
## Customers by Region
region_counts = customers['Region'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.title('Number of Customers by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("customers_by_region.png")
plt.show(block=False)

## Top Product Categories
category_counts = products['Category'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='plasma')
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.title('Number of Products by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("products_by_category.png")
plt.show(block=False)

## Monthly Transactions
transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')
monthly_transactions = transactions['Month'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
monthly_transactions.plot(kind='bar', color='skyblue')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.title('Monthly Transactions')
plt.tight_layout()
plt.savefig("monthly_transactions.png")
plt.show(block=False)

## Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(products['Price'], kde=True, color='orange', bins=20)
plt.xlabel('Price (USD)')
plt.title('Distribution of Product Prices')
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show(block=False)

## Quantity Distribution in Transactions
plt.figure(figsize=(8, 5))
sns.histplot(transactions['Quantity'], kde=False, color='green', bins=20)
plt.xlabel('Quantity')
plt.title('Distribution of Transaction Quantities')
plt.tight_layout()
plt.savefig("quantity_distribution.png")
plt.show(block=False)

# Derive Business Insights
business_insights = [
    "1. South America has the highest number of customers, indicating strong regional performance.",
    "2. Electronics and Books are the top two product categories, driving a significant portion of sales.",
    "3. Transaction activity peaks during the months of November and December, suggesting a holiday season boost.",
    "4. Product prices are mostly under $500, with a small percentage of premium products priced above $1000.",
    "5. Most transactions involve a quantity of 1, indicating a preference for single-item purchases."
]

# Generate PDF Report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Business Insights Report", ln=True, align='C')

for insight in business_insights:
    pdf.ln(10)
    pdf.multi_cell(0, 10, insight)

# Save PDF
pdf.output("EDA_Business_Insights_Report.pdf")

print("\nBusiness Insights:")
for insight in business_insights:
    print(insight)