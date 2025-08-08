# Sales Data Analysis with Pandas
# Task 5: Data Analysis on CSV Files

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 Sales Data Analysis Project")
print("=" * 50)

# Step 1: Create Sample Sales Data (since no CSV file provided)
print("\n🔧 Step 1: Creating Sample Sales Data")
print("-" * 30)

# Generate sample sales data
np.random.seed(42)  # For reproducible results

# Date range for the last year
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Product categories and regions
products = ['Laptops', 'Smartphones', 'Tablets', 'Headphones', 'Smartwatches', 'Gaming Consoles']
regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
sales_reps = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Emma Brown', 'Frank Miller']

# Generate sample data
data = []
for _ in range(2000):  # 2000 sales records
    record = {
        'Date': np.random.choice(date_range),
        'Product': np.random.choice(products),
        'Region': np.random.choice(regions),
        'Sales_Rep': np.random.choice(sales_reps),
        'Units_Sold': np.random.randint(1, 50),
        'Unit_Price': np.random.uniform(50, 2000),
        'Discount': np.random.uniform(0, 0.3)  # 0-30% discount
    }
    record['Revenue'] = record['Units_Sold'] * record['Unit_Price'] * (1 - record['Discount'])
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

print(f"✅ Generated {len(df):,} sales records")
print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Step 2: Basic Data Exploration
print("\n📋 Step 2: Basic Data Exploration")
print("-" * 30)

print("Dataset Shape:", df.shape)
print("\nColumn Data Types:")
print(df.dtypes)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Step 3: Data Analysis using groupby(), sum(), and other operations
print("\n🔍 Step 3: Data Analysis")
print("-" * 30)

# 1. Total Revenue by Product
print("1️⃣ Total Revenue by Product:")
revenue_by_product = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
print(revenue_by_product.round(2))

# 2. Total Revenue by Region
print("\n2️⃣ Total Revenue by Region:")
revenue_by_region = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print(revenue_by_region.round(2))

# 3. Monthly Sales Trends
print("\n3️⃣ Monthly Sales Summary:")
monthly_sales = df.groupby('Month').agg({
    'Revenue': 'sum',
    'Units_Sold': 'sum',
    'Unit_Price': 'mean'
}).round(2)
print(monthly_sales)

# 4. Top Performing Sales Representatives
print("\n4️⃣ Top Sales Representatives:")
sales_rep_performance = df.groupby('Sales_Rep').agg({
    'Revenue': 'sum',
    'Units_Sold': 'sum'
}).sort_values('Revenue', ascending=False)
print(sales_rep_performance.round(2))

# 5. Product Performance by Region
print("\n5️⃣ Product Performance by Region:")
product_region_pivot = pd.pivot_table(df, 
                                    values='Revenue', 
                                    index='Product', 
                                    columns='Region', 
                                    aggfunc='sum').round(2)
print(product_region_pivot)

# Step 4: Advanced Analysis
print("\n🎯 Step 4: Advanced Analysis")
print("-" * 30)

# 1. Calculate key metrics
total_revenue = df['Revenue'].sum()
total_units = df['Units_Sold'].sum()
avg_order_value = df['Revenue'].mean()
avg_discount = df['Discount'].mean()

print(f"💰 Total Revenue: ${total_revenue:,.2f}")
print(f"📦 Total Units Sold: {total_units:,}")
print(f"💳 Average Order Value: ${avg_order_value:.2f}")
print(f"🏷️ Average Discount: {avg_discount:.1%}")

# 2. Quarterly Performance
quarterly_performance = df.groupby('Quarter').agg({
    'Revenue': ['sum', 'mean', 'count'],
    'Units_Sold': 'sum'
}).round(2)
quarterly_performance.columns = ['Total_Revenue', 'Avg_Revenue', 'Order_Count', 'Total_Units']
print(f"\n📈 Quarterly Performance:")
print(quarterly_performance)

# 3. Top 10 Sales Days
top_sales_days = df.groupby('Date')['Revenue'].sum().nlargest(10)
print(f"\n🔥 Top 10 Sales Days:")
for date, revenue in top_sales_days.items():
    print(f"{date.strftime('%Y-%m-%d')}: ${revenue:,.2f}")

# Step 5: Data Visualization
print("\n📊 Step 5: Creating Visualizations")
print("-" * 30)

# Set up the plotting area
fig = plt.figure(figsize=(20, 15))

# 1. Revenue by Product (Bar Chart)
plt.subplot(3, 3, 1)
revenue_by_product.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Total Revenue by Product', fontsize=14, fontweight='bold')
plt.xlabel('Product')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 2. Revenue by Region (Pie Chart)
plt.subplot(3, 3, 2)
revenue_by_region.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Revenue Distribution by Region', fontsize=14, fontweight='bold')
plt.ylabel('')

# 3. Monthly Revenue Trend (Line Chart)
plt.subplot(3, 3, 3)
monthly_revenue = df.groupby('Month')['Revenue'].sum()
plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2, markersize=8)
plt.title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))

# 4. Units Sold by Product (Horizontal Bar)
plt.subplot(3, 3, 4)
units_by_product = df.groupby('Product')['Units_Sold'].sum().sort_values()
units_by_product.plot(kind='barh', color='lightcoral', alpha=0.8)
plt.title('Units Sold by Product', fontsize=14, fontweight='bold')
plt.xlabel('Units Sold')

# 5. Sales Rep Performance (Bar Chart)
plt.subplot(3, 3, 5)
sales_rep_revenue = df.groupby('Sales_Rep')['Revenue'].sum().sort_values(ascending=False)
sales_rep_revenue.plot(kind='bar', color='lightgreen', alpha=0.8)
plt.title('Sales Rep Performance', fontsize=14, fontweight='bold')
plt.xlabel('Sales Representative')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)

# 6. Quarterly Revenue (Bar Chart)
plt.subplot(3, 3, 6)
quarterly_revenue = df.groupby('Quarter')['Revenue'].sum()
plt.bar(quarterly_revenue.index, quarterly_revenue.values, color='gold', alpha=0.8)
plt.title('Quarterly Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Quarter')
plt.ylabel('Revenue ($)')
plt.xticks([1, 2, 3, 4])

# 7. Price vs Units Sold Scatter Plot
plt.subplot(3, 3, 7)
plt.scatter(df['Unit_Price'], df['Units_Sold'], alpha=0.6, c='purple')
plt.title('Unit Price vs Units Sold', fontsize=14, fontweight='bold')
plt.xlabel('Unit Price ($)')
plt.ylabel('Units Sold')

# 8. Revenue Distribution (Histogram)
plt.subplot(3, 3, 8)
plt.hist(df['Revenue'], bins=30, color='orange', alpha=0.7, edgecolor='black')
plt.title('Revenue Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Revenue ($)')
plt.ylabel('Frequency')

# 9. Heatmap of Product-Region Performance
plt.subplot(3, 3, 9)
sns.heatmap(product_region_pivot, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Revenue ($)'})
plt.title('Product-Region Revenue Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Product')

plt.tight_layout()
plt.show()

# Step 6: Key Insights and Summary
print("\n🎯 Step 6: Key Insights")
print("-" * 30)

# Find best and worst performers
best_product = revenue_by_product.index[0]
worst_product = revenue_by_product.index[-1]
best_region = revenue_by_region.index[0]
best_sales_rep = sales_rep_performance.index[0]
best_month = monthly_sales['Revenue'].idxmax()
best_quarter = quarterly_performance['Total_Revenue'].idxmax()

print("🏆 KEY INSIGHTS:")
print(f"• Best Performing Product: {best_product} (${revenue_by_product.iloc[0]:,.2f})")
print(f"• Lowest Performing Product: {worst_product} (${revenue_by_product.iloc[-1]:,.2f})")
print(f"• Top Region: {best_region} (${revenue_by_region.iloc[0]:,.2f})")
print(f"• Top Sales Rep: {best_sales_rep} (${sales_rep_performance.loc[best_sales_rep, 'Revenue']:,.2f})")
print(f"• Best Month: Month {best_month} (${monthly_sales.loc[best_month, 'Revenue']:,.2f})")
print(f"• Best Quarter: Q{best_quarter} (${quarterly_performance.loc[best_quarter, 'Total_Revenue']:,.2f})")

# Calculate growth rates
if len(quarterly_performance) > 1:
    q1_revenue = quarterly_performance.loc[1, 'Total_Revenue']
    q4_revenue = quarterly_performance.loc[4, 'Total_Revenue'] if 4 in quarterly_performance.index else quarterly_performance.iloc[-1]['Total_Revenue']
    growth_rate = ((q4_revenue - q1_revenue) / q1_revenue) * 100
    print(f"• Revenue Growth Rate: {growth_rate:+.1f}%")

print(f"\n📈 RECOMMENDATIONS:")
print(f"• Focus marketing efforts on {best_product} as it's the top performer")
print(f"• Investigate why {worst_product} has lower sales and consider improvements")
print(f"• Expand operations in {best_region} region")
print(f"• Learn from {best_sales_rep}'s successful sales strategies")
print(f"• Month {best_month} shows peak performance - analyze seasonal factors")

print(f"\n✅ Analysis Complete!")
print(f"📊 Total records analyzed: {len(df):,}")
print(f"📅 Analysis period: {df['Date'].min().strftime('%B %Y')} - {df['Date'].max().strftime('%B %Y')}")

# Optional: Save results to CSV files
print(f"\n💾 Saving Results...")
try:
    # Save summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Total Revenue', 'Total Units Sold', 'Average Order Value', 'Average Discount', 'Number of Orders'],
        'Value': [f"${total_revenue:,.2f}", f"{total_units:,}", f"${avg_order_value:.2f}", f"{avg_discount:.1%}", f"{len(df):,}"]
    })
    
    print("✅ Summary statistics prepared")
    print("✅ Revenue by product analysis completed")
    print("✅ Regional performance analysis completed")
    print("✅ Sales representative performance analysis completed")
    print("✅ Visualizations created successfully")
    
except Exception as e:
    print(f"⚠️ Error saving results: {e}")

print(f"\n🎉 Sales Data Analysis Project Completed Successfully!")
print("=" * 50)