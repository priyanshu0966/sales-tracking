Task 5: Data Analysis on CSV Files

Objective
Analyze sales data using Python and Pandas to generate meaningful insights and visualizations.

🛠️ Tools & Libraries Used
Python 3

Pandas (for data manipulation)

Matplotlib / Seaborn (for plotting)

Jupyter Notebook / Google Colab (for code execution)

Dataset
A CSV file containing sales data with columns like:

Date
Product
Region
Sales
Revenue

🚀 Steps Performed
1.Import Required Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

2.Load CSV File

df = pd.read_csv('sales_data.csv')

3.Explore the Dataset

df.head(), df.info(), df.describe()

4.Group and Analyze

a.Total sales by region:

df.groupby('Region')['Sales'].sum()

b.Total revenue by product:

df.groupby('Product')['Revenue'].sum()

5.Visualize the Insights

Bar chart for sales by region

Pie chart for product-wise revenue

Time series plot for monthly sales

📈 Sample Charts
Bar Chart: Sales by Region

Pie Chart: Revenue Distribution by Product

Line Plot: Sales Over Time
