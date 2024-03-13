import pandas as pd

# Load the dataset
sales_data = pd.read_csv("sales_prediction_dataset.csv")

# Display the first few rows of the dataset
print(sales_data.head())

# Check for missing values and data types
print(sales_data.info())
