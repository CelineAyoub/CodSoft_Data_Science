import pandas as pd

# Load the dataset
iris_data = pd.read_csv("iris_flower_dataset.csv")

# Display the first few rows of the dataset
print(iris_data.head())

# Check for missing values and data types
print(iris_data.info())

# Check the distribution of the target variable
print(iris_data['species'].value_counts())
