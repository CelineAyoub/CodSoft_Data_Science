# Drop unnecessary columns
sales_data = sales_data.drop(['Unnamed: 0', 'Product'], axis=1)

# Handle missing values if any
# For simplicity, let's assume that missing values are already handled

# Split the dataset into features (X) and target variable (y)
X = sales_data.drop('Sales', axis=1)
y = sales_data['Sales']

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
