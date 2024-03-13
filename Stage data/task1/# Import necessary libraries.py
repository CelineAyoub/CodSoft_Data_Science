import pandas as pd

# Load the dataset
titanic_data = pd.read_csv("titanic_dataset.csv")

# Display the first few rows of the dataset
print(titanic_data.head())

# Check for missing values
print(titanic_data.isnull().sum())

# Drop unnecessary columns (e.g., PassengerId, Name, Ticket, Cabin)
titanic_data = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing values for Age and Fare with median
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)

# Encode categorical variables (Sex and Embarked)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

# Split the dataset into features (X) and target variable (y)
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
