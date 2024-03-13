import pandas as pd

# Load the dataset
credit_card_data = pd.read_csv("creditcardfraud.csv")

# Display the first few rows of the dataset
print(credit_card_data.head())

# Check for class imbalance
print(credit_card_data['Class'].value_counts())

# Import necessary libraries for preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Separate features and target variable
X = credit_card_data.drop('Class', axis=1)
y = credit_card_data['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Train a classification algorithm (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model's performance
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
