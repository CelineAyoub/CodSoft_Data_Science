# Split the dataset into features (X) and target variable (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Encoding categorical variable 'species'
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
