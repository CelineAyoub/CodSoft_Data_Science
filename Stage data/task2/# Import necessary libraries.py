import pandas as pd

# Load the dataset
movies_data = pd.read_csv("imdb_india_movies.csv")

# Display the first few rows of the dataset
print(movies_data.head())

# Check for missing values
print(movies_data.isnull().sum())

# Drop unnecessary columns
movies_data = movies_data.drop(['imdb_title_id', 'original_title', 'date_published', 'duration', 'country', 'language', 'writer', 'production_company', 'budget', 'usa_gross_income', 'worlwide_gross_income', 'metascore', 'reviews_from_users', 'reviews_from_critics'], axis=1)

# Fill missing values for 'director', 'actors', and 'description'
movies_data['director'].fillna('Unknown', inplace=True)
movies_data['actors'].fillna('Unknown', inplace=True)
movies_data['description'].fillna('Unknown', inplace=True)

# Convert 'avg_vote' to numeric
movies_data['avg_vote'] = pd.to_numeric(movies_data['avg_vote'], errors='coerce')

# Drop rows with missing values in 'avg_vote'
movies_data.dropna(subset=['avg_vote'], inplace=True)

# Encode categorical variables (genre)
movies_data = pd.get_dummies(movies_data, columns=['genre'], drop_first=True)

# Convert 'director' and 'actors' to lists
movies_data['director'] = movies_data['director'].apply(lambda x: x.split(','))
movies_data['actors'] = movies_data['actors'].apply(lambda x: x.split(','))

# Split the dataset into features (X) and target variable (y)
X = movies_data.drop('avg_vote', axis=1)
y = movies_data['avg_vote']

# Convert lists in 'director' and 'actors' to binary features
for director in set([director for sublist in X['director'] for director in sublist]):
    X['director_' + director] = X['director'].apply(lambda x: 1 if director in x else 0)
    
for actor in set([actor for sublist in X['actors'] for actor in sublist]):
    X['actor_' + actor] = X['actors'].apply(lambda x: 1 if actor in x else 0)

# Drop 'director' and 'actors' columns
X = X.drop(['director', 'actors', 'description'], axis=1)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
