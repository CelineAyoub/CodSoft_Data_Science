# Make predictions on the testing set
predictions = model.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R-squared:", r2_score(y_test, predictions))
