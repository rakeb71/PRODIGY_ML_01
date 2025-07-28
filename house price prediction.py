# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
X = np.array([500, 700, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500]).reshape(-1, 1)
y = np.array([150000, 175000, 190000, 210000, 250000, 275000, 300000, 330000, 350000, 400000])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
# Step 5: Make predictions on test data
y_pred = model.predict(X_test)

# Step 6: Print actual vs predicted
print("\nPredicted vs Actual:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual Price: ${actual}, Predicted Price: ${predicted:.2f}")

# Step 7: Plot everything in one figure
plt.scatter(X_train, y_train, color='blue', label='Training Data')     # Blue = train
plt.scatter(X_test, y_test, color='red', label='Test Data')           # Red = test
plt.plot(X, model.predict(X), color='green', linewidth=2, label='Regression Line')  # Green = regression line

# Add labels and show plot
plt.xlabel("Square Footage")
plt.ylabel("House Price ($)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
