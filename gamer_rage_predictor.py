# 🎮 Gamer Rage Level Predictor using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'hours_played': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'losses':       [0, 1, 2, 3, 3, 4, 5, 6, 7, 8],
    'rage_level':   [10, 20, 30, 40, 50, 60, 70, 80, 85, 95]
}

df = pd.DataFrame(data)

# Features and target
X = df[['hours_played', 'losses']]
y = df['rage_level']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Evaluation
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Plot actual vs predicted
plt.scatter(y, y_pred, color='orange')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.xlabel("Actual Rage Level")
plt.ylabel("Predicted Rage Level")
plt.title("🎮 Gamer Rage Predictor")
plt.grid(True)
plt.show()
