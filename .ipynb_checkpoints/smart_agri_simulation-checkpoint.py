# =======================
# SMART AGRICULTURE SIMULATION
# =======================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Simulate 1000 sensor readings
np.random.seed(42)

data = pd.DataFrame({
    "soil_moisture": np.random.uniform(20, 80, 1000),
    "soil_temp": np.random.uniform(10, 35, 1000),
    "air_temp": np.random.uniform(15, 40, 1000),
    "humidity": np.random.uniform(30, 90, 1000),
    "light": np.random.uniform(200, 1000, 1000),
    "rainfall": np.random.uniform(0, 50, 1000),
})

# Simulate yield (depends on moisture, light, temp)
data["yield"] = (
    0.4 * data["soil_moisture"] +
    0.3 * data["light"] / 10 -
    0.2 * abs(data["air_temp"] - 25) +
    np.random.normal(0, 5, 1000)
)

# Train/Test split
X = data.drop("yield", axis=1)
y = data["yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Model Performance:")
print("MAE:", mae)
print("R² Score:", r2)

# Feature importance
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")
plt.show()
