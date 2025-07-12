import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# 🔄 Load your dataset
data_path = os.path.join("data", "trademe_data.csv")
df = pd.read_csv(data_path)

# Clean up (adapt if needed)
df = df.dropna(subset=["bedrooms", "bathrooms", "floor_area", "land_area", "price"])

# Define features and target
X = df[["bedrooms", "bathrooms", "floor_area", "land_area"]]
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, os.path.join("model", "model.pkl"))

print("✅ Model saved to /model/model.pkl")
