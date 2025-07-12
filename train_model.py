import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 🔄 Load your Trade Me data (replace with your actual file path)
df = pd.read_csv("trademe_data.csv")

# 🧹 Filter and clean (example only — adapt based on your real data)
df = df.dropna(subset=["bedrooms", "bathrooms", "floor_area", "land_area", "price"])

# 🧠 Features and target
X = df[["bedrooms", "bathrooms", "floor_area", "land_area"]]
y = df["price"]

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Save model to use in your app
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
