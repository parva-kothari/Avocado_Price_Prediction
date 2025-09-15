import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# === Load your avocado dataset ===
# Replace this with your dataset file if needed
df = pd.read_csv("avocado.csv")

# === Basic feature engineering ===
df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["type"] = df["type"].map({"conventional": 0, "organic": 1})

df["Volume_per_Bag"] = np.where(df["Total Bags"] > 0, df["Total Volume"] / df["Total Bags"], 0)
df["Small_Bags_Ratio"] = np.where(df["Total Bags"] > 0, df["Small Bags"] / df["Total Bags"], 0)
df["Large_Bags_Ratio"] = np.where(df["Total Bags"] > 0, df["Large Bags"] / df["Total Bags"], 0)
df["Total_PLU_Volume"] = df["4046"] + df["4225"] + df["4770"]
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# One-hot encode regions (drop_first=True to avoid multicollinearity)
region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
df = pd.concat([df, region_dummies], axis=1)

region_dummy_cols = list(region_dummies.columns)

# === Select features ===
selected_features = [
    "Total Volume","4046","4225","4770","Total Bags",
    "Small_Bags_Ratio","Large_Bags_Ratio","type",
    "year","month_sin","month_cos","Volume_per_Bag",
    "Total_PLU_Volume"
] + region_dummy_cols

X = df[selected_features]
y = df["AveragePrice"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Build pipeline ===
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.01))
])

pipeline.fit(X_train, y_train)

# === Save model ===
artifact = {
    "pipeline": pipeline,
    "selected_features": selected_features,
    "region_dummy_cols": region_dummy_cols
}
joblib.dump(artifact, "model_svr.joblib")

print("✅ SVR model trained and saved as model_svr.joblib")
