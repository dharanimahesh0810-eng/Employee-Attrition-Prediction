"""
train_model.py
----------------
Trains the Employee Attrition model (based on the original notebook logic)
and saves the artifacts needed by app.py (the Streamlit interface).

Run this ONCE before starting the Streamlit app:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

# 1. Load data -----------------------------------------------------------
# Make sure HR-Employee-Attrition.csv is in the same folder as this script
df = pd.read_csv("HR-Employee-Attrition.csv")

# 2. Encode target (Yes/No -> 1/0), same as the notebook -----------------
df['Attrition'] = (
    df['Attrition'].astype(str).str.strip()
    .map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
)

# 3. Drop columns -----------------------------------------------------
# - the highly-correlated columns dropped in the notebook
# - constant / ID columns that carry no predictive signal
#   (EmployeeCount, EmployeeNumber, Over18, StandardHours)
drop_cols = [
    'JobLevel', 'PerformanceRating', 'YearsWithCurrManager',
    'YearsInCurrentRole', 'TotalWorkingYears',
    'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 4. Split features / target ---------------------------------------------
X = df.drop(columns=['Attrition'])
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)

# 5. Scale numeric columns -------------------------------------------
# Fit the scaler ONLY on training data (avoids data leakage into the test set)
numerical_columns = X_train.select_dtypes(include=['float', 'int']).columns.tolist()
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# 6. Train pipeline: OneHotEncoder + LogisticRegression -------------------
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))

# 7. Save artifacts needed by the Streamlit app ---------------------------
joblib.dump(model, "attrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save feature metadata so the Streamlit app can build its form automatically
feature_info = {}
for col in X.columns:
    if col in numerical_columns:
        feature_info[col] = {
            "type": "numeric",
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
        }
    else:
        feature_info[col] = {
            "type": "categorical",
            "options": sorted(X[col].dropna().unique().tolist()),
        }

joblib.dump(
    {
        "columns": X.columns.tolist(),
        "numerical_columns": numerical_columns,
        "feature_info": feature_info,
    },
    "feature_info.pkl",
)

print("Saved: attrition_model.pkl, scaler.pkl, feature_info.pkl")
