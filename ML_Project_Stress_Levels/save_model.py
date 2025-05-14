import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load and prepare data
df = pd.read_csv("Cleaned_Final.csv")

# 1. Drop PHQ items and leakage-related columns
phq_cols = [col for col in df.columns if col.startswith("PHQ")]
leakage_cols = [
    "Depression_Value", "Anxiety_Value", "Stress_Value",
    "Anxiety_Label", "Stress_Label"
]
columns_to_drop = [col for col in phq_cols + leakage_cols if col in df.columns]

# 2. Get target before dropping
y = df["Depression_Label"]

# 3. Drop leakage columns
X = df.drop(columns=columns_to_drop + ["Depression_Label"])

# 4. Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. Standardize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train XGBoost model
model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    scale_pos_weight=len(y_encoded) / (2 * np.bincount(y_encoded)[1:].sum())
)
model.fit(X_scaled, y_encoded)

# Save the model, scaler, and label encoder
pickle.dump(model, open('xgboost_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))

print("Model, scaler, and label encoder have been saved successfully!") 