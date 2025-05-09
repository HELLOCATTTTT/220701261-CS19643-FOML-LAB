# model/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import pickle
import os

# Load the dataset
data_path = os.path.join('..', 'data', 'blood_don.csv')
data = pd.read_csv(data_path)

# Features and Target
X = data.drop('target', axis=1)
y = data['target']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Imbalance with SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

# Train XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_resampled, y_resampled)

# Evaluate
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
model_path = os.path.join('blood_donor_xgb_model.pkl')
pickle.dump(model, open(model_path, 'wb'))
print("Model saved successfully!")
