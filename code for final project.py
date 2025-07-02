# Import necessary libraries
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load your data (replace with actual data loading)
# X_train = pd.read_csv('train_data.csv')
# X_test = pd.read_csv('test_data.csv')

# Identify categorical columns (non-numeric data that needs encoding)
categorical_cols = X_train.select_dtypes(include='object').columns

# One-hot encode categorical variables
# drop_first=True avoids multicollinearity in linear models
X_train_processed = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_processed = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Ensure both datasets have exactly the same columns after encoding
# This handles cases where test set might have different categories
train_cols = X_train_processed.columns
test_cols = X_test_processed.columns

# Add any missing columns to test set (initialize with zeros)
for col in set(train_cols) - set(test_cols):
    X_test_processed[col] = 0  # These represent categories not present in test set

# Add any missing columns to train set (should be rare)
for col in set(test_cols) - set(train_cols):
    X_train_processed[col] = 0  # These represent categories not present in train set

# Enforce identical column order between train and test
X_test_processed = X_test_processed[train_cols]

# Identify numerical columns for scaling (after encoding)
numerical_cols = X_train_processed.select_dtypes(include=np.number).columns

# Initialize StandardScaler (z-score normalization)
scaler = StandardScaler()

# Fit scaler ONLY on training data and transform both sets
# Important to prevent data leakage from test set
X_train_processed[numerical_cols] = scaler.fit_transform(X_train_processed[numerical_cols])
X_test_processed[numerical_cols] = scaler.transform(X_test_processed[numerical_cols])

# Verification outputs (visible in Spyder's variable explorer)
print("Processed training data shape:", X_train_processed.shape)
print("Processed test data shape:", X_test_processed.shape)

# Display first few rows (useful for quick visual check)
print("\nTraining data preview:")
print(X_train_processed.head())

print("\nTest data preview:")
print(X_test_processed.head())

# Verify column alignment (critical for model training)
print("\nColumns match:", all(X_train_processed.columns == X_test_processed.columns))