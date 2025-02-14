import pandas as pd
import numpy as np
from pygam import LinearGAM, s

# Load the training data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
df_train = pd.read_csv(train_url)

# Ensure 'Timestamp' column exists before parsing dates
if 'Timestamp' in df_train.columns:
    df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'])
else:
    raise ValueError("Column 'Timestamp' not found in training data.")

def feature_engineering(df):
    """Extract useful time-based features."""
    df['hour'] = df['hour']
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    return df

# Apply feature engineering
df_train = feature_engineering(df_train)

# Define independent (X) and dependent (y) variables
X_train = df_train[['hour', 'dayofweek', 'month']]
y_train = df_train['trips']

# Define and train the GAM model
model = LinearGAM(s(0) + s(1) + s(2))  # Smooth splines for each feature
modelFit = model.fit(X_train, y_train)

# Load the test dataset
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
df_test = pd.read_csv(test_url)

# Ensure 'Timestamp' column exists before parsing dates
if 'Timestamp' in df_test.columns:
    df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'])
else:
    raise ValueError("Column 'Timestamp' not found in test data.")

# Apply feature engineering
df_test = feature_engineering(df_test)
X_test = df_test[['hour', 'dayofweek', 'month']]

# Make predictions for the test period
pred = modelFit.predict(X_test)

# Store predictions in the test dataset
df_test['predicted_trips'] = pred

# Save predictions to a CSV file
df_test[['Timestamp', 'predicted_trips']].to_csv("nyc_taxi_predictions.csv", index=False)

print("Model training and prediction complete. Predictions saved to nyc_taxi_predictions.csv.")