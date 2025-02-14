import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import joblib
from datetime import datetime

# Load training data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
df_train = pd.read_csv(train_url)

if 'timestamp' in df_train.columns:
    df_train['timestamp'] = df_train['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df_train['hour'] = df_train['timestamp'].dt.hour
    df_train['dayofweek'] = df_train['timestamp'].dt.dayofweek
else:
    df_train['hour'] = 0
    df_train['dayofweek'] = 0

df_train['is_weekend'] = df_train['dayofweek'].isin([5, 6]).astype(int)

X_train = df_train[['hour', 'dayofweek', 'is_weekend']]
y_train = df_train['trips']

# Define and train the model
model = LinearGAM(s(0) + s(1) + s(2))
modelFit = model.fit(X_train, y_train)

# Save the model for future use
joblib.dump(modelFit, 'pygam_taxi_model.pkl')

# Load test data
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
df_test = pd.read_csv(test_url)

if 'timestamp' in df_test.columns:
    df_test['timestamp'] = df_test['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df_test['hour'] = df_test['timestamp'].dt.hour
    df_test['dayofweek'] = df_test['timestamp'].dt.dayofweek
else:
    df_test['hour'] = 0
    df_test['dayofweek'] = 0

df_test['is_weekend'] = df_test['dayofweek'].isin([5, 6]).astype(int)

X_test = df_test[['hour', 'dayofweek', 'is_weekend']]

# Generate predictions
pred = modelFit.predict(X_test)

# Save predictions
np.savetxt("taxi_predictions.csv", pred, delimiter=",")