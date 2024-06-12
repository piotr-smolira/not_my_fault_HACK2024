import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load your datasets
df_no_faults = pd.read_csv('data/LBNL_FDD/LBNL_FDD_Dataset_FCU/FCU_FaultFree.csv', parse_dates=['Datetime'], index_col='Datetime')
df_faults = pd.read_csv('data/LBNL_FDD/LBNL_FDD_Dataset_FCU/FCU_OADMPRStuck_0.csv', parse_dates=['Datetime'], index_col='Datetime')

# Add a label to indicate faults
df_no_faults['label'] = 0
df_faults['label'] = 1

# Combine datasets
df = pd.concat([df_no_faults, df_faults])

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Handle missing values
# df = df.fillna(method='ffill').fillna(method='bfill')

# Normalize the data
df['FCU_DMPR'] = (df['FCU_DMPR'] - df['FCU_DMPR'].mean()) / df['FCU_DMPR'].std()

# # Feature Engineering
# Create lag features
df['lag1'] = df['RM_TEMP'].shift(1)
df['lag2'] = df['RMCLGSPT'].shift(2)
df['lag3'] = df['RMHTGSPT'].shift(3)

# Drop rows with NaN values (introduced by lag features)
df = df.dropna()

# Define features and target
X = df[['FCU_DMPR', 'lag1', 'lag2', 'lag3']]
y = df['label']


# # Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# # Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize feature importance
feature_importance = model.feature_importances_
features = X.columns
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest')
plt.show()

# # Use the model to predict faults
# Assuming you have new data in `new_data.csv`
new_data = pd.read_csv('FCU_OADMPRStuck_80.csv', parse_dates=['timestamp'], index_col='timestamp')

# Preprocess new data
# new_data = new_data.fillna(method='ffill').fillna(method='bfill')
new_data['FCU_DMPR'] = (new_data['FCU_DMPR'] - new_data['FCU_DMPR'].mean()) / new_data['FCU_DMPR'].std()
new_data['lag1'] = new_data['value'].shift(1)
new_data['lag2'] = new_data['value'].shift(2)
new_data['lag3'] = new_data['value'].shift(3)
new_data = new_data.dropna()

# Predict faults in new data
X_new = new_data[['FCU_DMPR', 'lag1', 'lag2', 'lag3']]
new_data['predicted_fault'] = model.predict(X_new)

# Visualize the detected faults
plt.figure(figsize=(10, 6))
plt.plot(new_data.index, new_data['FCU_DMPR'], label='Time Series')
plt.scatter(new_data.index[new_data['predicted_fault'] == 1], new_data['FCU_DMPR'][new_data['predicted_fault'] == 1], color='red', label='Detected Fault')
plt.xlabel('Time')
plt.ylabel('FCU_DMPR')
plt.title('Detected Faults in New Time Series Data')
plt.legend()
plt.show()
