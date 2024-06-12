import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('data/LBNL_FDD/LBNL_FDD_Dataset_FCU/FCU_OADMPRStuck_0.csv')

# Convert the 'timestamp' column to datetime format 
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Set the 'timestamp' column as the index
data.set_index('Datetime', inplace=True)

# Plot each time series
# for column in (data.columns):
column_array = ['FCU_DMPR', 'FCU_DMPR_DM', 'FCU_OAT']
for column in (column_array):
    
    plt.plot(data.index, data[column], label=column)

# Customize the plot
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()

# Show the plot
plt.show()