#!/usr/bin/env python
# coding: utf-8

# # Addepar Stock Index Forecasting

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_frames = []
for year in range(2010, 2023):
    df = pd.read_csv(f"C:/Users/shoai/OneDrive/Desktop/Edinburgh Spring 2024/HackTheBurgh/Addepar/Global Markets Data/{year}_Global_Markets_Data.csv")
    data_frames.append(df)
    

# Combines all data from all years into a single df
full_data = pd.concat(data_frames)


# In[3]:


full_data


# 
# ## We will determine trends based on the following factors:
# ### 1 - The combination of closing index and volume
# ### 2 - Month of year
# 
# ### We will also use the GDP increase/decrease percentage that is relevant for each stock index to further analyze the trends

# In[4]:


# Calculating percentage change in closing price and volume
# Converting the 'Date' column to datetime 
full_data['Date'] = pd.to_datetime(full_data['Date'])

# Grouping by 'Ticker' and 'Month', then calculating the percentage change
full_data.set_index('Date', inplace=True)
monthly_data = full_data.groupby(['Ticker', pd.Grouper(freq='M')]).agg({'Close': 'last', 'Volume': 'sum'})

# Calculating monthly percentage changes for 'Close'
monthly_data['Close_change'] = monthly_data.groupby('Ticker')['Close'].pct_change()

# Since volume is already aggregated as a sum over the month, we compare it to the previous month
monthly_data['Volume_change'] = monthly_data.groupby('Ticker')['Volume'].pct_change()

full_data.reset_index(inplace=True)

# # Turning Date back into a column
monthly_data.reset_index(inplace=True)

# Creating cyclical features for each month to make room for better analysis
monthly_data['Month'] = pd.to_datetime(monthly_data['Date']).dt.month
monthly_data['Month_sin'] = np.sin(2 * np.pi * monthly_data['Month']/12)
monthly_data['Month_cos'] = np.cos(2 * np.pi * monthly_data['Month']/12)



# In[6]:


monthly_data


# In[7]:


# Identifying which ticker indicates which country, to be able
# to match the suitable gdp to the ticker
ticker_to_country = {
    '^NYA': 'United Kingdom',          # NYSE Composite (New York Stock Exchange)
    '^IXIC': 'United States',        # NASDAQ Composite
    '^FTSE': 'United Kingdom',         # FTSE 100 Index (Financial Times Stock Exchange)
    '^NSEI': 'India',      # Nifty 50 (National Stock Exchange of India)
    '^BSESN': 'India',     # BSE SENSEX (Bombay Stock Exchange)
    '^N225': 'Japan',      # Nikkei 225
    '000001.SS': 'China',  # SSE Composite Index (Shanghai Stock Exchange)
    '^N100': 'Eurozone',   # Euronext 100 (European Stock Exchange)
    '^DJI': 'United States',         # Dow Jones Industrial Average
    '^GSPC': 'United States',        # S&P 500 Index
    'GC=F': 'Global',      # Gold Futures (Global Commodity)
    'CL=F': 'Global'       # Crude Oil Futures (Global Commodity)
}

# Loading gdp data per coutnry from the gdp dataset
gdp_data = pd.read_csv(r"C:\Users\shoai\OneDrive\Desktop\Edinburgh Spring 2024\HackTheBurgh\Addepar\GDP by Country\imf-dm-export-20230513.csv")

# Setting country as index, to be able to get the gdp per country for a specific year as in this dataset the columns correspond
# to years and each country is in a row
gdp_data = gdp_data.set_index('Country')

# converting the date to a datetime datatype for easy access and trimming
monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])


# Initializing an empty dictionary to hold the GDP data (excluding Eurozone and Global commodities)
gdp_dict = {}

monthly_data['Country'] = monthly_data['Ticker'].map(ticker_to_country)

# Adding a new column for GDP growth in monthly_data
monthly_data['GDP_Growth'] = 0.0

# Country Name
countries = ticker_to_country.values()

# Populate the dictionary with the data from gdp_data DataFrame
for country in countries:
    if country not in gdp_dict and not country in ['Eurozone', 'Global']:
        gdp_dict[country] = {}
        
    for year in range(2010, 2023): 
            if country not in ['Eurozone', 'Global']:
                gdp_value = gdp_data.loc[country, str(year)]
                if gdp_value != 'no data':
                    gdp_dict[country][str(year)] = gdp_value
                else:
                    gdp_dict[country][str(year)] = None  
        

global_gdp_growth = {
    2022: 3.08,
    2021: 6.02,
    2020: -3.07,
    2019: 2.59,
    2018: 3.29,
    2017: 3.39,
    2016: 2.81,
    2015: 3.08,
    2014: 3.07,
    2013: 2.81,
    2012: 2.71,
    2011: 3.32,
    2010: 4.54
}

european_union_gdp_growth = {
    2022: 3.54,
    2021: 5.47,
    2020: -5.67,
    2019: 1.81,
    2018: 2.07,
    2017: 2.84,
    2016: 1.98,
    2015: 2.31,
    2014: 1.60,
    2013: -0.08,
    2012: -0.70,
    2011: 1.89,
    2010: 2.23
}


# Loop over monthly_data to add the GDP growth rate
for index, row in monthly_data.iterrows():
    # Get year and country from each row
    year = row['Date'].year
    country = row['Country']
    
    if country == "Eurozone":  # European tickers
        monthly_data.at[index, 'GDP_Growth'] = european_union_gdp_growth[year]
    elif country == "Global":  # Global commodities
        monthly_data.at[index, 'GDP_Growth'] = global_gdp_growth[year]
    else:
        # Get the GDP growth rate from the dictionary by country/year
        gdp_growth = gdp_dict[country][str(year)]
        # Assign the GDP growth rate to the 'GDP_Growth' column in the monthly_data df
        monthly_data.at[index, 'GDP_Growth'] = gdp_growth



# In[8]:


monthly_data


# In[12]:


# Install TensorFlow using pip in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install tensorflow')


# In[15]:


import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split


# In[28]:


from sklearn.metrics import mean_squared_error


# In[95]:


monthly_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values (or you can choose to fill them with some value)
monthly_data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(monthly_data[['Close_change', 'Volume_change', 'Month_sin', 'Month_cos', 'GDP_Growth']])


def create_sequences(data, n_steps, close_change_idx, volume_change_idx):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :])

        y.append(data[i, [close_change_idx, volume_change_idx]])
    return np.array(X), np.array(y)

n_steps = 6   # Using the past 6 months to predict
Close_change_idx = 0
Volume_change_idx = 1
X, y = create_sequences(scaled_features, n_steps, Close_change_idx, Volume_change_idx)

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# LSTM (Long Short-Term memory)

model = Sequential() 

# LSTM layer with 50 neurons
# returns the full sequence to the next 'layer' as we are stacking layers
model.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))

# Dropout is used to prevent overfitting. In this case 20% of the input to the layer will be randomly excluded 
# from the updates during training
model.add(Dropout(0.2))

# Adding the final layer and dropout for regularization
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# This layer takes the final output of the last Layer and gives us a single predictive value
model.add(Dense(units = 2))


# Adam optimizer is a variant of stochastic gradient descent
# we are using the MSE as the loss function for regression
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)


# Evaluate the model
y_pred = model.predict(X_test)

dummy_features = np.zeros((y_pred.shape[0], scaled_features.shape[1] - y_pred.shape[1]))
y_pred_full = np.concatenate([y_pred, dummy_features], axis=1)

# Inverse transform using the full array
y_pred_rescaled = scaler.inverse_transform(y_pred_full)

# Extract only the columns that were predicted (e.g., first 2 columns for close_change and volume_change)
y_pred_rescaled = y_pred_rescaled[:, :y_pred.shape[1]]

# Calculate performance metrics, such as MSE
mse = mean_squared_error(y_test, y_pred_rescaled)
print(f"Mean Squared Error: {mse}")
        


# In[102]:


numeric_columns = ['Close_change', 'Volume_change', 'Month_sin', 'Month_cos', 'GDP_Growth']


scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(monthly_data[numeric_columns])

# Concatenating the scaled features back with the 'Ticker' column to be able to group the predictions
scaled_features_with_ticker = np.concatenate(
    (monthly_data[['Ticker']].values, scaled_features),
    axis=1
)

# Then convert it back to a DataFrame if you need to
scaled_features_with_ticker_df = pd.DataFrame(
    scaled_features_with_ticker, 
    columns=['Ticker'] + numeric_columns
)


# In[122]:


indices = monthly_data['Ticker'].to_list()
unique_indices = set(indices)
def get_last_sequences(X, n_steps):
    last_sequences = {}
    for index in unique_indices:
        index_data = X[X['Ticker'] == index].iloc[:,1:]  
        # Ensure we have enough data for the last sequence
        if len(index_data) >= n_steps:
            last_sequence = index_data[-n_steps:].values
            last_sequences[index] = last_sequence
    return last_sequences

def predict_next_step(model, last_sequences):
    predictions = {}
    for index, sequence in last_sequences.items():
        # Ensuring the sequence is a NumPy array to reshape
        sequence_np = np.array(sequence, dtype=np.float32)
        
        # Reshaping the sequence to match the input shape of the model
        sequence_reshaped = sequence_np.reshape((1, -1, sequence_np.shape[-1]))
        
        # Converting NumPy array to TensorFlow tensor
        sequence_tensor = tf.convert_to_tensor(sequence_reshaped, dtype=tf.float32)
        
        # Using the model to predict the next step
        predictions[index] = model.predict(sequence_tensor)
    return predictions


def generate_recommendations(predictions):
    recommendations = {}
    for index, prediction in predictions.items():
        close_change, volume_change = prediction[0]
        if close_change < 0 and volume_change > 0:
            rec = "Sell"
        elif close_change < 0 and volume_change < 0:
            rec = "Buy"
        elif close_change > 0 and volume_change > 0:
            rec = "Buy"
        elif close_change > 0 and volume_change < 0:
            rec = "Sell"
        # Add more conditions based on your logic
        else:
            rec = "Hold"
        recommendations[index] = rec
    return recommendations

n_steps = 8
# Assuming `X` is your feature matrix and `n_steps` is the number of time steps used for sequences
last_sequences = get_last_sequences(scaled_features_with_ticker_df, n_steps)

# Get predictions for the next step for each index
next_step_predictions = predict_next_step(model, last_sequences)

# Generate recommendations for each index
final_recommendations = generate_recommendations(next_step_predictions)

# Print or save the recommendations
for index, recommendation in final_recommendations.items():
    print(f"Index: {index}, Recommendation: {recommendation}")



# In[ ]:




