import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("global-data-on-sustainable-energy (1).csv")
print(df.head(5))

print(df.isnull().sum())

# Select numeric columns for scaling
numeric_columns = ['Year', 'Access to electricity (% of population)', 
                   'Electricity from fossil fuels (TWh)', 
                   'Electricity from nuclear (TWh)', 
                   'Electricity from renewables (TWh)', 
                   'Renewable energy share in the total final energy consumption (%)', 
                   'Primary energy consumption per capita (kWh/person)']

# Select non-numeric columns to keep
non_numeric_columns = ['Entity']

# Extract numeric and non-numeric columns separately
numeric_features = df[numeric_columns]
non_numeric_features = df[non_numeric_columns]

# Initialize the scaler
scaler = MinMaxScaler()

# Scale the numeric features
scaled_numeric_features = scaler.fit_transform(numeric_features)

# Convert back to DataFrame for numeric columns
scaled_numeric_df = pd.DataFrame(scaled_numeric_features, columns=numeric_columns)

# Concatenate the non-numeric columns with scaled numeric columns
final_df = pd.concat([non_numeric_features.reset_index(drop=True), scaled_numeric_df.reset_index(drop=True)], axis=1)

print(final_df.head())

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['Primary energy consumption per capita (kWh/person)'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)  # Forecast for next 30 years
print(forecast)