import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go

# Load the dataset
df = pd.read_csv("global-data-on-sustainable-energy (1).csv")
print(df.head(5))

# Check for missing values
print(df.isnull().sum())

# Convert 'Year' to integer type, handling non-integer values if necessary
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

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

# Function to forecast energy consumption for a given country
def forecast_energy_consumption(country, steps=30):
    # Filter data for the selected country
    country_data = final_df[final_df['Entity'] == country].copy()
    
    # Check if the country data has enough points for ARIMA
    if len(country_data) < 2:  # Change this threshold based on your needs
        print(f"Not enough data to forecast for {country}")
        return None, None, None

    # Set the index to Year for time series analysis and convert to datetime
    country_data['Year'] = pd.to_datetime(country_data['Year'], format='%Y')
    country_data.set_index('Year', inplace=True)

    # Train ARIMA model on the country's energy consumption data
    model = ARIMA(country_data['Primary energy consumption per capita (kWh/person)'], order=(5, 1, 0))
    
    try:
        model_fit = model.fit()
        # Forecast for the next 'steps' years
        forecast = model_fit.forecast(steps=steps)
        
        # Create future year range as a list of years
        future_years = list(range(country_data.index.year.max() + 1, country_data.index.year.max() + 1 + steps))
        
        return country_data, forecast, future_years
    except Exception as e:
        print(f"Could not generate forecast for {country}: {e}")
        return None, None, None

# Example of forecasting for all countries
countries = final_df['Entity'].unique()
forecast_results = {}

for country in countries:
    historical_data, forecast, future_years = forecast_energy_consumption(country)
    if forecast is not None:  # Ensure there's a valid forecast
        forecast_results[country] = {
            'historical_years': historical_data.index.year.tolist(),
            'historical_consumption': historical_data['Primary energy consumption per capita (kWh/person)'].tolist(),
            'future_years': future_years,
            'forecast': forecast.tolist()
        }

# Create a Plotly figure with interactive dropdown for countries
fig = go.Figure()

# Add the first country's data (initially selected)
initial_country = countries[0]
fig.add_trace(go.Scatter(x=forecast_results[initial_country]['historical_years'],
                         y=forecast_results[initial_country]['historical_consumption'],
                         mode='lines', name=f'{initial_country} Historical'))

fig.add_trace(go.Scatter(x=forecast_results[initial_country]['future_years'],
                         y=forecast_results[initial_country]['forecast'],
                         mode='lines', name=f'{initial_country} Forecast', line=dict(dash='dash')))

# Add layout and dropdown
fig.update_layout(
    title='Energy Consumption Forecast per Country',
    xaxis_title='Year',
    yaxis_title='Energy Consumption',
    updatemenus=[{
        'buttons': [
            {
                'args': [{'x': [forecast_results[country]['historical_years'], forecast_results[country]['future_years']],
                          'y': [forecast_results[country]['historical_consumption'], forecast_results[country]['forecast']]}],
                'label': country,
                'method': 'update'
            } for country in countries
        ],
        'direction': 'down',
        'showactive': True
    }]
)

# Show the figure
fig.show()