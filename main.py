import pandas as pd
import xgboost as xgb

# Load your dataset
df = pd.read_csv('global-data-on-sustainable-energy (1).csv')

# Check the data structure
print("DataFrame Head:")
print(df.head())

# Check the columns
print("Columns in the DataFrame:")
print(df.columns)

# Check unique country names
print("Unique Countries in the DataFrame:")
print(df['Entity'].unique())

# Check for missing values in the relevant column
print("Missing values in 'Primary energy consumption per capita (kWh/person)':")
print(df['Primary energy consumption per capita (kWh/person)'].isnull().sum())

# Preprocess the data
df['Year'] = pd.to_datetime(df['Year'], format='%Y').dt.year  # Extract year as an integer
df['Consumption'] = df['Primary energy consumption per capita (kWh/person)'].astype(float)

# Check data for the United States
us_data = df[df['Entity'] == 'United States']
print("Data for United States before feature creation:")
print(us_data)

# Function to create features for forecasting
def create_features(data):
    data['Lag1'] = data['Consumption'].shift(1)
    data['Lag2'] = data['Consumption'].shift(2)
    data['Lag3'] = data['Consumption'].shift(3)
    data['RollingMean'] = data['Consumption'].rolling(window=3).mean()

    print("Data after feature creation:")
    print(data)

    # Remove rows with NaN values
    data.dropna(inplace=True)

    # Check if we have enough rows after dropping NaN values
    if data.shape[0] < 4:  # Less than 4 rows after dropping means insufficient data
        print("Not enough data after feature creation.")
        return data.iloc[0:0]  # Return empty DataFrame

    return data

# Function to generate forecasts using XGBoost
def generate_xgb_forecast(country, steps=30):
    try:
        country_data = df[df['Entity'] == country].copy()
        
        # Check if the filtered data is empty
        if country_data.empty:
            print(f"No data available for {country}.")
            return None, None
        
        print(f"Data for {country}:")
        print(country_data)

        country_data = create_features(country_data)

        # Check if the data after creating features is empty
        if country_data.empty:
            print(f"No valid data available for feature creation for {country}.")
            return None, None
        
        # Prepare data for XGBoost
        X = country_data[['Year', 'Lag1', 'Lag2', 'Lag3', 'RollingMean']]
        y = country_data['Consumption']

        # Check if X or y is empty
        if X.empty or y.empty:
            print(f"Empty feature set or target for {country}.")
            return None, None

        # Print sizes of X and y
        print("Feature set size:", X.shape)
        print("Target size:", y.shape)

        # Split into train and test sets
        if len(y) > steps:
            train_X = X[:-steps]
            train_y = y[:-steps]
            test_X = X[-steps:]
        else:
            print(f"Not enough data for {country} to split into train and test sets.")
            return None, None

        # Train the model
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(train_X, train_y)

        # Forecasting
        forecast = []
        for i in range(steps):
            input_data = test_X.iloc[i].values.reshape(1, -1)
            forecast_value = model.predict(input_data)
            forecast.append(forecast_value[0])

        last_year = country_data['Year'].iloc[-1]
        forecast_years = [last_year + i + 1 for i in range(steps)]

        return forecast, forecast_years
    except Exception as e:
        print(f"Could not generate forecast for {country}: {e}")
        return None, None

# Test for a specific country
forecast, years = generate_xgb_forecast('United States')

if forecast is not None and years is not None:
    print(f"Forecast for United States for years {years}:")
    print(forecast)
else:
    print("Forecast generation failed.")