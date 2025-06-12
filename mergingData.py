import pandas as pd

# Load transit and weather data
transit_df = pd.read_csv('cleaned_transit_data.csv')  # adjust filename if needed
weather_df = pd.read_csv('/Users/ahmedhasan/desktop/weather_data.csv')

# Check column names
print("Transit Columns:", transit_df.columns)
print("Weather Columns:", weather_df.columns)

# Create datetime column in transit_df by combining 'day' and 'time_period'
transit_df['datetime'] = pd.to_datetime(
    transit_df['day'].astype(str) + ' ' + transit_df['time_period'].astype(str),
    errors='coerce',
    utc=True
)

# Convert weather datetime to pandas datetime (UTC)
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], errors='coerce', utc=True)

# Drop rows with invalid datetime values
transit_df.dropna(subset=['datetime'], inplace=True)
weather_df.dropna(subset=['datetime'], inplace=True)

# Extract date only from datetime for merging
transit_df['date'] = transit_df['datetime'].dt.date
weather_df['date'] = weather_df['datetime'].dt.date

# Merge on 'date' column
merged_df = pd.merge(transit_df, weather_df, on='date', how='inner')

# Save merged data
merged_df.to_csv('merged_transit_weather.csv', index=False)

# Debug output
print("Merged data saved as 'merged_transit_weather.csv'")
print("Sample transit datetimes:\n", transit_df['datetime'].head(10))
print("Sample weather datetimes:\n", weather_df['datetime'].head(10))
print("Merged Data Sample:\n", merged_df.head())
