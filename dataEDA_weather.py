import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv('merged_transit_weather.csv')
df['datetime'] = pd.to_datetime(df['datetime_x'], errors='coerce')
df = df.dropna(subset=['late_stops', 'on-time_stops'])

# Derived columns
df['total_stops'] = df['early_stops'] + df['late_stops'] + df['on-time_stops']
df['on_time_pct'] = df['on-time_stops'] / df['total_stops']

# Wind gust level classification (exclude 'Extreme' later)
df['windgust_level'] = pd.cut(
    df['windgust'],
    bins=[-1, 20, 40, 60, 200],
    labels=['Low', 'Medium', 'High', 'Extreme']
)

# Snow level classification (will filter to only 'None' and 'Light')
df['snow_level'] = pd.cut(
    df['snow'],
    bins=[-0.1, 0.1, 2, 100],
    labels=['None', 'Snow', 'Heavy']
)

# --- Summary Statistics ---
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nTransit Stats:\n", df[['early_stops', 'late_stops', 'on-time_stops', 'on_time_pct']].describe().round(2))
print("\nWeather Stats:\n", df[['temp', 'humidity', 'precip', 'snow', 'windgust', 'visibility']].describe().round(2))

# --- Correlation Heatmap ---
weather_cols = ['temp', 'tempmax', 'tempmin', 'dew', 'humidity', 'precip', 'snow', 'windgust', 'windspeed', 'visibility']
corr_matrix = df[weather_cols + ['early_stops', 'late_stops', 'on-time_stops', 'on_time_pct']].corr()

plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation: Transit Performance and Weather Features")
plt.tight_layout()
plt.show()

# --- Scatter: On-Time Stops vs Temperature ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='temp', y='on-time_stops', hue='day_type', alpha=0.6)
plt.title("On-Time Stops vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("On-Time Stops")
plt.tight_layout()
plt.show()

# --- Scatter: Late Stops vs Wind Gust ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='windgust', y='late_stops', color='darkred', alpha=0.6)
plt.title("Late Stops vs Wind Gust")
plt.xlabel("Wind Gust (km/h)")
plt.ylabel("Late Stops")
plt.tight_layout()
plt.show()

# --- 5. Barplot: Avg Late Stops by Wind Gust Level (Exclude 'Extreme') ---
wind_summary = (
    df[df['windgust_level'].isin(['Low', 'Medium', 'High'])]
    .groupby('windgust_level')['late_stops']
    .mean()
    .round(1)
)

plt.figure(figsize=(8, 6))
sns.barplot(x=wind_summary.index, y=wind_summary.values, palette='Reds')
plt.title("Average Late Stops by Wind Gust Level")
plt.ylabel("Avg Late Stops")
plt.xlabel("Wind Gust Level")
plt.tight_layout()
plt.show()

# --- 6. Barplot: Avg Early & Late Stops by Snow Level (Only 'None' and 'Snow') ---
snow_filtered = df[df['snow_level'].isin(['None', 'Snow'])]
snow_summary = (
    snow_filtered
    .groupby('snow_level')[['early_stops', 'late_stops']]
    .mean()
    .round(1)
    .reset_index()
)

snow_melt = snow_summary.melt(id_vars='snow_level', var_name='Stop Type', value_name='Avg Stops')

plt.figure(figsize=(10, 6))
sns.barplot(data=snow_melt, x='snow_level', y='Avg Stops', hue='Stop Type', palette='coolwarm')
plt.title("Average Stops by Snow Level")
plt.xlabel("Snow Level")
plt.ylabel("Average Number of Stops")
plt.tight_layout()
plt.show()

# --- Final: Print correlation of weather with late stops ---
correlations = df[weather_cols + ['late_stops']].corr()['late_stops'].sort_values(ascending=False)
print("\nCorrelation of Weather with Late Stops:\n", correlations.round(2))
