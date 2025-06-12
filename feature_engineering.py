import pandas as pd

# load the cleaned transit dataset and parse 'day' as datetime
df = pd.read_csv("cleaned_transit_data.csv", parse_dates=['day'])

# total number of stops = early + late + on-time
df['total_stops'] = df['early_stops'] + df['late_stops'] + df['on-time_stops']

# calculate on-time stop percentage
df['on_time_pct'] = df['on-time_stops'] / df['total_stops']

# binary target: 1 = highly punctual (≥ 80% on-time), 0 = not
df['high_punctuality'] = (df['on_time_pct'] >= 0.8).astype(int)

# create a copy of the dataframe for encoding
df_encoded = df.copy()

# encode 'day_type': Weekday = 0, Weekend = 1
df_encoded['day_type'] = df_encoded['day_type'].map({'Weekday': 0, 'Weekend': 1})

# extract day of week name from 'day' column
df_encoded['day_of_week'] = df_encoded['day'].dt.day_name()

# one-hot encode 'route_name' and 'day_of_week'; drop first to avoid multicollinearity
df_encoded = pd.get_dummies(df_encoded, columns=['route_name', 'day_of_week'], drop_first=True)
