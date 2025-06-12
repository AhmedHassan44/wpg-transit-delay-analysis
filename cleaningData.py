import pandas as pd

# ----------------------------
# 1. Load the raw transit dataset
# ----------------------------
df = pd.read_csv("/Users/ahmedhasan/desktop/transitData.csv")
print("Initial data loaded.")
print(df.head())

# ----------------------------
# 2. Standardize column names (remove spaces, lowercase)
# ----------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\nStandardized column names:")
print(df.columns)

# ----------------------------
# 3. Convert 'day' column to datetime format
# ----------------------------
df['day'] = pd.to_datetime(df['day'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
print("\nConverted 'day' column to datetime.")

# ----------------------------
# 4. Filter data from October 2024 to March 2025
# ----------------------------
start_date = "2024-10-01"
end_date = "2025-03-31"
df_filtered = df[(df['day'] >= start_date) & (df['day'] <= end_date)]
print(f"\nFiltered data from {start_date} to {end_date}.")
print(df_filtered.shape)

# ----------------------------
# 5. Check for nulls and duplicates
# ----------------------------
print("\nNull values per column:")
print(df_filtered.isnull().sum())

print("\nNumber of duplicate rows:")
print(df_filtered.duplicated().sum())

# Drop rows with missing critical values
df_filtered = df_filtered.dropna(subset=['day', 'route_number', 'route_destination'])

# ----------------------------
# 6. Ensure numerical columns are properly typed
# ----------------------------

numeric_cols = ['early_stops', 'late_stops', 'on-time_stops']
df_filtered[numeric_cols] = df_filtered[numeric_cols].apply(pd.to_numeric, errors='coerce')
print("\nConverted stop counts to numeric.")

# ----------------------------
# 7. Save cleaned data to new CSV
# ----------------------------
df_filtered.to_csv("cleaned_transit_data.csv", index=False)
print("\n✅ Cleaned data saved to 'cleaned_transit_data.csv'")
