import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# better styling
sns.set(style="whitegrid")

# load cleaned data
df = pd.read_csv("cleaned_transit_data.csv")

# ensure day is datetime
df['day'] = pd.to_datetime(df['day'])

# total stops = early + late + on-time
df['total_stops'] = df['early_stops'] + df['late_stops'] + df['on-time_stops']

# on-time percentage
df['on_time_pct'] = df['on-time_stops'] / df['total_stops']

# remove rows with no stops (0 total)
df = df[df['total_stops'] > 0]

# extract features
df['month'] = df['day'].dt.to_period('M').astype(str)
df['day_of_week'] = df['day'].dt.day_name()
df['hour'] = df['time_period'].str.extract(r'(\d{1,2}):')[0].astype(float)

### 1. Summary: Total trips and average punctuality per route
route_summary = (
    df.groupby('route_name')
    .agg(total_trips=('total_stops', 'count'),
         avg_on_time_pct=('on_time_pct', 'mean'),
         avg_late=('late_stops', 'mean'),
         avg_early=('early_stops', 'mean'))
    .sort_values('total_trips', ascending=False)
)

print("\n🔍 Top 10 Routes by Total Records:\n")
print(route_summary.head(10).round(2))

### 2. Filter out routes with very few records (< 10 days)
filtered_routes = route_summary[route_summary['total_trips'] >= 10]

# 3. Top 10 Most Punctual Routes
top_punctual = filtered_routes.sort_values('avg_on_time_pct', ascending=False).head(10)
print("\n✅ Most Punctual Routes:\n", top_punctual[['avg_on_time_pct']])

plt.figure(figsize=(10, 5))
sns.barplot(data=top_punctual, x=top_punctual.index, y='avg_on_time_pct', palette='Greens_d')
plt.xticks(rotation=45)
plt.title("Top 10 Most Punctual Routes")
plt.ylabel("Average On-Time %")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 4. Top 10 Least Punctual Routes
least_punctual = filtered_routes.sort_values('avg_on_time_pct').head(10)
print("\n❌ Least Punctual Routes:\n", least_punctual[['avg_on_time_pct']])

plt.figure(figsize=(10, 5))
sns.barplot(data=least_punctual, x=least_punctual.index, y='avg_on_time_pct', palette='Reds_d')
plt.xticks(rotation=45)
plt.title("Top 10 Least Punctual Routes")
plt.ylabel("Average On-Time %")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

### 5. Weekday vs Weekend On-Time %
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='day_type', y='on_time_pct', palette='Set2')
plt.title("On-Time %: Weekday vs Weekend")
plt.ylabel("On-Time Percentage")
plt.tight_layout()
plt.show()

### 6. Monthly Trend of On-Time %
monthly_avg = df.groupby('month')['on_time_pct'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_avg, x='month', y='on_time_pct', marker='o', color='blue')
plt.xticks(rotation=45)
plt.title("Monthly On-Time Performance Trend")
plt.ylabel("Average On-Time %")
plt.xlabel("Month")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

### 7. Heatmap: On-Time by Day of Week and Hour
pivot = df.pivot_table(index='day_of_week', columns='hour', values='on-time_stops', aggfunc='mean')
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot = pivot.reindex(days_order)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.3)
plt.title("Average On-Time Stops by Hour and Day")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.show()

### 8. Distribution of Stop Counts
stop_cols = ['early_stops', 'late_stops', 'on-time_stops']
df[stop_cols].hist(bins=30, figsize=(10, 5), color='skyblue')
plt.suptitle("Distribution of Stop Types")
plt.tight_layout()
plt.show()

### 9. On-Time % for Most Frequent Routes (Top 10)
top_routes = df['route_name'].value_counts().head(10).index
df_top_routes = df[df['route_name'].isin(top_routes)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='route_name', y='on_time_pct', data=df_top_routes, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("On-Time % Distribution of Top 10 Frequent Routes")
plt.ylabel("On-Time Percentage")
plt.tight_layout()
plt.show()
