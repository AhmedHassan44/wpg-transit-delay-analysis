# 🚍 Winnipeg Transit Delay Prediction Using Machine Learning

This project analyzes Winnipeg Transit data to explore bus punctuality patterns and develop predictive models using machine learning. It combines data preprocessing, exploratory data analysis (EDA), visualization, and ML classification and regression models to derive actionable insights and predict transit performance.

---

## 📌 Project Objectives

- Analyze Winnipeg Transit on-time performance across routes, days, and weather conditions.
- Identify the most and least punctual routes and patterns in transit delays.
- Build classification and regression models to predict on-time status and delay duration.

---

## 🧠 Machine Learning Models & Results

| Model               | Task           | Performance               |
|--------------------|----------------|---------------------------|
| Logistic Regression | Classification | ✅ Accuracy: 99.12%        |
| Random Forest       | Classification | ✅ Accuracy: 99.31%        |
| XGBoost Classifier  | Classification | ✅ Accuracy: 99.17%        |
| Linear Regression   | Regression     | ✅ R² Score: 0.7343        |
| XGBoost Regressor   | Regression     | ✅ R² Score: 0.7416        |
|                     |                | ✅ MAE: 31 mins, RMSE: 52.45 |

---

## 📊 Exploratory Data Analysis (EDA)

### 🔝 Top 10 Most Punctual Routes
- **Consistently on-time** (e.g., Lineenwoods East/West, St. Mars).
- Action: Analyze features contributing to their efficiency.

### 🔻 10 Least Punctual Routes
- **High delays** (e.g., Logan–Berry, Southdale Express).
- Action: Prioritize improvement strategies.

### 📆 Weekday vs Weekend On-Time Performance
- Weekdays show more delays due to traffic.
- Action: Add peak-hour buffers to schedules.

### 📉 Monthly Trend (Oct 2024 – Mar 2025)
- Delays increase in colder months.
- Action: Implement winter-specific adjustments.

### 🕒 Hourly Performance by Day
- Delays spike during rush hours (7–9 AM, 4–6 PM).
- Action: Adjust schedules around peak traffic.

### ☁️ Weather Insights
- **Temperature**: Better punctuality in warm weather.
- **Wind Gusts**: Delays rise sharply above 25 km/h.
- **Snow**: Both late and early stops increase in snowfall.
- Action: Use weather-adjusted dynamic scheduling.

---

## 💻 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/WpgTransit-ML-Analysis.git
   cd WpgTransit-ML-Analysis

