import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TransitDataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_reg = self.X_test_reg = self.y_train_reg = self.y_test_reg = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        print("✅ Data loaded successfully.")

    def create_on_time_status(self):
        self.df['on_time_status'] = self.df['on-time_stops'] > (
            self.df['early_stops'] + self.df['late_stops']
        )
        self.df['on_time_status'] = self.df['on_time_status'].astype(int)
        print("✅ 'on_time_status' column created.")

    def preprocess_all(self):
        drop_cols = ['key', 'datetime_x', 'datetime_y', 'date', 'day', 'tempmax', 'tempmin', 'dew', 'windgust']
        self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Encode categorical columns
        obj_cols = self.df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))

        self.df.fillna(0, inplace=True)

        # Classification data
        X = self.df.drop(columns=['on_time_status'])
        y = self.df['on_time_status']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Regression data
        X_reg = self.df.drop(columns=['late_stops'])
        y_reg = self.df['late_stops']
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_train_test_regression_data(self):
        return self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg

    def get_processed_data(self):
        return self.df
