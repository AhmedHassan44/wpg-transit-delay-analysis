import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

class XGBoostTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.classifier_model = None
        self.regressor_model = None
        self.X_train_class = self.X_test_class = self.y_train_class = self.y_test_class = None
        self.X_train_reg = self.X_test_reg = self.y_train_reg = self.y_test_reg = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            print("✅ Data loaded.")
        except FileNotFoundError:
            print("❌ File not found.")

    def create_on_time_status(self):
        if self.df is not None:
            self.df['on_time_status'] = self.df['on-time_stops'] > (self.df['early_stops'] + self.df['late_stops'])
            self.df['on_time_status'] = self.df['on_time_status'].astype(int)
            print("✅ 'on_time_status' column created.")

    def preprocess_data(self):
        if self.df is None:
            print("❌ Load data first.")
            return

        drop_cols = ['key', 'datetime_x', 'datetime_y', 'date', 'day', 'tempmax', 'tempmin', 'dew', 'windgust']
        self.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Encode object columns
        obj_cols = self.df.select_dtypes(include='object').columns
        for col in obj_cols:
            self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))

        self.df.fillna(0, inplace=True)

        # Classification prep
        if 'on_time_status' in self.df.columns:
            X_class = self.df.drop(columns=['on_time_status'])
            y_class = self.df['on_time_status']
            self.X_train_class, self.X_test_class, self.y_train_class, self.y_test_class = train_test_split(
                X_class, y_class, test_size=0.2, random_state=42
            )

        # Regression prep
        if 'late_stops' in self.df.columns:
            X_reg = self.df.drop(columns=['late_stops'])
            y_reg = self.df['late_stops']
            self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )

        print("✅ Data preprocessed.")

    def train_xgb_classifier(self):
        self.classifier_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.classifier_model.fit(self.X_train_class, self.y_train_class)
        preds = self.classifier_model.predict(self.X_test_class)

        acc = accuracy_score(self.y_test_class, preds)
        print("\n📊 XGBoost Classifier Results")
        print(f"✅ Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test_class, preds))

    def train_xgb_regressor(self):
        self.regressor_model = XGBRegressor(random_state=42)
        self.regressor_model.fit(self.X_train_reg, self.y_train_reg)
        preds = self.regressor_model.predict(self.X_test_reg)

        mae = mean_absolute_error(self.y_test_reg, preds)
        rmse = np.sqrt(mean_squared_error(self.y_test_reg, preds))
        r2 = r2_score(self.y_test_reg, preds)

        print("\n📊 XGBoost Regressor Results")
        print(f"✅ MAE : {mae:.2f}")
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ R²  : {r2:.4f}")


if __name__ == "__main__":
    trainer = XGBoostTrainer("cleaned_transit_data.csv")
    trainer.load_data()
    trainer.create_on_time_status()
    trainer.preprocess_data()
    trainer.train_xgb_classifier()
    trainer.train_xgb_regressor()
