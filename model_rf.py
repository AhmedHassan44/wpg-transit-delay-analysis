from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import numpy as np

def train_rf_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n📊 Random Forest Classification Results")
    print("✅ Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

def train_rf_regressor(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n📊 Random Forest Regression Results")
    print(f"✅ MAE : {mean_absolute_error(y_test, preds):.2f}")
    print(f"✅ RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")