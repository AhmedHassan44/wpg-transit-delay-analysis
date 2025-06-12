from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n📊 Linear Regression Results")
    print(f"✅ MAE : {mean_absolute_error(y_test, preds):.2f}")
    print(f"✅ RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    print(f"✅ R2  : {r2_score(y_test, preds):.4f}")