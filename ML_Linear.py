from data_processor import TransitDataProcessor
from model_linear import train_linear_regression

if __name__ == "__main__":
    processor = TransitDataProcessor("cleaned_transit_data.csv")
    processor.load_data()
    processor.create_on_time_status()
    processor.preprocess_all()

    print("\n🧠 Running Linear Regression")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = processor.get_train_test_regression_data()
    train_linear_regression(X_train_reg, X_test_reg, y_train_reg, y_test_reg)