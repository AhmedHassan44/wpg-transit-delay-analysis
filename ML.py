from data_processor import TransitDataProcessor
from model_rf import train_rf_classifier, train_rf_regressor

if __name__ == "__main__":
    processor = TransitDataProcessor("cleaned_transit_data.csv")
    processor.load_data()
    processor.create_on_time_status()
    processor.preprocess_all()

    print("\n🧠 Running Random Forest Classifier")
    X_train, X_test, y_train, y_test = processor.get_train_test_data()
    train_rf_classifier(X_train, X_test, y_train, y_test)

    print("\n🧠 Running Random Forest Regressor")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = processor.get_train_test_regression_data()
    train_rf_regressor(X_train_reg, X_test_reg, y_train_reg, y_test_reg) 