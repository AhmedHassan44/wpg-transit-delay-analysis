from data_processor import TransitDataProcessor

if __name__ == "__main__":
    processor = TransitDataProcessor("cleaned_transit_data.csv")
    processor.load_data()
    processor.create_on_time_status()

    df = processor.get_processed_data()
    print(df[['early_stops', 'late_stops', 'on-time_stops', 'on_time_status']].head())