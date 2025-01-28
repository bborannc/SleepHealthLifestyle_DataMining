from sklearn.preprocessing import StandardScaler
import pandas as pd


file_path = 'encoded_sleep_health_lifestyle_dataset.csv'
data_encoded = pd.read_csv(file_path)


scaler = StandardScaler()


columns_to_scale = data_encoded.columns.drop('Person ID')
data_encoded[columns_to_scale] = scaler.fit_transform(data_encoded[columns_to_scale])


data_encoded.to_csv('standardized_sleep_health_lifestyle_dataset.csv', index=False)

print("Standardized dataset saved as 'standardized_sleep_health_lifestyle_dataset.csv'")
