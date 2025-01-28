import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


file_path = 'sleep_health_lifestyle_dataset.csv'
data = pd.read_csv(file_path)


label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')


label_columns = ['Gender', 'BMI Category', 'Sleep Disorder']
for col in label_columns:
    data[f'{col}_Encoded'] = label_encoder.fit_transform(data[col])


onehot_columns = ['Occupation']
for col in onehot_columns:
    encoded = pd.DataFrame(
        onehot_encoder.fit_transform(data[[col]]),
        columns=onehot_encoder.get_feature_names_out([col])
    )
    data = pd.concat([data, encoded], axis=1)


data[['BP_Systolic', 'BP_Diastolic']] = data['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(int)


columns_to_drop = label_columns + onehot_columns + ['Blood Pressure (systolic/diastolic)']
data_encoded = data.drop(columns=columns_to_drop, axis=1)


print(data_encoded.head())


data_encoded.to_csv('encoded_sleep_health_lifestyle_dataset.csv', index=False)