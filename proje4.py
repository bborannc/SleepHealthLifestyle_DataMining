from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd


file_path = 'standardized_sleep_health_lifestyle_dataset.csv'
data = pd.read_csv(file_path)


data['Sleep_Disorder_Encoded'] = data['Sleep_Disorder_Encoded'].apply(lambda x: 1 if x > 0 else 0)


X = data.drop(columns=['Sleep_Disorder_Encoded'])  # Özellikler
y = data['Sleep_Disorder_Encoded']  # Hedef değişken


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


hidden_layer_configurations = [
    (32,),  # 1 gizli katman, 32 nöron
    (32, 32),  # 2 gizli katman, her biri 32 nöron
    (32, 32, 32)  # 3 gizli katman, her biri 32 nöron
]


results = {}


for hidden_layers in hidden_layer_configurations:
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gizli Katman Yapılandırması: {hidden_layers}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
