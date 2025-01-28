from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd


file_path = 'standardized_sleep_health_lifestyle_dataset.csv'
data = pd.read_csv(file_path)


data['Sleep_Disorder_Encoded'] = data['Sleep_Disorder_Encoded'].apply(lambda x: 1 if x > 0 else 0)


X = data.drop(columns=['Sleep_Disorder_Encoded'])  # Özellikler
y = data['Sleep_Disorder_Encoded']  # Hedef değişken


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print(f"Naive Bayes Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
