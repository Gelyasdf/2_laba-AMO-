import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

DATA_PATH = 'data/processed/data_prepared.csv'
MODEL_PATH = 'models/model.pkl'

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Чтение данных
data = pd.read_csv(DATA_PATH)

# Разделение данных на признаки и целевую переменную
X = data.drop('quality', axis=1)
y = data['quality']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, MODEL_PATH)
print('Model trained and saved successfully.')
