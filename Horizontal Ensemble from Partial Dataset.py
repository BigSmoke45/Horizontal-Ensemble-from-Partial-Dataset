# Імпорт необхідних бібліотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Завантаження даних
data = pd.read_csv("A:/loan_data.csv")

# Попередня обробка даних
# Заповнення пропущених значень
imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Кодування категоріальних ознак
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Масштабування числових ознак
scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(exclude=['object']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Розподіл даних на тренувальний та тестовий набори
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання нейронних мереж і збереження найкращих моделей
best_models = []
for i in range(5):  # Приклад: навчання п’яти нейронних мереж
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(f'best_model_{i}.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
    best_models.append(load_model(f'best_model_{i}.h5'))

# Збір передбачень від усіх моделей
predictions = []
for model in best_models:
    predictions.append((model.predict(X_test) > 0.5).astype(int))

# Прийняття рішення на основі голосування
ensemble_predictions = (sum(predictions) >= (len(predictions) / 2)).astype(int)

# Оцінка точності ансамблю
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print("Accuracy of Ensemble Model:", ensemble_accuracy)
