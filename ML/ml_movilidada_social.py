import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("data_ml_ETL_sd.csv");

X = data[['Sexo', 'Estado civil', 'Edad', 'Educacion', 'Ingreso anual', 
          'Ocupacion', 'Tamano de asentamiento', 'Ingreso mensual']]
y = data['Nivel socioeconomico']  # Variable a predecir

# Dividir datos (80% entrenamiento, 20% validación)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(train_X, train_y)

# Predicción y evaluación
prediccion = modelo.predict(val_X)
print("Error absoluto medio (MAE):", mean_absolute_error(val_y, prediccion))

