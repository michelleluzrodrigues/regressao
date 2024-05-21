from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Carregar dataset
data_4000v = pd.read_csv('datasets/data_4000v/env_vital_signals.txt', sep=',', header=None)

data_4000v = data_4000v.drop(data_4000v.columns[[0,1,2]], axis=1)

data_4000v.columns = ['qPA', 'pulso', 'frequência respiratória','gravidade', 'gravidade id']

# Separar features e target
X_vitals = data_4000v[['qPA', 'pulso', 'frequência respiratória']]
y_gravity = data_4000v['gravidade']

# Dividir dataset em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_vitals, y_gravity, test_size=0.2, random_state=42)



# Treinar modelo de regressão (Decision Tree)
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)

# Avaliar modelo
y_pred_train = tree_regressor.predict(X_train)
y_pred_val = tree_regressor.predict(X_val)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

print(f"RMSE Treino: {rmse_train}")
print(f"RMSE Validação: {rmse_val}")

# Salvar modelo treinado
joblib.dump(tree_regressor, 'tree_regressor.pkl')
