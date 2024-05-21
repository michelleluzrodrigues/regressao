import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Carregar dataset de prioridade de resgate
rescue_prior = pd.read_csv('datasets/data_300v_90x90/rescue_prior.txt', sep=',', header=None)

rescue_prior.columns = ['x1', 'x2', 'x3','x4', 'p']

# Separar features e target
X_priority = rescue_prior[['x1', 'x2', 'x3', 'x4']]
y_priority = rescue_prior['p']

# Dividir dataset em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_priority, y_priority, test_size=0.2, random_state=42)

# Treinar modelo de regressão (MLP)
mlp_priority = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
mlp_priority.fit(X_train, y_train)

# Avaliar modelo
y_pred_train = mlp_priority.predict(X_train)
y_pred_val = mlp_priority.predict(X_val)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

print(f"RMSE Treino: {rmse_train}")
print(f"RMSE Validação: {rmse_val}")

# Salvar modelo treinado
import joblib
joblib.dump(mlp_priority, 'mlp_priority.pkl')
