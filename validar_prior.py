import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def validar_e_gerar_saida(input_file, model_file='mlp_priority.pkl'):
    # Carregar o modelo treinado
    model = joblib.load(model_file)
    
    # Carregar o arquivo de dados para predição
    data = pd.read_csv(input_file, sep=',', header=None)
    
    data.columns = ['x1', 'x2', 'x3','x4', 'p']
    X_new = data[['x1', 'x2', 'x3','x4']]
    y_true = data['p']
    
    # Fazer a predição
    predictions = model.predict(X_new)
    
    # Adicionar as previsões como a última coluna
    data['predictions'] = predictions
    
    rmse = mean_squared_error(y_true, predictions, squared=False)
    
    print(f"RMSE: {rmse}")



validar_e_gerar_saida('datasets/data_400v_90x90/rescue_prior_target.txt')