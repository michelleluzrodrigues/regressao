import pandas as pd
import joblib

def validar_e_gerar_saida(input_file, model_file='tree_regressor.pkl', output_file='pred.txt'):
    # Carregar o modelo treinado
    model = joblib.load(model_file)
    
    # Carregar o arquivo de dados para predição
    data = pd.read_csv(input_file, sep=',', header=None)
    
    data.columns = ['qPA', 'pulso', 'frequência respiratória', 'gravidade']
    X_new = data[['qPA', 'pulso', 'frequência respiratória']]
    
    # Fazer a predição
    predictions = model.predict(X_new)
    
    # Adicionar as previsões como a última coluna
    data['predictions'] = predictions
    
    # Salvar o arquivo resultante
    data.to_csv(output_file, index=False, header=False)
    print(f"Previsões salvas em {output_file}")

# Exemplo de chamada da função
# validar_e_gerar_saida('datasets/data_4000v/env_vital_signals.txt')

validar_e_gerar_saida('datasets/data_400v_90x90/rescue_prior_blind.txt')