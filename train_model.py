Adiciona script train_model.py para treinar modelo
import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow

# Inicia um experimento no MLflow
mlflow.start_run()

# Carrega os dados
data = pd.read_csv('inputs/vendas_sorvete.txt')

# Separa X e y
X = data[['Temperatura']]
y = data['Vendas']

# Treina modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Registra parâmetros e métricas no MLflow
mlflow.log_param("Modelo", "LinearRegression")
mlflow.log_metric("Score", model.score(X, y))

# Salva modelo
mlflow.sklearn.log_model(model, "modelo_vendas")

print("Modelo treinado e registrado com sucesso!")
