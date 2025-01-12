"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.19.10
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

def train_isolation_forest(data: pd.DataFrame, features: list, model_params: dict) -> tuple:
    """
    Entrena un modelo Isolation Forest y genera predicciones.

    Args:
        data (pd.DataFrame): Datos procesados para entrenamiento.
        features (list): Lista de características a usar en el modelo.
        model_params (dict): Parámetros del modelo Isolation Forest.

    Returns:
        tuple: DataFrame con las predicciones y el modelo entrenado.
    """
    # Entrenar el modelo
    model = IsolationForest(**model_params)
    data['anomaly_score'] = model.fit_predict(data[features])

    return data, model


def save_results(data: pd.DataFrame, model, result_path: str, model_path: str):
    """
    Guarda los resultados y el modelo entrenado.

    Args:
        data (pd.DataFrame): Datos con predicciones.
        model: Modelo entrenado.
        result_path (str): Ruta para guardar resultados.
        model_path (str): Ruta para guardar el modelo.
    """
    # Guardar resultados
    data[['user_id', 'transaction_date', 'transaction_amount', 'anomaly_score']].to_csv(result_path, index=False)

    # Guardar el modelo
    joblib.dump(model, model_path)

    print(f"Results saved to {result_path}")
    print(f"Model saved to {model_path}")
