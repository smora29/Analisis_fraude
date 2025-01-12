"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.10
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los datos eliminando valores nulos y transformando las columnas necesarias.
    
    Args:
        raw_data (pd.DataFrame): Datos originales sin procesar.

    Returns:
        pd.DataFrame: Datos limpios.
    """
    # Eliminar filas con valores nulos en columnas críticas
    clean_data = raw_data.dropna(subset=["transaction_amount", "transaction_date", "user_id", "account_number"])
    
    # Convertir transaction_date a formato datetime
    clean_data["transaction_date"] = pd.to_datetime(clean_data["transaction_date"])
    
    return clean_data


def feature_engineering(clean_data: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables derivadas para el modelo.

    Args:
        clean_data (pd.DataFrame): Datos limpios.

    Returns:
        pd.DataFrame: Datos con variables derivadas.
    """
    # Crear copia para evitar SettingWithCopyWarning
    clean_data = clean_data.copy()

    # Agregar variables temporales
    clean_data.loc[:, 'year'] = clean_data['transaction_date'].dt.year
    clean_data.loc[:, 'month'] = clean_data['transaction_date'].dt.month
    clean_data.loc[:, 'day'] = clean_data['transaction_date'].dt.day

    # Calcular diferencia de tiempo entre transacciones
    clean_data.loc[:, 'time_diff'] = clean_data.groupby('account_number')['transaction_date'].diff().dt.total_seconds()

    # Eliminar filas con NaN en 'time_diff'
    clean_data = clean_data.dropna(subset=['time_diff'])

    # Crear columna de día de la semana
    clean_data.loc[:, 'day_of_week'] = clean_data['transaction_date'].dt.dayofweek

    # Indicar si es fin de semana
    clean_data.loc[:, 'is_weekend'] = clean_data['day_of_week'].isin([5, 6]).astype(int)

    # Crear ventanas temporales de 24 horas y calcular métricas agregadas
    clean_data.loc[:, 'window_24h'] = clean_data['transaction_date'].dt.floor('24H')
    aggregated = clean_data.groupby(['user_id', 'window_24h']).agg(
        total_amount=('transaction_amount', 'sum'),
        transaction_count=('transaction_amount', 'count'),
        average_amount=('transaction_amount', 'mean'),
        std_amount=('transaction_amount', 'std')
    ).reset_index()

    # Unir las métricas agregadas al dataset original
    features = pd.merge(clean_data, aggregated, on=['user_id', 'window_24h'], how='left')

    # Convertir columnas relevantes a tipo float
    for col in ['transaction_amount', 'time_diff']:
        features[col] = pd.to_numeric(features[col], errors='coerce')

    # Eliminar filas con NaN después de la conversión
    features = features.dropna(subset=['transaction_amount', 'time_diff'])

    # Normalizar características numéricas
    features.loc[:, 'transaction_amount'] = (features['transaction_amount'] - features['transaction_amount'].mean()) / features['transaction_amount'].std()
    features.loc[:, 'time_diff'] = (features['time_diff'] - features['time_diff'].mean()) / features['time_diff'].std()

    return features

# def scaling(features: pd.DataFrame) -> pd.DataFrame:
#     """
#     Escala las variables numéricas para el modelo DBSCAN.
    
#     Args:
#         features (pd.DataFrame): Datos con variables derivadas.

#     Returns:
#         pd.DataFrame: Datos escalados.
#     """
#     # Seleccionar columnas numéricas relevantes
#     columns_to_scale = ["transaction_amount", "total_amount", "average_amount", "transaction_count"]
    
#     scaler = StandardScaler()
#     scaled_values = scaler.fit_transform(features[columns_to_scale])
    
#     # Reemplazar las columnas originales con sus versiones escaladas
#     for i, col in enumerate(columns_to_scale):
#         features[col] = scaled_values[:, i]
    
#     return features

def normalize_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza características numéricas seleccionadas.
    
    Args:
        feature_data (pd.DataFrame): Datos con características derivadas.

    Returns:
        pd.DataFrame: Datos con características normalizadas.
    """
    # Agregar variable time_diff
    feature_data['time_diff'] = feature_data.groupby('account_number')['transaction_date'].diff().dt.total_seconds()
    
    # Eliminar NaN introducidos por time_diff
    feature_data = feature_data.dropna(subset=['time_diff'])

    # Convertir a tipo numérico
    features = ['transaction_amount', 'time_diff']
    feature_data[features] = feature_data[features].apply(pd.to_numeric, errors='coerce')
    
    # Eliminar filas con NaN después de la conversión
    feature_data = feature_data.dropna(subset=features)
    
    # Normalizar características numéricas
    feature_data[features] = feature_data[features].apply(lambda x: (x - x.mean()) / x.std())
    
    return feature_data
