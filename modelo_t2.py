# 1. Importar las librerías necesarias
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import mlflow

# Configuración para una mejor visualización
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def save_dataframe_to_csv(df, base_filename="base_2"):
    """
    Guarda un DataFrame en un archivo CSV en la carpeta del proyecto.
    Añade una marca de tiempo al nombre del archivo para evitar sobrescribir.

    Args:
        df (pd.DataFrame): El DataFrame que se va a guardar.
        base_filename (str): El nombre base para el archivo CSV.
    """
    # Crear un nombre de archivo único con una marca de tiempo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.csv"
    try:
        df.to_csv(filename, index=False)
        print(f"\nDataFrame guardado exitosamente como: {filename}")
    except Exception as e:
        print(f"\nOcurrió un error al guardar el archivo: {e}")

# 2. Cargar el dataset en un DataFrame de pandas
try:
    df = pd.read_csv('engineered_heart_data.csv')
    print("Dataset 'engineered_heart_data.csv' cargado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'engineered_heart_data.csv' no se encontró.")
    print("Asegúrate de que el archivo CSV generado en el Taller 1 esté en el mismo directorio.")
    df = None

if df is not None:
    # 3. Preparación de datos para la regresión
    print("\n--- Preparando datos para el modelo de regresión ---")

    # Seleccionar la variable objetivo
    TARGET_VARIABLE = 'thalch'
    print(f"Variable objetivo: {TARGET_VARIABLE}")

    # Eliminar filas con valores nulos en la variable objetivo
    df.dropna(subset=[TARGET_VARIABLE], inplace=True)

    # Separar características (X) y objetivo (y)
    X = df.drop(TARGET_VARIABLE, axis=1)
    y = df[TARGET_VARIABLE]

    # Identificar columnas categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    print(f"Columnas categóricas: {list(categorical_features)}")
    print(f"Columnas numéricas: {list(numerical_features)}")

    # Convertir columnas numéricas a float64 para evitar advertencias de MLflow
    # sobre esquemas con enteros que no soportan valores nulos.
    X[numerical_features] = X[numerical_features].astype('float64')

    # Crear un transformador para preprocesar las columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Establecer la URI de seguimiento de MLflow para usar el servidor local
    mlflow.set_tracking_uri("http://localhost:5000")

    # Habilitar autologging para scikit-learn
    mlflow.sklearn.autolog()

    # Iniciar un "run" de MLflow
    with mlflow.start_run():
        # 4. Entrenamiento del modelo de Regresión Lineal
        print("\n--- Entrenando modelo de Regresión Lineal ---")
        # Crear y entrenar un pipeline que incluye el preprocesador y el modelo
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])
        pipeline.fit(X_train, y_train)
        print("Modelo entrenado exitosamente.")

        # 4.1. Mostrar coeficientes del modelo
        print("\n--- Coeficientes del Modelo de Regresión Lineal ---")
        # Acceder a los pasos del pipeline para obtener la información
        preprocessor_step = pipeline.named_steps['preprocessor']
        regressor_step = pipeline.named_steps['regressor']
        feature_names = preprocessor_step.get_feature_names_out()
        coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient': regressor_step.coef_})
        coefficients['Absolute_Coefficient'] = abs(coefficients['Coefficient'])
        coefficients = coefficients.sort_values(by='Absolute_Coefficient', ascending=False)
        print(coefficients.drop(columns='Absolute_Coefficient').to_string())

        # 5. Evaluación del modelo
        print("\n--- Evaluando el modelo ---")
        y_pred = pipeline.predict(X_test)

        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Mostrar métricas en formato de tabla
        metrics_df = pd.DataFrame({
            'Metric': ['Error Cuadrático Medio (MSE)', 'Coeficiente de Determinación (R²)'],
            'Value': [f"{mse:.2f}", f"{r2:.2f}"]
        })
        print(metrics_df.to_string(index=False))

        # 6. Visualización de los resultados
        # Gráfico de valores reales vs. predichos
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.6})
        plt.xlabel("Valores Reales (thalch)")
        plt.ylabel("Predicciones del Modelo")
        plt.title("Regresión Lineal: Valores Reales vs. Predicciones")
        plt.show()

        # Gráfico de residuos
        plt.figure(figsize=(10, 6))
        sns.residplot(x=y_pred, y=y_test - y_pred, scatter_kws={'alpha': 0.5})
        plt.xlabel("Predicciones del Modelo")
        plt.ylabel("Residuos")
        plt.title("Gráfico de Residuos")
        plt.show()
