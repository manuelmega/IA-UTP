import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Frecuencia Cardíaca Máxima",
    description="Una API para predecir 'thalch' usando un modelo de regresión lineal entrenado.",
    version="1.0.0"
)

# --- Carga del Modelo ---
# Establecer la URI de seguimiento de MLflow para conectarse al servidor
mlflow.set_tracking_uri("http://localhost:5000")

try:
    # Obtener el último 'run' del experimento por defecto (experiment_id='0')
    runs = mlflow.search_runs(experiment_ids=['0'], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        raise RuntimeError("No se encontraron runs en MLflow.")
    
    latest_run_id = runs.iloc[0].run_id
    # Construir la URI del modelo
    model_uri = f"runs:/{latest_run_id}/model"
    
    # Cargar el pipeline del modelo (que incluye preprocesador y regresor)
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Modelo cargado exitosamente desde el run_id: {latest_run_id}")

except Exception as e:
    print(f"Error al cargar el modelo desde MLflow: {e}")
    # Si no se puede cargar el modelo, la API no podrá hacer predicciones.
    # Podríamos detener el inicio de la app, pero por ahora solo imprimiremos el error.
    model = None

# --- Definición del Modelo de Datos de Entrada ---
# Pydantic model para validar los datos de entrada de la API
class HeartDataInput(BaseModel):
    age: float
    trestbps: float
    chol: float
    oldpeak: float
    sex: str = Field(..., example="male")
    cp: str = Field(..., example="asymptomatic")
    fbs: str = Field(..., example=">120")
    restecg: str = Field(..., example="normal")
    exang: str = Field(..., example="no")
    slope: str = Field(..., example="upsloping")
    ca: str = Field(..., example="0")
    thal: str = Field(..., example="normal")
    age_group: str = Field(..., example="50-59")
    trestbps_group: str = Field(..., example="High")
    chol_group: str = Field(..., example="High")

# --- Endpoint de Predicción ---
@app.post("/predict")
def predict(data: HeartDataInput):
    """
    Realiza una predicción de la frecuencia cardíaca máxima (`thalch`).

    - **data**: Un objeto JSON con las características del paciente.
    
    Retorna un objeto JSON con la predicción.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Revise los logs del servidor.")

    try:
        # Convertir los datos de entrada Pydantic a un DataFrame de pandas
        input_df = pd.DataFrame([data.model_dump()])
        
        # Realizar la predicción
        prediction = model.predict(input_df)
        
        # Devolver la predicción en una respuesta JSON
        return {"predicted_thalch": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {str(e)}")