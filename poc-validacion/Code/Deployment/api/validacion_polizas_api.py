# -*- coding: utf-8 -*-
import os
import pandas as pd
import uvicorn
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Create the app
app = FastAPI( title="API Modelo de clasificación de polizas 📃📊 ",
    description="""A partír de la información de una póliza el modelo predice si la póliza es válida o no.""",
    version="0.0.1")

# Load trained Pipeline
model_path = os.path.join(os.path.dirname(__file__), '../../../Models/best_model_validacion_polizas')
model = load_model(model_path)

# Define input/output pydantic models
class InputModel(BaseModel):
    Prima: float = Field(default=5001.791696691247)
    Cobertura: str = Field(default='Premium')
    Edad_Cliente: float = Field(default=78.0)
    Termino_Poliza: float = Field(default=1.0)
    Historial_Reclamos: float = Field(default=4.0)
    Fecha_Creacion: str = Field(default='2018-04-02')
    Estado_Poliza: str = Field(default='Vencida')
    Region: str = Field(default='Sur')
    Genero_Cliente: str = Field(default='Femenino')
    Tipo_Vehiculo: str = Field(default='Camión')
    Valor_Vehiculo: float = Field(default=110718.88521071748)
    Kilometraje_Anual: float = Field(default=6205.0)
    Ingresos_Cliente: float = Field(default=1017.7196315728144)
    Dependientes: float = Field(default=3.0)
    Riesgo: str = Field(default='Alto')
    Fecha_Ultimo_Reclamo: str = Field(default='2015-03-06')
    Cobertura_Salud: str = Field(default='Sí')
    Monto_Reclamos: float = Field(default=73797.76174213919)
    Descuento: float = Field(default=0.2118011203613879)

class OutputModel(BaseModel):
    prediction: int

# Define predict function
@app.post("/predict", response_model=OutputModel)
def predict(data: InputModel):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
