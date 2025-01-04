# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("validacion_polizas_model")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)