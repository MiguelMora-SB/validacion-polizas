# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import gradio as gr

# Create the app
app = FastAPI( title="API Modelo de clasificación de polizas 📃📊 ",
    description="""A partír de la información de una póliza el modelo predice si la póliza es válida o no.""",
    version="0.0.1")

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

def predict_gradio(Prima, Cobertura, Edad_Cliente, Termino_Poliza, Historial_Reclamos, Fecha_Creacion, Estado_Poliza, Region, Genero_Cliente, Tipo_Vehiculo, Valor_Vehiculo, Kilometraje_Anual, Ingresos_Cliente, Dependientes, Riesgo, Fecha_Ultimo_Reclamo, Cobertura_Salud, Monto_Reclamos, Descuento):
    input_data = {
        "Prima": Prima,
        "Cobertura": Cobertura,
        "Edad_Cliente": Edad_Cliente,
        "Termino_Poliza": Termino_Poliza,
        "Historial_Reclamos": Historial_Reclamos,
        "Fecha_Creacion": Fecha_Creacion,
        "Estado_Poliza": Estado_Poliza,
        "Region": Region,
        "Genero_Cliente": Genero_Cliente,
        "Tipo_Vehiculo": Tipo_Vehiculo,
        "Valor_Vehiculo": Valor_Vehiculo,
        "Kilometraje_Anual": Kilometraje_Anual,
        "Ingresos_Cliente": Ingresos_Cliente,
        "Dependientes": Dependientes,
        "Riesgo": Riesgo,
        "Fecha_Ultimo_Reclamo": Fecha_Ultimo_Reclamo,
        "Cobertura_Salud": Cobertura_Salud,
        "Monto_Reclamos": Monto_Reclamos,
        "Descuento": Descuento
    }
    input_df = pd.DataFrame([input_data])
    predictions = predict_model(model, data=input_df)
    return predictions["prediction_label"].iloc[0]

gr_interface = gr.Interface(
    fn=predict_gradio,
    inputs=[
        gr.Number(label="Prima", value=5001.791696691247),
        gr.Textbox(label="Cobertura", value='Premium'),
        gr.Number(label="Edad_Cliente", value=78.0),
        gr.Number(label="Termino_Poliza", value=1.0),
        gr.Number(label="Historial_Reclamos", value=4.0),
        gr.Textbox(label="Fecha_Creacion", value='2018-04-02'),
        gr.Textbox(label="Estado_Poliza", value='Vencida'),
        gr.Textbox(label="Region", value='Sur'),
        gr.Textbox(label="Genero_Cliente", value='Femenino'),
        gr.Textbox(label="Tipo_Vehiculo", value='Camión'),
        gr.Number(label="Valor_Vehiculo", value=110718.88521071748),
        gr.Number(label="Kilometraje_Anual", value=6205.0),
        gr.Number(label="Ingresos_Cliente", value=1017.7196315728144),
        gr.Number(label="Dependientes", value=3.0),
        gr.Textbox(label="Riesgo", value='Alto'),
        gr.Textbox(label="Fecha_Ultimo_Reclamo", value='2015-03-06'),
        gr.Textbox(label="Cobertura_Salud", value='Sí'),
        gr.Number(label="Monto_Reclamos", value=73797.76174213919),
        gr.Number(label="Descuento", value=0.2118011203613879)
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="POC Clasificación de polizas 📃📊",
    description="A partír de la información de una póliza el modelo predice si la póliza es válida o no."
)

@app.get("/")
def root():
    gr_interface.launch(share=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)