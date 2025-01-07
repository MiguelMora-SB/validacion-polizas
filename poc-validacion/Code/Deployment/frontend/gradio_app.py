import os
import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load trained Pipeline
model_path = os.path.join(os.path.dirname(__file__), '../../../Models/best_model_validacion_polizas')
model = load_model(model_path)

# Definir la función predictiva
def predict_gradio(
    Prima,
    Cobertura,
    Edad_Cliente,
    Termino_Poliza,
    Historial_Reclamos,
    Fecha_Creacion,
    Estado_Poliza,
    Region,
    Genero_Cliente,
    Tipo_Vehiculo,
    Valor_Vehiculo,
    Kilometraje_Anual,
    Ingresos_Cliente,
    Dependientes,
    Riesgo,
    Fecha_Ultimo_Reclamo,
    Cobertura_Salud,
    Monto_Reclamos,
    Descuento,
):
    try:
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
            "Descuento": Descuento,
        }
        input_df = pd.DataFrame([input_data])
        predictions = predict_model(model, data=input_df)
        return predictions["prediction_label"].iloc[0]
    except Exception as e:
        return f"Error: {e}"

# Crear la interfaz de Gradio
gr_interface = gr.Interface(
    fn=predict_gradio,
    inputs=[
        gr.Number(label="Prima", value=1000),
        gr.Textbox(label="Cobertura", value="Básica"),
        gr.Number(label="Edad Cliente", value=30),
        gr.Number(label="Término Póliza", value=12),
        gr.Number(label="Historial Reclamos", value=0),
        gr.Textbox(label="Fecha Creación", value="2023-01-01"),
        gr.Textbox(label="Estado Póliza", value="Activa"),
        gr.Textbox(label="Región", value="Centro"),
        gr.Textbox(label="Género Cliente", value="Masculino"),
        gr.Textbox(label="Tipo Vehículo", value="Sedán"),
        gr.Number(label="Valor Vehículo", value=20000),
        gr.Number(label="Kilometraje Anual", value=15000),
        gr.Number(label="Ingresos Cliente", value=50000),
        gr.Number(label="Dependientes", value=2),
        gr.Textbox(label="Riesgo", value="Bajo"),
        gr.Textbox(label="Fecha Último Reclamo", value="2022-12-01"),
        gr.Textbox(label="Cobertura Salud", value="Completa"),
        gr.Number(label="Monto Reclamos", value=0),
        gr.Number(label="Descuento", value=10),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="POC Clasificación de pólizas 📃📊",
    description="A partir de la información de una póliza el modelo predice si la póliza es válida o no.",
)

# Lanzar la aplicación
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    gr_interface.launch(server_name="0.0.0.0", server_port=port)
