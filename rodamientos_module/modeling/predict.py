"""Inferencia con el modelo final ya entrenado (seccion 7.6 - propuesta de
utilizacion). No implementa un despliegue real; sirve para aplicar el
modelo guardado en models/ sobre una nueva senal de vibracion, siguiendo
el mismo flujo descrito en el diagrama de la seccion 7.6.2:

    senal -> segmentacion en ventanas -> extraccion de caracteristicas ->
    modelo -> condicion estimada
"""

import joblib
import pandas as pd

from rodamientos_module import config
from rodamientos_module.features import ConstructorTablaModelado


def cargar_modelo(ruta_modelo=None):
    ruta_modelo = ruta_modelo or (config.DIR_MODELOS / "modelo_final.pkl")
    return joblib.load(ruta_modelo)


def predecir_senal(senal, modelo=None, tamano_ventana=None, fs=None):
    """Segmenta una senal cruda en ventanas, extrae caracteristicas y
    devuelve la condicion estimada para cada ventana.

    Uso previsto: aplicar el modelo final a una senal nueva (no vista
    durante el entrenamiento ni la evaluacion), tal como se describe en
    la propuesta de despliegue de la seccion 7.6.
    """
    modelo = modelo or cargar_modelo()
    constructor = ConstructorTablaModelado(
        tamano_ventana=tamano_ventana or config.TAMANO_VENTANA,
        fs=fs or config.FRECUENCIA_MUESTREO,
    )

    ventanas = constructor.segmentar(senal)
    filas = [constructor.extractor.extraer_todas(v) for v in ventanas]
    tabla_caracteristicas = pd.DataFrame(filas)

    predicciones = modelo.predict(tabla_caracteristicas[config.CARACTERISTICAS_MODELO])
    tabla_caracteristicas["condicion_estimada"] = predicciones
    return tabla_caracteristicas
