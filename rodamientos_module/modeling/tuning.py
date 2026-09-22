"""Comparacion de configuraciones de hiperparametros (respuesta a la
observacion del docente: "no demuestra como selecciono ese hiperparametro
ni compara configuraciones").

Toda comparacion se realiza EXCLUSIVAMENTE sobre el conjunto de ajuste
(los 18 archivos que quedan tras separar el test independiente); el
conjunto de test final nunca entra a esta clase.
"""

import pandas as pd

from rodamientos_module.modeling.evaluator import Evaluador
from rodamientos_module.modeling.models import RandomForestModel, SVMModel


class AjustadorHiperparametros:
    """Compara configuraciones de un modelo sobre el conjunto de ajuste."""

    def __init__(self, features, columna_clase="clase"):
        self.evaluador = Evaluador(features=features, columna_clase=columna_clase)

    def comparar_random_forest(self, df_ajuste, folds, configuraciones):
        filas = []
        for config in configuraciones:
            modelo = RandomForestModel(**config)
            tabla_fold, _ = self.evaluador.evaluar_por_fold(modelo, df_ajuste, folds)
            promedio = Evaluador.resumen_promedio(tabla_fold)
            filas.append({**config, **promedio.to_dict()})
        return pd.DataFrame(filas)

    def comparar_svm(self, df_ajuste, folds, configuraciones):
        filas = []
        for config in configuraciones:
            modelo = SVMModel(**config)
            tabla_fold, _ = self.evaluador.evaluar_por_fold(modelo, df_ajuste, folds)
            promedio = Evaluador.resumen_promedio(tabla_fold)
            filas.append({**config, **promedio.to_dict()})
        return pd.DataFrame(filas)

    @staticmethod
    def mejor_configuracion(tabla_resultados, columnas_config, metrica="balanced_accuracy"):
        """Selecciona la fila con mejor metrica y devuelve solo las columnas
        de configuracion (listas para pasarselas a **kwargs de un modelo)."""
        mejor_fila = tabla_resultados.loc[tabla_resultados[metrica].idxmax()]
        configuracion = {columna: mejor_fila[columna] for columna in columnas_config}
        # max_depth suele venir como float por los NaN de pandas; se corrige a int/None
        if "max_depth" in configuracion:
            valor = configuracion["max_depth"]
            configuracion["max_depth"] = None if pd.isna(valor) else int(valor)
        if "n_estimators" in configuracion:
            configuracion["n_estimators"] = int(configuracion["n_estimators"])
        return configuracion
