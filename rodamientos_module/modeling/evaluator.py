"""Evaluacion de modelos mediante validacion cruzada agrupada por archivo
experimental (secciones 7.4.5 y 7.5).

Toda la logica de "entrenar en N-1 archivos, evaluar en el archivo que
falta, repetir por cada particion" vive en una sola clase, para no
duplicarla entre la comparacion inicial, el ajuste de hiperparametros y
la comparacion de modelos ya ajustados.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from rodamientos_module.particionado import ParticionadorPorArchivo


class Evaluador:
    """Evalua un BaseModel mediante validacion cruzada agrupada por archivo."""

    def __init__(self, features, columna_clase="clase", orden_clases=None):
        self.features = features
        self.columna_clase = columna_clase
        self.orden_clases = orden_clases

    def evaluar_por_fold(self, modelo_base_model, df, folds):
        """Entrena y evalua modelo_base_model en cada particion de folds.

        Devuelve (tabla_por_fold, tabla_predicciones):
        - tabla_por_fold: una fila por particion, con las metricas.
        - tabla_predicciones: una fila por ventana evaluada, con la clase
          real y la predicha (insumo para matrices de confusion por fold).
        """
        registros_metricas = []
        registros_predicciones = []

        for fold in folds:
            idx_train, idx_test = ParticionadorPorArchivo.indices_fold(df, folds, fold)
            X_train = df.loc[idx_train, self.features]
            y_train = df.loc[idx_train, self.columna_clase]
            X_test = df.loc[idx_test, self.features]
            y_test = df.loc[idx_test, self.columna_clase]

            modelo = modelo_base_model.construir()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            registros_metricas.append({
                "fold": fold,
                "modelo": modelo_base_model.nombre,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
            })

            for indice, real, predicho in zip(idx_test, y_test, y_pred):
                registros_predicciones.append({
                    "fold": fold,
                    "modelo": modelo_base_model.nombre,
                    "indice": indice,
                    "real": real,
                    "predicho": predicho,
                })

        tabla_por_fold = pd.DataFrame(registros_metricas)
        tabla_predicciones = pd.DataFrame(registros_predicciones)
        return tabla_por_fold, tabla_predicciones

    @staticmethod
    def resumen_promedio(tabla_por_fold):
        columnas_metricas = [
            "accuracy", "balanced_accuracy", "precision_macro",
            "recall_macro", "f1_macro",
        ]
        return tabla_por_fold[columnas_metricas].mean()

    def matriz_confusion(self, tabla_predicciones):
        """Matriz de confusion agregada sobre todas las particiones."""
        return confusion_matrix(
            tabla_predicciones["real"],
            tabla_predicciones["predicho"],
            labels=self.orden_clases,
        )

    def matrices_por_fold(self, tabla_predicciones):
        """Una matriz de confusion por cada particion (Anexo D.2)."""
        matrices = {}
        for fold in sorted(tabla_predicciones["fold"].unique()):
            datos_fold = tabla_predicciones[tabla_predicciones["fold"] == fold]
            matrices[fold] = confusion_matrix(
                datos_fold["real"], datos_fold["predicho"], labels=self.orden_clases
            )
        return matrices
