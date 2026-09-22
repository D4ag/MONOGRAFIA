"""Orquesta el flujo completo de modelado y evaluacion (secciones 7.4 y 7.5).

Se exponen cuatro funciones, pensadas para llamarse en orden desde el
notebook 3.0-da-modelado.ipynb:

1. comparacion_inicial: reproduce la comparacion original de las 4
   particiones sobre TODO el dataset (trazabilidad con las Tablas 9-14
   ya presentadas en la monografia).
2. ajustar_hiperparametros: compara configuraciones de RF y SVM SOLO
   sobre el conjunto de ajuste (18 archivos).
3. comparar_modelos_ajustados: compara baseline/RL/SVM/RF ya ajustados,
   tambien solo sobre el conjunto de ajuste, para decidir el modelo final.
4. evaluar_modelo_final: reentrena con todo el conjunto de ajuste y
   evalua UNA sola vez sobre el test independiente.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from rodamientos_module import config
from rodamientos_module.modeling.evaluator import Evaluador
from rodamientos_module.modeling.models import (
    BaselineModel,
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
)
from rodamientos_module.modeling.tuning import AjustadorHiperparametros


def comparacion_inicial(df_modelo_base):
    """Comparacion de Baseline/RL/SVM/RF con las 4 particiones originales
    sobre todo el dataset (994 ventanas). Corresponde a las Tablas 9-14."""
    evaluador = Evaluador(
        features=config.CARACTERISTICAS_MODELO,
        columna_clase=config.COLUMNA_CLASE,
        orden_clases=config.ORDEN_CLASES,
    )
    modelos = {
        "Baseline": BaselineModel(),
        "Regresion Logistica": LogisticRegressionModel(random_state=config.RANDOM_STATE),
        "SVM": SVMModel(random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestModel(n_estimators=200, random_state=config.RANDOM_STATE),
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        tabla_fold, predicciones = evaluador.evaluar_por_fold(
            modelo, df_modelo_base, config.FOLDS_ARCHIVOS_COMPLETOS
        )
        resultados[nombre] = {
            "por_fold": tabla_fold,
            "promedio": Evaluador.resumen_promedio(tabla_fold),
            "predicciones": predicciones,
        }
    return resultados


def ajustar_hiperparametros(df_ajuste):
    """Compara configuraciones de Random Forest y SVM SOLO sobre el
    conjunto de ajuste, respondiendo a la observacion de justificar
    hiperparametros en vez de fijarlos sin comparacion."""
    ajustador = AjustadorHiperparametros(features=config.CARACTERISTICAS_MODELO)

    configuraciones_rf = [
        {"n_estimators": 50, "max_depth": None},
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
    ]
    tabla_rf = ajustador.comparar_random_forest(
        df_ajuste, config.FOLDS_AJUSTE, configuraciones_rf
    )

    configuraciones_svm = [
        {"kernel": "linear", "C": 1},
        {"kernel": "rbf", "C": 0.1},
        {"kernel": "rbf", "C": 1},
        {"kernel": "rbf", "C": 10},
    ]
    tabla_svm = ajustador.comparar_svm(
        df_ajuste, config.FOLDS_AJUSTE, configuraciones_svm
    )

    mejor_rf = AjustadorHiperparametros.mejor_configuracion(
        tabla_rf, ["n_estimators", "max_depth"]
    )
    mejor_svm = AjustadorHiperparametros.mejor_configuracion(
        tabla_svm, ["kernel", "C"]
    )
    return tabla_rf, tabla_svm, mejor_rf, mejor_svm


def comparar_modelos_ajustados(df_ajuste, mejor_rf, mejor_svm):
    """Compara Baseline/RL/SVM(ajustado)/RF(ajustado) sobre el conjunto de
    ajuste, para decidir el modelo final ANTES de tocar el test independiente."""
    evaluador = Evaluador(
        features=config.CARACTERISTICAS_MODELO, columna_clase=config.COLUMNA_CLASE
    )
    modelos = {
        "Baseline": BaselineModel(),
        "Regresion Logistica": LogisticRegressionModel(random_state=config.RANDOM_STATE),
        "SVM (ajustado)": SVMModel(random_state=config.RANDOM_STATE, **mejor_svm),
        "Random Forest (ajustado)": RandomForestModel(
            random_state=config.RANDOM_STATE, **mejor_rf
        ),
    }

    filas = []
    for nombre, modelo in modelos.items():
        tabla_fold, _ = evaluador.evaluar_por_fold(modelo, df_ajuste, config.FOLDS_AJUSTE)
        promedio = Evaluador.resumen_promedio(tabla_fold)
        filas.append({"modelo": nombre, **promedio.to_dict()})

    tabla_comparacion = pd.DataFrame(filas)
    return tabla_comparacion, modelos


def evaluar_modelo_final(df_ajuste, df_test_final, modelo_base_model):
    """Reentrena modelo_base_model con TODO el conjunto de ajuste y lo
    evalua UNA UNICA VEZ sobre el conjunto de test independiente.

    Este es el unico resultado que debe reportarse como desempeno final
    del proyecto (respuesta a la observacion de no tener un test
    independiente posterior a la seleccion del modelo).
    """
    X_ajuste = df_ajuste[config.CARACTERISTICAS_MODELO]
    y_ajuste = df_ajuste[config.COLUMNA_CLASE]
    X_test = df_test_final[config.CARACTERISTICAS_MODELO]
    y_test = df_test_final[config.COLUMNA_CLASE]

    modelo = modelo_base_model.construir()
    modelo.fit(X_ajuste, y_ajuste)
    y_pred = modelo.predict(X_test)

    resultado = {
        "modelo": modelo_base_model.nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
    matriz = confusion_matrix(y_test, y_pred, labels=config.ORDEN_CLASES)
    return resultado, matriz, modelo
