"""Modelos de clasificacion del proyecto (seccion 7.4).

Cada modelo implementa el contrato BaseModel. Agregar un modelo nuevo solo
requiere heredar de BaseModel e implementar construir() -- el resto del
pipeline (Evaluador, AjustadorHiperparametros, train.py) no necesita
cambiar (principio abierto/cerrado).
"""

from abc import ABC, abstractmethod

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BaseModel(ABC):
    """Contrato comun para todos los modelos de clasificacion del proyecto."""

    nombre: str = "BaseModel"

    @abstractmethod
    def construir(self):
        """Devuelve un estimador/pipeline de scikit-learn listo para .fit().

        Se construye una instancia NUEVA cada vez que se llama, para que
        el mismo objeto BaseModel pueda reutilizarse en varios folds sin
        arrastrar el ajuste (fit) de una particion anterior.
        """
        raise NotImplementedError


class BaselineModel(BaseModel):
    """Modelo base (seccion 7.4.1): predice siempre la clase mayoritaria."""

    nombre = "Baseline"

    def construir(self):
        return DummyClassifier(strategy="most_frequent")


class LogisticRegressionModel(BaseModel):
    """Regresion logistica multiclase (seccion 7.4.2), con estandarizacion
    integrada en un Pipeline para que el escalado se ajuste solo con el
    train de cada fold."""

    nombre = "Regresion Logistica"

    def __init__(self, random_state=42):
        self.random_state = random_state

    def construir(self):
        return Pipeline([
            ("escalador", StandardScaler()),
            ("clasificador", LogisticRegression(
                max_iter=1000, random_state=self.random_state
            )),
        ])


class SVMModel(BaseModel):
    """Maquina de Vectores de Soporte (seccion 7.4.3), kernel y C configurables
    para poder compararlos en AjustadorHiperparametros."""

    nombre = "SVM"

    def __init__(self, kernel="rbf", C=1.0, random_state=42):
        self.kernel = kernel
        self.C = C
        self.random_state = random_state

    def construir(self):
        return Pipeline([
            ("escalador", StandardScaler()),
            ("clasificador", SVC(
                kernel=self.kernel, C=self.C, random_state=self.random_state
            )),
        ])


class RandomForestModel(BaseModel):
    """Random Forest (seccion 7.4.4), n_estimators y max_depth configurables
    para poder compararlos en AjustadorHiperparametros. No requiere
    estandarizacion (modelo basado en arboles)."""

    nombre = "Random Forest"

    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def construir(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )


# Registro de modelos disponibles, por si se quiere iterar sobre todos
# desde un notebook o script sin importar cada clase por separado.
MODEL_REGISTRY = {
    "baseline": BaselineModel,
    "logistica": LogisticRegressionModel,
    "svm": SVMModel,
    "random_forest": RandomForestModel,
}
