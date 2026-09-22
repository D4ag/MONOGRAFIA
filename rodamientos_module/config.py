"""Configuracion y constantes del proyecto de clasificacion de rodamientos (CWRU).

Centraliza rutas, parametros de senal y las particiones documentadas en la
monografia (seccion 7.3), para que notebooks y modulos usen siempre la misma
version de la verdad en vez de repetir literales.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto (estilo cookiecutter-data-science)
# ---------------------------------------------------------------------------
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIR_DATOS_RAW = RAIZ_PROYECTO / "data" / "raw"
DIR_DATOS_INTERIM = RAIZ_PROYECTO / "data" / "interim"
DIR_DATOS_PROCESSED = RAIZ_PROYECTO / "data" / "processed"
DIR_MODELOS = RAIZ_PROYECTO / "models"
DIR_REPORTES_FIGURAS = RAIZ_PROYECTO / "reports" / "figures"
DIR_REFERENCIAS = RAIZ_PROYECTO / "references"

# ---------------------------------------------------------------------------
# Parametros de senal (seccion 7.3.1 - 7.3.2)
# ---------------------------------------------------------------------------
FRECUENCIA_MUESTREO = 12000  # Hz, ensayos de 12 kHz Drive End
TAMANO_VENTANA = 4096        # muestras por ventana (seccion 7.3.2, Tabla 7)

# ---------------------------------------------------------------------------
# Variables predictoras seleccionadas (seccion 7.3.4)
# ---------------------------------------------------------------------------
CARACTERISTICAS_MODELO = [
    "rms",
    "skewness",
    "kurtosis",
    "factor_cresta",
    "centroide_espectral",
]

COLUMNA_CLASE = "clase"
COLUMNA_GRUPO = "archivo"

ORDEN_CLASES = ["Normal", "Ball", "Inner Race", "Outer Race"]

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Archivos experimentales por condicion (seccion 7.2.1, Tabla 1)
# ---------------------------------------------------------------------------
ARCHIVOS_POR_CLASE = {
    "Normal": ["97.mat", "98.mat", "99.mat", "100.mat"],
    "Ball": ["118.mat", "119.mat", "120.mat", "121.mat"],
    "Inner Race": ["105.mat", "106.mat", "107.mat", "108.mat"],
    "Outer Race": [
        "130.mat", "131.mat", "132.mat", "133.mat",
        "144.mat", "145.mat", "146.mat", "147.mat",
        "156.mat", "158.mat", "159.mat", "160.mat",
    ],
}

# ---------------------------------------------------------------------------
# Particiones originales (4 folds), usadas en la comparacion inicial
# documentada en las secciones 7.4 y 7.5 sobre el conjunto completo.
# ---------------------------------------------------------------------------
FOLDS_ARCHIVOS_COMPLETOS = {
    1: {"Ball": ["118.mat"], "Inner Race": ["105.mat"], "Normal": ["100.mat"],
        "Outer Race": ["130.mat", "131.mat", "132.mat"]},
    2: {"Ball": ["119.mat"], "Inner Race": ["106.mat"], "Normal": ["97.mat"],
        "Outer Race": ["133.mat", "144.mat", "145.mat"]},
    3: {"Ball": ["120.mat"], "Inner Race": ["107.mat"], "Normal": ["98.mat"],
        "Outer Race": ["146.mat", "147.mat", "156.mat"]},
    4: {"Ball": ["121.mat"], "Inner Race": ["108.mat"], "Normal": ["99.mat"],
        "Outer Race": ["158.mat", "159.mat", "160.mat"]},
}

# ---------------------------------------------------------------------------
# Conjunto de test independiente, reservado ANTES de cualquier comparacion
# de modelos o ajuste de hiperparametros (respuesta a observacion del
# docente: "no existe un test independiente posterior a la seleccion").
# ---------------------------------------------------------------------------
ARCHIVOS_TEST_FINAL = {
    "Normal": ["100.mat"],
    "Ball": ["118.mat"],
    "Inner Race": ["105.mat"],
    "Outer Race": ["130.mat", "144.mat", "156.mat"],
}

# Particiones (3 folds) sobre el conjunto de ajuste (18 archivos restantes
# una vez separado el test final), agrupadas por archivo experimental.
FOLDS_AJUSTE = {
    1: {"Normal": ["97.mat"], "Ball": ["119.mat"], "Inner Race": ["106.mat"],
        "Outer Race": ["131.mat", "132.mat", "133.mat"]},
    2: {"Normal": ["98.mat"], "Ball": ["120.mat"], "Inner Race": ["107.mat"],
        "Outer Race": ["145.mat", "146.mat", "147.mat"]},
    3: {"Normal": ["99.mat"], "Ball": ["121.mat"], "Inner Race": ["108.mat"],
        "Outer Race": ["158.mat", "159.mat", "160.mat"]},
}
