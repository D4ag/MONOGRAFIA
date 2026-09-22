# Clasificacion de condicion de rodamientos (CWRU)

Monografia de Ciencia de Datos — Diplomado en Ciencia de Datos, UMSS.
Clasificacion multiclase supervisada (Normal / Ball / Inner Race / Outer
Race) a partir de senales de vibracion del conjunto CWRU, siguiendo la
metodologia CRISP-DM.

## Integrante

- Diego Alvarado Garcia

## Organizacion del proyecto

Estructura basada en el patron cookiecutter-data-science, adaptada al
alcance de la monografia (sin servicios de despliegue ni tracking de
experimentos, ya que la seccion 7.6 presenta una **propuesta** de
despliegue, no una implementacion real):

```
├── README.md              <- Este archivo.
├── requirements.txt       <- Dependencias para reproducir el analisis.
│
├── data
│   ├── raw                <- Archivos .mat originales del conjunto CWRU (24 archivos).
│   ├── interim             <- Tablas intermedias (ej. calidad de datos, estadisticos).
│   └── processed           <- Tabla minable final (994 ventanas x 5 caracteristicas).
│
├── models                 <- Modelo final entrenado (modelo_final.pkl).
│
├── notebooks               <- Notebooks numerados; solo narran el analisis,
│                              la logica reutilizable vive en rodamientos_module/.
│   ├── 1.0-da-comprension-datos.ipynb     -> seccion 7.2
│   ├── 2.0-da-preparacion-datos.ipynb     -> seccion 7.3
│   ├── 3.0-da-modelado.ipynb              -> seccion 7.4
│   └── 4.0-da-evaluacion-resultados.ipynb -> seccion 7.5
│
├── references              <- Diccionario de datos (Anexo A de la monografia).
├── reports/figures         <- Figuras exportadas para el documento Word.
│
└── rodamientos_module       <- Codigo fuente del proyecto.
    ├── config.py            <- Rutas, parametros de senal y particiones (unica fuente de verdad).
    ├── dataset.py            <- CargadorCWRU: lectura y validacion de archivos .mat.
    ├── features.py           <- ExtractorCaracteristicas + ConstructorTablaModelado.
    ├── particionado.py       <- ParticionadorPorArchivo: folds y test independiente.
    │
    ├── eda/                  <- Analisis exploratorio (seccion 7.2).
    │
    └── modeling
        ├── models.py         <- BaseModel (ABC) + Baseline/LogisticRegression/SVM/RandomForest.
        ├── evaluator.py       <- Evaluador: validacion cruzada agrupada por archivo.
        ├── tuning.py          <- AjustadorHiperparametros: comparacion de configuraciones.
        ├── train.py           <- Orquesta comparacion inicial, ajuste y evaluacion final.
        └── predict.py         <- Inferencia con el modelo final (seccion 7.6).
```

## Decisiones metodologicas clave

- **Unidad de independencia:** el archivo experimental, no la ventana. Las
  994 ventanas provienen de solo 24 registros; toda validacion agrupa por
  archivo para evitar fuga de informacion (`ParticionadorPorArchivo`).
- **Test independiente:** 6 archivos (`config.ARCHIVOS_TEST_FINAL`) se
  separan ANTES de comparar modelos o ajustar hiperparametros, y se
  evaluan una unica vez al final (`train.evaluar_modelo_final`).
- **Hiperparametros:** justificados por comparacion explicita de
  configuraciones sobre el conjunto de ajuste (`modeling/tuning.py`), no
  fijados de antemano.

## Como reproducir el analisis

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Ejecutar los notebooks en orden (1.0 -> 2.0 -> 3.0 -> 4.0). Cada uno
importa `rodamientos_module` y llama a sus clases; el codigo de
procesamiento no se reescribe en los notebooks.
