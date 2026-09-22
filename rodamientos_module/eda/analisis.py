"""Analisis exploratorio de datos (seccion 7.2) y de caracteristicas
(seccion 7.3.3-7.3.4): funciones auxiliares para producir las figuras y
tablas descritas en la monografia, separadas de rodamientos_module.features
(que solo calcula las 5 caracteristicas finales usadas en el modelado).

Responsabilidad unica: analisis exploratorio y visualizacion, no
modelado.
"""

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

# Multiplicadores estandar de frecuencias caracteristicas para el rodamiento
# SKF 6205-2RS JEM (Drive End) del conjunto CWRU, expresados como razon
# respecto de la frecuencia de giro (fr = rpm / 60), documentados por el
# Bearing Data Center de Case Western Reserve University.
FACTOR_BPFO = 3.5848
FACTOR_BPFI = 5.4152
FACTOR_BSF = 2.3567
FACTOR_FTF = 0.3983


def frecuencias_caracteristicas(rpm):
    """Devuelve BPFO, BPFI, BSF y FTF (en Hz) para una velocidad de giro dada."""
    fr = rpm / 60.0
    return {
        "BPFO": FACTOR_BPFO * fr,
        "BPFI": FACTOR_BPFI * fr,
        "BSF": FACTOR_BSF * fr,
        "FTF": FACTOR_FTF * fr,
    }


def envolvente_espectro(senal, fs=12000):
    """Calcula el espectro de la envolvente de una senal mediante la
    transformada de Hilbert (demodulacion de amplitud), utilizado para
    resaltar las frecuencias caracteristicas de falla (seccion 7.2.4)."""
    senal_centrada = senal - np.mean(senal)
    envolvente = np.abs(hilbert(senal_centrada))
    envolvente_centrada = envolvente - np.mean(envolvente)
    n = len(envolvente_centrada)
    espectro = np.abs(np.fft.rfft(envolvente_centrada))
    frecuencias = np.fft.rfftfreq(n, d=1 / fs)
    return frecuencias, espectro


def caracteristicas_exploratorias_ventana(ventana, fs=12000):
    """Calcula un conjunto ampliado de caracteristicas temporales y
    frecuenciales (mas alla de las 5 finales) para el analisis exploratorio
    de correlacion (seccion 7.3.3, Figura 6)."""
    rms = np.sqrt(np.mean(ventana ** 2))
    n = len(ventana)
    ventana_centrada = ventana - np.mean(ventana)
    espectro = np.fft.rfft(ventana_centrada)
    frecuencias = np.fft.rfftfreq(n, d=1 / fs)
    magnitud = np.abs(espectro)
    frecuencias_no_dc = frecuencias[1:]
    magnitud_no_dc = magnitud[1:]

    centroide_espectral = np.sum(frecuencias_no_dc * magnitud_no_dc) / np.sum(magnitud_no_dc)
    frecuencia_rms = np.sqrt(
        np.sum((frecuencias_no_dc ** 2) * magnitud_no_dc) / np.sum(magnitud_no_dc)
    )
    energia_espectral = np.sum(magnitud_no_dc ** 2)

    return {
        "media": np.mean(ventana),
        "std": np.std(ventana),
        "rms": rms,
        "min": np.min(ventana),
        "max": np.max(ventana),
        "pico_pico": np.ptp(ventana),
        "skewness": skew(ventana),
        "kurtosis": kurtosis(ventana),
        "factor_cresta": np.max(np.abs(ventana)) / rms,
        "centroide_espectral": centroide_espectral,
        "frecuencia_rms": frecuencia_rms,
        "energia_espectral": energia_espectral,
    }


def tabla_caracteristicas_exploratorias(registros, tamano_ventana=4096, fs=12000):
    """Construye una tabla ampliada de caracteristicas por ventana (no solo
    las 5 finales), usada exclusivamente para el analisis exploratorio de
    correlacion entre variables (Figura 6). No se usa para el modelado."""
    filas = []
    for registro in registros:
        senal = registro["senal"]
        n_ventanas = len(senal) // tamano_ventana
        for i in range(n_ventanas):
            ventana = senal[i * tamano_ventana:(i + 1) * tamano_ventana]
            caracteristicas = caracteristicas_exploratorias_ventana(ventana, fs=fs)
            filas.append({
                "archivo": registro["archivo"],
                "clase": registro["clase"],
                "ventana": i,
                **caracteristicas,
            })
    return pd.DataFrame(filas)


def rms_por_ventana(senal, tamano_ventana=4096):
    n_ventanas = len(senal) // tamano_ventana
    return np.array([
        np.sqrt(np.mean(senal[i * tamano_ventana:(i + 1) * tamano_ventana] ** 2))
        for i in range(n_ventanas)
    ])
