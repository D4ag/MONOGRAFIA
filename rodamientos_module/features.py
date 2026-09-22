"""Ingenieria de caracteristicas: segmentacion en ventanas y extraccion de
caracteristicas temporales y frecuenciales (secciones 7.3.2 - 7.3.5).

Responsabilidad unica dividida en dos clases:
- ExtractorCaracteristicas: calcula las caracteristicas de UNA ventana.
- ConstructorTablaModelado: segmenta senales completas en ventanas y arma
  la tabla minable (994 filas x 5 caracteristicas + metadatos).
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class ExtractorCaracteristicas:
    """Extrae caracteristicas temporales y frecuenciales de una ventana de senal."""

    def __init__(self, fs=12000):
        self.fs = fs

    def temporales(self, ventana):
        rms = np.sqrt(np.mean(ventana ** 2))
        return {
            "rms": rms,
            "skewness": skew(ventana),
            "kurtosis": kurtosis(ventana),
            "factor_cresta": np.max(np.abs(ventana)) / rms,
        }

    def frecuenciales(self, ventana):
        n = len(ventana)
        ventana_centrada = ventana - np.mean(ventana)
        espectro = np.fft.rfft(ventana_centrada)
        frecuencias = np.fft.rfftfreq(n, d=1 / self.fs)
        magnitud = np.abs(espectro)

        # Se descarta la componente DC (indice 0) antes de calcular el centroide
        frecuencias_no_dc = frecuencias[1:]
        magnitud_no_dc = magnitud[1:]

        centroide_espectral = (
            np.sum(frecuencias_no_dc * magnitud_no_dc) / np.sum(magnitud_no_dc)
        )
        return {"centroide_espectral": centroide_espectral}

    def extraer_todas(self, ventana):
        caracteristicas = {}
        caracteristicas.update(self.temporales(ventana))
        caracteristicas.update(self.frecuenciales(ventana))
        return caracteristicas


class ConstructorTablaModelado:
    """Segmenta las senales en ventanas de tamano fijo y construye la tabla minable."""

    def __init__(self, tamano_ventana=4096, fs=12000):
        self.tamano_ventana = tamano_ventana
        self.extractor = ExtractorCaracteristicas(fs=fs)

    def segmentar(self, senal):
        """Segmentacion consecutiva sin solapamiento; las muestras finales
        que no completan una ventana se descartan (seccion 7.3.2)."""
        n_ventanas = len(senal) // self.tamano_ventana
        return [
            senal[i * self.tamano_ventana:(i + 1) * self.tamano_ventana]
            for i in range(n_ventanas)
        ]

    def construir(self, registros):
        """registros: salida de CargadorCWRU.cargar_todos()
        (lista de dicts con 'archivo', 'clase', 'senal').
        """
        filas = []
        for registro in registros:
            ventanas = self.segmentar(registro["senal"])
            for indice_ventana, ventana in enumerate(ventanas):
                caracteristicas = self.extractor.extraer_todas(ventana)
                filas.append({
                    "archivo": registro["archivo"],
                    "clase": registro["clase"],
                    "ventana": indice_ventana,
                    **caracteristicas,
                })
        return pd.DataFrame(filas)
