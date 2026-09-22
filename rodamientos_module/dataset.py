"""Carga y validacion de las senales Drive End (DE) del conjunto CWRU.

Responsabilidad unica: leer los archivos .mat crudos desde data/raw,
seleccionar la senal DE correcta (incluyendo el caso especial de archivos
con mas de una senal identificable, ej. 99.mat), extraer la RPM cuando
esta disponible, y construir las tablas de calidad y de estadisticos
descriptivos documentadas en las secciones 7.2.1 - 7.2.3 de la monografia.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


class CargadorCWRU:
    """Carga y valida las senales DE de un subconjunto de archivos CWRU."""

    def __init__(self, directorio_datos, archivos_por_clase):
        self.directorio_datos = Path(directorio_datos)
        self.archivos_por_clase = archivos_por_clase
        self._clase_por_archivo = {
            archivo: clase
            for clase, archivos in archivos_por_clase.items()
            for archivo in archivos
        }

    def _extraer_senal_de(self, contenido_mat, nombre_archivo):
        """Selecciona la variable *_DE_time correcta dentro del archivo .mat.

        Algunos archivos (ej. 99.mat) contienen mas de una senal
        identificable en su estructura; en ese caso se selecciona
        explicitamente la que corresponde al identificador numerico del
        archivo, en vez de asumir que todos los archivos tienen la misma
        estructura interna (ver seccion 7.2.2).
        """
        claves_de = [k for k in contenido_mat.keys() if k.endswith("_DE_time")]
        if not claves_de:
            raise ValueError(f"No se encontro una senal DE en {nombre_archivo}")

        if len(claves_de) > 1:
            identificador = re.findall(r"\d+", nombre_archivo)[0]
            claves_filtradas = [k for k in claves_de if identificador in k]
            claves_de = claves_filtradas or claves_de

        return contenido_mat[claves_de[0]].ravel().astype(float)

    def _extraer_rpm(self, contenido_mat):
        claves_rpm = [k for k in contenido_mat.keys() if "RPM" in k]
        if not claves_rpm:
            return np.nan
        return float(np.ravel(contenido_mat[claves_rpm[0]])[0])

    def cargar_archivo(self, nombre_archivo):
        """Carga un unico archivo .mat y devuelve su registro completo."""
        ruta = self.directorio_datos / nombre_archivo
        contenido = sio.loadmat(ruta)
        senal = self._extraer_senal_de(contenido, nombre_archivo)
        rpm = self._extraer_rpm(contenido)
        return {
            "archivo": nombre_archivo,
            "clase": self._clase_por_archivo[nombre_archivo],
            "senal": senal,
            "rpm": rpm,
            "muestras": len(senal),
        }

    def cargar_todos(self):
        """Carga todos los archivos configurados en archivos_por_clase."""
        return [self.cargar_archivo(archivo) for archivo in self._clase_por_archivo]

    @staticmethod
    def tabla_calidad(registros):
        """Replica la Tabla 2: NaN, infinitos, duplicados y RPM disponible."""
        senales_vistas = {}
        filas = []
        for registro in registros:
            senal = registro["senal"]
            hash_senal = hash(senal.tobytes())
            duplicado = hash_senal in senales_vistas
            senales_vistas[hash_senal] = registro["archivo"]
            filas.append({
                "archivo": registro["archivo"],
                "clase": registro["clase"],
                "muestras": registro["muestras"],
                "nan_en_de": int(np.isnan(senal).sum()),
                "infinitos_en_de": int(np.isinf(senal).sum()),
                "rpm_disponible": not np.isnan(registro["rpm"]),
                "duplicado": duplicado,
            })
        return pd.DataFrame(filas)

    @staticmethod
    def tabla_estadisticos_descriptivos(registros, fs=12000):
        """Replica la Tabla 3: media, desviacion, RMS, min y max por archivo."""
        filas = []
        for registro in registros:
            senal = registro["senal"]
            filas.append({
                "archivo": registro["archivo"],
                "clase": registro["clase"],
                "muestras": registro["muestras"],
                "duracion_s": registro["muestras"] / fs,
                "rpm": registro["rpm"],
                "media": np.mean(senal),
                "desv_estandar": np.std(senal),
                "rms": np.sqrt(np.mean(senal ** 2)),
                "min": np.min(senal),
                "max": np.max(senal),
            })
        return pd.DataFrame(filas)
