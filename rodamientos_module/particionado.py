"""Estrategia de particionado y validacion (secciones 7.3.6 y 7.3.7).

Toda la logica de "que archivo va en que conjunto" vive aca, en un solo
lugar, para que sea imposible que un archivo termine repetido entre el
conjunto de ajuste y el test independiente sin que una verificacion lo
detecte de inmediato.
"""


class ParticionadorPorArchivo:
    """Separa el conjunto de datos usando el archivo experimental como
    unidad de agrupacion, evitando que ventanas del mismo registro queden
    repartidas entre entrenamiento y evaluacion (fuga de informacion)."""

    def __init__(self, df, columna_grupo="archivo", columna_clase="clase"):
        self.df = df
        self.columna_grupo = columna_grupo
        self.columna_clase = columna_clase

    def separar_test_final(self, archivos_test_final: dict):
        """Devuelve (df_ajuste, df_test_final) segun el diccionario de
        archivos reservados como test independiente."""
        lista_test = [a for lista in archivos_test_final.values() for a in lista]
        es_test = self.df[self.columna_grupo].isin(lista_test)
        df_test = self.df[es_test].copy()
        df_ajuste = self.df[~es_test].copy()
        return df_ajuste, df_test

    @staticmethod
    def indices_fold(df, folds: dict, fold_actual: int, columna_grupo="archivo"):
        """Devuelve (indices_train, indices_test) para una particion,
        usando el archivo experimental como unidad de agrupacion."""
        archivos_test = [a for lista in folds[fold_actual].values() for a in lista]
        es_test = df[columna_grupo].isin(archivos_test)
        return df[~es_test].index, df[es_test].index

    @staticmethod
    def verificar_sin_interseccion(*grupos_de_archivos):
        """Verifica que ningun archivo se repita entre conjuntos
        (ej. folds de ajuste vs. test final). Lanza ValueError si encuentra
        una interseccion, en vez de dejar pasar una fuga de datos en silencio."""
        vistos = set()
        for grupo in grupos_de_archivos:
            interseccion = vistos & set(grupo)
            if interseccion:
                raise ValueError(f"Archivos repetidos entre conjuntos: {interseccion}")
            vistos |= set(grupo)
        return True

    @staticmethod
    def archivos_en_folds(folds: dict):
        """Aplana un diccionario de folds a la lista de archivos que contiene."""
        return [a for fold in folds.values() for lista in fold.values() for a in lista]
