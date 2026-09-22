# Diccionario de datos

Corresponde al Anexo A de la monografia.

## Variables originales del conjunto CWRU (archivos .mat)

| Variable | Descripcion | Unidad / Tipo |
|---|---|---|
| `DE_time` | Senal de aceleracion medida en el extremo de accionamiento (Drive End) | m/s^2, serie temporal |
| `FE_time` | Senal de aceleracion medida en el extremo del ventilador (Fan End) | m/s^2, serie temporal |
| `BA_time` | Senal de aceleracion medida en la carcasa del motor (Base) | m/s^2, serie temporal, disponible solo en algunos registros |
| `RPM` | Velocidad de giro del eje durante el ensayo | revoluciones por minuto |

## Variables construidas utilizadas en el modelado

| Variable | Descripcion | Formula / metodo | Definida en |
|---|---|---|---|
| `rms` | Valor eficaz de la ventana de senal DE | sqrt(mean(x^2)) | `rodamientos_module.features.ExtractorCaracteristicas.temporales` |
| `skewness` | Asimetria de la distribucion de amplitudes | scipy.stats.skew(x) | idem |
| `kurtosis` | Curtosis (Fisher) de la distribucion de amplitudes | scipy.stats.kurtosis(x) | idem |
| `factor_cresta` | Relacion entre el valor pico y el RMS | max(\|x\|) / rms | idem |
| `centroide_espectral` | Frecuencia media ponderada por la magnitud del espectro FFT | sum(f * \|X(f)\|) / sum(\|X(f)\|) | `rodamientos_module.features.ExtractorCaracteristicas.frecuenciales` |
| `archivo` | Identificador del registro experimental de origen (usado como grupo de validacion) | Nombre de archivo .mat | `rodamientos_module.dataset.CargadorCWRU` |
| `clase` | Condicion del rodamiento (variable objetivo) | Normal / Ball / Inner Race / Outer Race | idem |
