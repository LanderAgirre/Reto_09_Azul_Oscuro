# Reto_09_Azul_Oscuro 🚀

Este repositorio contiene el código, datos y notebooks usados para el reto de selección y evaluación de carteras, limpieza/preprocesamiento de precios y experimentos de Data Science/ML (incluyendo análisis Monte Carlo y experimentos con redes GRU).

IMPORTANTE ⚠️ — Algunos pasos son computacionalmente muy costosos (Montecarlo, MLFLOW). Por defecto debes ejecutar SOLO las secciones 1, 2 y 4 (preprocesamiento, selección de cartera y pasos de Data Science ligeros). Las secciones pesadas están documentadas pero marcadas para ejecución opcional y bajo tu responsabilidad.

## Estructura principal
- `1.Procesamiento.ipynb` - Notebook principal de preprocesamiento y cálculo de métricas (splits, imputación, ajuste de series, visualizaciones).
- `2.Selección_Cartera.ipynb` - Notebook de generación y evaluación de carteras (métricas, correlación, selección top390, visualizaciones).
- `3.Montecarlo.ipynb` - NO EJECUTAR POR COSTE COMPUTACIONAL. 
- `4.DataScience.ipynb` - Esta sección agrupa los pasos de análisis de series temporales, visualización y modelado no costoso.
- `MLFLOW.ipynb` - NO EJECUTAR POR COSTE COMPUTACIONAL
- `5.Red.ipynb` - Experimentos de predicción de series temporales
- `Funciones.py` - Módulos auxiliares reutilizables.
---

## 1. Preprocesamiento

Propósito: limpiar los precios, imputar valores faltantes con datos de mercado reales (yfinance), detectar y ajustar splits, y generar `df_ajustado` listo para análisis.

Qué contiene / pasos importantes:
- Pivotar a formato `fecha x símbolo`.
- Imputación por yfinance (primero `Adj Close` o `Close` según la fuente y el ajuste); el notebook incluye fallbacks por símbolo.
- Detección de splits con `yf.Ticker(...).splits` y ajuste manual (multiplicar/dividir según fechas).
- Export: `Datos/Transformados/limpio.csv`.

Ejecutar: abre `1.Procesamiento.ipynb` y ejecuta las primeras celdas hasta que `df_ajustado` esté generado. Verifica la salida de la celda que imprime misses y splits.

Aviso: la imputación por yfinance hace múltiples llamadas a la API; ten en cuenta los límites y la conectividad. Si fallan llamadas, el notebook imprime los errores y continúa.

---

## 2. Selección de cartera 🧾📈

Propósito: generar combinaciones de carteras (tríos por defecto), calcular métricas (rentabilidad anualizada, volatilidad anualizada, Sharpe, drawdown, correlaciones parciales) y seleccionar las carteras top (por score y/o Sharpe).

Qué contiene:
- `cartera_simple` y `generar_carteras` — generadores de combinaciones y métricas.
- Filtrado `top390` — selección de las mejores 390 carteras según criterios.
- Función `evaluar_carteras_desde_top` — evalúa carteras ya listadas en `top390` (útil si ya dispones del archivo `top390` que contiene la columna `cartera`).

Ejecutar: abre `2.Selección_Cartera.ipynb`. Para reproducir el flujo ligero ejecuta las celdas que:
1. Cargan `df_limpio` / `df_ajustado`.
2. Ejecutan `cartera_simple` y generan `top390`.
3. Ejecutan `generar_carteras` o `evaluar_carteras_desde_top` según necesites.

Salida esperada: `carteras` DataFrame con columnas clave — `cartera`, `rentabilidad_anualizada`, `volatilidad_anualizada`, `rentabilidad_acumulada`, `sharpe`, `drawdown_max`, `corr_mas_alta`, `corr_media_tercero`, `score_corr`.

Nota: si ya tienes `top390` preprocesado (con columnas de métricas), usa `evaluar_carteras_desde_top(df_ajustado, top390)` para re-evaluar con precios actuales.

---

## 3. Montecarlo — hasta comparación (🔒 NO EJECUTAR POR COSTE COMPUTACIONAL)

Descripción: aquí se realizan simulaciones Monte Carlo extensas para comparar distribuciones de retornos y riesgos entre carteras.

IMPORTANTE ⛔: Estas celdas son temporales y computacionalmente intensivas. No las ejecutes en tu máquina local. Para eso estan guardados los modelos en la carpeta de Modelos. 

---

## 4. Data Science 🧠

Esta sección agrupa los pasos de análisis, visualización y modelado no costoso:

Ejecución recomendada: puedes ejecutar esto tras completar 1 y 2. Los notebooks ya incluyen las celdas para generar estas gráficas.

---

## MLFLOW 📋 NO EJECUTAR POR COSTOSO Y PORQUE SE HA GUARDADO EL PROCESO PERO ES PARA LA ASIGNATURA DE HERRAMIENTAS

Se documenta cómo registrar experimentos con MLFLOW. Recomendación: NO EJECUTAR
- No ejecutar por defecto. Usa MLFLOW si quieres reproducir y trackear experimentos ML (necesitas servidor MLFLOW o `mlflow ui` local y almacenamiento de artefactos). 

## 5. Red neuronal: GRU 🧩

En los experimentos de predicción de series temporales se eligió una GRU (Gated Recurrent Unit) por su balance entre capacidad y velocidad frente a LSTM.

Nota: entrenar redes puede ser costoso. Si lo haces localmente, reduce epochs / batch_size para pruebas rápidas.

---

## 3.1 Montecarlo — ENTREGADO EN DÍAS ANTERIORES

Después de seleccionar las carteras a comparar, se corre un Montecarlo más exhaustivo (post-comparación) para validar robustez.

---

## Recomendación de ejecución (rápida)
1. Ejecuta solo el preprocesamiento y la imputación: `1.Procesamiento.ipynb` (celdas iniciales hasta `df_ajustado`).
2. Ejecuta selección de carteras: `2.Selección_Cartera.ipynb` (hasta generación y evaluación de `top390`).
3. Ejecuta los pasos ligeros de Data Science visuales y resúmenes.

Evitar "Montecarlo" y "mlflow"

---

## Archivos de salida y dónde buscar resultados

- `Datos/Transformados/limpio.csv` — precios ajustados/imputados.
- `Datos/Transformados/close.csv` — dataset de entrada limpio.

