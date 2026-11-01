---
id: Yaswanth2026
title: "A Study on Forecasting of Crude Oil Prices Employing Optimized Deep Learning Models and LUBE Method."
authors: "Gollu Yaswanth; Hanumanthu Lohith; Mamuduri Jerusha; Sibarama Panigrahi"
year: 2026
doi: "10.1007/s10690-025-09554-2"
journal: "Asia-Pacific Financial Markets"
editorial_quality: "Q2 (Springer)"
status: "Exclude"
exclusion_reason: "E6: Predice el PRECIO del WTI (intervalos de predicción del precio en USD/barril), no la volatilidad realizada, ni la varianza condicional, ni un proxy de volatilidad."
problem_type: "Interval forecast"
horizon: "single-step"
proxy: "Not applicable (price target, not volatility)"
architectural_family: "Recurrent"
architectural_variant: "LSTM + LUBE"
market_asset: "WTI Crude Oil Spot Price"
---

# Resumen
El estudio aplica el método LUBE (Lower Upper Bound Estimation) con modelos de Deep Learning (LSTM, GRU, BiLSTM y CNN) para generar intervalos de predicción (PIs) del **precio** del crudo WTI en USD/barril. Usa una función de pérdida personalizada que combina pinball loss, PICP y PINAW. Evalúa en tres frecuencias (mensual, semanal, diaria) y cuatro niveles de confianza (0.9, 0.8, 0.7, 0.6), y compara contra métodos paramétricos basados en MSE + distribución gaussiana y MSE + distribución t-location scale. Reporta que el LSTM optimizado con LUBE es el mejor modelo en las tres frecuencias con PICP ≈ 100% y el menor PINAW. **El estudio se excluye del corpus porque el target es el precio, no la volatilidad.**

# 1. Método y Datos

**¿Qué se predice exactamente?**
Intervalos de predicción del **precio** del WTI (en USD/barril). **No se predice volatilidad, ni RV, ni varianza condicional, ni proxy de volatilidad.** El trabajo es de pronóstico probabilístico del nivel de precio. El título del paper lo declara explícitamente: “Forecasting of Crude Oil Prices”, no “volatility”.

**Fuente de los datos:**
- **WTI spot price (Dollars per Barrel).** Tres frecuencias:
  - Mensual: 01/01/1986 – 22/06/2022 (438 obs).
  - Semanal: 03/01/1986 – 01/07/2022 (1,905 obs).
  - Diaria: 02/01/1986 – 30/06/2022 (9,196 obs).
- **Data availability:** no se reporta explícitamente. Los precios de WTI son públicos (EIA).

**Partición Temporal:** split 70% train, 15% validation, 15% test. El texto dice “first 70%... next 15%... last 15%”, lo que sugiere orden cronológico. **No se proporcionan fechas de corte.** No hay rolling origin.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** **Hold-out estático, esquema no declarado.** El modelo se entrena una vez sobre el 70% inicial y se evalúa una vez sobre el 15% final. No hay reentrenamiento entre puntos. No se declara si la predicción es open-loop con actuales o recursiva. Dado que se usa una sliding window con ventana fija y el target es el siguiente valor de la serie, lo más probable es open-loop con actuales, pero **el paper no lo especifica**.

**Normalización:** MinMaxScaler a [0,1]. **No se especifica si el scaler se ajusta solo con el train set o con la serie completa.** El paper describe la normalización como paso previo al split, lo que sugiere que se aplica sobre la serie completa, pero no lo declara.

**Ingeniería de Características:** no hay descomposición ni selección de features. Se usa una sliding window sobre la serie de precios normalizada. No hay variables exógenas. No hay look-ahead por descomposición (no aplica), pero sí posible fuga por normalización global no declarada.

# 2. Arquitectura Final

Cuatro modelos DL, todos con 2 capas ocultas y 2 neuronas de salida (cota inferior y superior del intervalo):

**LSTM (mejor modelo reportado):**
- HL-1: 64 neuronas, activación ELU, inicializador glorot_uniform, sin dropout.
- HL-2: 32 neuronas, activación Sigmoid, inicializador glorot_uniform, **dropout = 1**.
- Output: 2 neuronas, activación Linear.
- Learning rate 0.001, batch 8, Adam.

**GRU:**
- HL-1: 64 neuronas, ELU, glorot_uniform, dropout 0.18.
- HL-2: 32 neuronas, Tanh, he_normal, sin dropout.
- Output: 2 neuronas, Linear.
- Learning rate 0.086, batch 8, Adam.

**BiLSTM:**
- HL-1: 128 neuronas, Tanh, glorot_normal, sin dropout.
- HL-2: 64 neuronas, ELU, he_uniform, dropout 0.084.
- Output: 2 neuronas, Linear.
- Learning rate 0.009, batch 8, Adam.

**CNN:**
- HL-1: 32 neuronas, ELU, he_normal, sin dropout.
- HL-2: 16 neuronas, Sigmoid, glorot_normal, dropout = 1.
- Output: 2 neuronas, Linear.
- Learning rate 0.0001, batch 8, Adam.

**Observación crítica sobre dropout = 1:** un dropout rate de 1.0 significa que el 100% de las neuronas de esa capa se desactivan durante el entrenamiento. Es una configuración sin sentido práctico (la capa deja de aportar información). El paper lo presenta como “mitigación de overfitting”, pero es un error de diseño o un typo (probablemente debió ser 0.1). Afecta a LSTM y a CNN, dos de los cuatro modelos.

**Función de pérdida LUBE:**
$$\text{Loss} = \text{lower\_loss} + \text{upper\_loss} + \max(0.9 - \text{PICP}, 0)^2$$
donde lower_loss y upper_loss son pinball losses con $q=0.05$. **El término de penalización está calibrado para un nivel de confianza fijo de 0.9.** Sin embargo, el modelo se evalúa en niveles 0.6, 0.7, 0.8 y 0.9. Para los niveles inferiores a 0.9, el término de penalización se anula ($\max(0.9 - \text{PICP}, 0) = 0$ si PICP ≥ 0.9) y el intervalo no está calibrado para la cobertura nominal. Reportar PICP ≈ 100% en todos los niveles sugiere que los intervalos son demasiado anchos, no que el modelo esté bien calibrado.

# 3. Resultados y Comparaciones

**Modelos comparados:**
- **Propuestos (LUBE):** LSTM + LUBE, GRU + LUBE, BiLSTM + LUBE, CNN + LUBE.
- **Benchmarks paramétricos:** MSE + distribución gaussiana; MSE + distribución t-location scale (ambos con los mismos 4 modelos DL).

**Métricas y Rendimiento:**
PICP, PINAW, ACE y AWD. LSTM + LUBE obtiene los mejores resultados: PICP ≈ 100% y el menor PINAW en las tres frecuencias. CNN queda en segundo lugar. BiLSTM es el peor en PINAW y AWD.

**Pruebas de Significancia:**
Usa el test de Friedman + Nemenyi post-hoc con 20 ejecuciones independientes. Reporta diferencias estadísticamente significativas para PINAW y AWD, pero equivalencia para PICP y ACE en varias comparaciones. El reporte de Friedman es incompleto: los valores de rango promedio son iguales (40.50) para todos los modelos en PICP y ACE, lo que sugiere que la prueba no discrimina. La conclusión de que “LSTM es el mejor” se basa en los ranks de PINAW y AWD, no en una prueba de superioridad global.

**Sesgo de Reporte:**
Alto. Cinco problemas:
1. **Target inválido para el corpus:** el estudio predice precio, no volatilidad. Es la razón de exclusión.
2. **Inconsistencias internas graves en las tablas:**
   - Las Tablas 2 (LUBE) y 3 (paramétrico gaussiano) reportan **valores idénticos** para todos los modelos y niveles de confianza (ej. LSTM 0.9: PICP=100, PINAW=2.807, ACE=10, AWD=7.423 en ambas). Esto sugiere un error de copia-pega.
   - La Tabla 11 (comparación LUBE vs paramétrico) reporta para el paramétrico gaussiano un ACE de −19.6875 para LSTM 0.9, lo cual **no coincide** con la Tabla 3 (que reporta ACE=10). En cambio, coincide con la Tabla 4 (t-location scale). Es decir, la comparación parece usar los valores de la distribución t-location scale disfrazados de gaussiana.
   - El texto afirma que “Gaussian and t-location scale provide almost similar PICP and ACE”, pero las Tablas 3 y 4 muestran valores muy distintos (PICP=100 vs 70.31 en LSTM 0.9).
3. **Función de pérdida mal calibrada:** el término de penalización está fijado en 0.9, pero se evalúa en 0.6–0.8. En esos niveles, el modelo no está optimizado para la cobertura nominal.
4. **Dropout = 1 en dos modelos:** configuración sin sentido que invalida la interpretación de los resultados de LSTM y CNN.
5. **Hiperparámetros no optimizados en este trabajo:** se reutilizan de un paper previo del mismo grupo (Purohit & Panigrahi, 2024) sin reoptimizar.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros de los modelos.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** no se declara explícitamente. Los precios de WTI son públicos (EIA).
- **Métricas de calibración:** reporta PICP, PINAW, ACE y AWD, pero la calibración está mal definida (ver Sección 2).
- **Limitaciones reportadas:** los autores no reportan limitaciones explícitas en la sección de conclusiones.

## 5.2 Hiperparámetros reportados (valores finales)
- **Arquitectura:** Tabla 1 con capas, neuronas, activaciones, inicializadores, dropout, learning rate, batch size y optimizador.
- **Función de pérdida:** LUBE (pinball + penalización PICP).
- **Dropout = 1** en HL-2 de LSTM y CNN (configuración inválida).
- **Hiperparámetros:** reutilizados de Purohit & Panigrahi (2024), no optimizados en este trabajo.

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **No hay búsqueda en este trabajo.** Se reutilizan hiperparámetros de DE-DL (Purohit & Panigrahi, 2024). |
| Optimizador | Adam |
| Batch size | 8 |
| Learning rate | LSTM: 0.001; GRU: 0.086; BiLSTM: 0.009; CNN: 0.0001 |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No.** Usan un conjunto de validación separado (15%) para monitorear el entrenamiento, pero no para búsqueda de hiperparámetros. Los hiperparámetros se toman de un trabajo previo.

**¿Sobre qué conjunto se evalúa el fitness?** **No aplica.** No hay búsqueda de hiperparámetros en este trabajo.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** La **LUBE loss** definida en la Sección 2:
$$\text{Loss} = \text{pinball}_q(y, \hat{y}_{\text{low}}) + \text{pinball}_{1-q}(y, \hat{y}_{\text{up}}) + \max(0.9 - \text{PICP}, 0)^2$$
donde $q = 0.05$. La función de pérdida está mal calibrada para niveles distintos de 0.9.

## 5.4 Limitaciones detectadas por el revisor
1. **Target inválido para el corpus:** predice precio, no volatilidad.
2. **Dropout = 1 en dos modelos:** configuración inválida.
3. **Función de pérdida mal calibrada:** penalización fijada en 0.9, evaluación en 0.6–0.8.
4. **Inconsistencias entre tablas:** Tablas 2 y 3 idénticas; Tabla 11 mezcla valores de t-location scale con gaussian.
5. **Hiperparámetros no optimizados en este trabajo.**
6. **Sin código.**
7. **Partición sin fechas de corte.**
8. **Normalización no declarada.**
9. **Esquema de predicción no declarado (open-loop vs recursivo).**

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **No de forma exacta.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Hiperparámetros de los modelos | Sí (Tabla 1) | Bajo: están reportados, aunque reutilizados de otro trabajo |
| Función de pérdida LUBE | Sí (Algorithm 1) | Medio: está definida, pero mal calibrada |
| Partición con fechas de corte | No | Alto: no se sabe exactamente dónde termina el train |
| Normalización (momento del ajuste) | No | Medio: afecta la escala del target |
| Esquema de predicción (open-loop vs recursivo) | No declarado | Alto: no se sabe si el error se propaga |
| Código | No | Alto: no hay referencia ejecutable |
| Semilla | No | Medio: los modelos DL son estocásticos |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Sí, con un riesgo de divergencia alto.** Los supuestos que habría que tomar:

1. **Normalización:** Min-Max ajustado solo con el train.
2. **Partición:** 70/15/15 cronológico con fechas inferidas.
3. **Esquema de predicción:** rolling origin con reentrenamiento, open-loop con actuales.
4. **Dropout:** corregir a 0.1 (el valor de 1 es un error).
5. **Función de pérdida:** calibrar el término de penalización al nivel de confianza que se evalúa.
6. **Semilla:** fijar una semilla para los modelos DL.

**Riesgo de divergencia del desempeño reportado:** **alto.** El PICP ≈ 100% está inflado por intervalos demasiado anchos, no porque el modelo esté bien calibrado. El dropout = 1 en LSTM y CNN invalida los resultados de esos dos modelos. Las inconsistencias entre tablas sugieren errores de reporte que no se pueden corregir sin el código.

**Riesgo de divergencia de la arquitectura:** **medio.** El pipeline (sliding window + LSTM/GRU/BiLSTM/CNN + LUBE) es reconstruible con supuestos estándar.

**Riesgo de divergencia del pipeline:** **medio.** Los pasos están claros, pero la partición y el esquema de predicción requieren supuestos.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, partición sin fechas de corte, normalización no declarada, esquema de predicción no declarado, dropout inválido, función de pérdida mal calibrada e inconsistencias entre tablas, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación sería una aproximación, no una réplica. Sin embargo, **este punto es irrelevante para el benchmark porque el estudio se excluye por target inválido (E6).**

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **No.** El estudio predice el **precio** del WTI, no la volatilidad, ni la RV, ni la varianza condicional, ni un proxy de volatilidad. El título, el abstract y la metodología son explícitos: “Forecasting of Crude Oil Prices”, “prediction intervals for crude oil prices (COP)”. Usa DL como componente central (LSTM, GRU, BiLSTM, CNN), pero el target es el precio.
- **C2. Reporte suficiente:** **Tal vez.** Reporta hiperparámetros completos (aunque reutilizados), función de pérdida y arquitectura. Sin código.
- **C3. Calidad editorial:** **Sí.** Q2 en JCR, Springer.
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica** (el protocolo es reconstruible con supuestos).
- **E6. Solo precio, sin volatilidad:** **Sí.** El estudio predice intervalos de precio, no volatilidad. Es la razón de exclusión.
- **E7. GARCH como contribución principal:** **No aplica**.

**Resultado:** **Excluido.** Motivo: **E6**.

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** precio, no volatilidad. → **excluyente (E6)**.
- **Partición:** 70/15/15 sin fechas de corte. → ambiguo.
- **Predicción:** hold-out (entrena una vez, evalúa una vez, sin rolling). Esquema de input no declarado. → alto.
- **Fuga:** sin descomposición (no aplica), pero normalización no declarada y función de pérdida mal calibrada. → moderado.
- **Reporte selectivo:** un solo split, una sola corrida por configuración, sin varianza. → alto.
- **Inferencia:** Friedman + Nemenyi con 20 ejecuciones, pero los ranks son idénticos en PICP y ACE (no discriminan). → moderado.
- **Comparación:** 4 modelos DL con LUBE vs los mismos 4 con MSE + Gauss/t-location. Sin benchmarks no-DL (ARIMA, GARCH). → moderado.
- **Reproducibilidad:** sin código, dropout inválido, inconsistencias entre tablas, hiperparámetros reutilizados. → parcial.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** WTI spot público (EIA). → **Sí**.
- **Reconstructibilidad del pipeline:** **sí, con supuestos.** Sliding window + LSTM/GRU/BiLSTM/CNN + LUBE. → **Sí**.
- **Viabilidad computacional:** **sí.** Los modelos son ligeros. → **Sí**.

**Decisión provisional:**
- **Entra al benchmark:** **No.**
- **Tipo de entrada:** **N/A (excluido).**
- **Correcciones necesarias:** N/A. El estudio se excluye por target inválido, no por defectos corregibles. Aun si se corrigieran los defectos metodológicos (dropout, calibración, tablas), el target seguiría siendo precio, no volatilidad.
- **Brecha principal:** N/A para el benchmark. El estudio se reporta en la revisión como **evidencia de que existe literatura de pronóstico probabilístico del precio que no aborda la volatilidad**, lo que refuerza la motivación del benchmark de enfocarse específicamente en volatilidad. No compite en la comparación de modelos.