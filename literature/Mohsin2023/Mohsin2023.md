---
id: Mohsin2023
title: "A novel deep-learning technique for forecasting oil price volatility using historical prices of five precious metals in context of green financing – A comparison of deep learning, machine learning, and statistical models."
authors: "Muhammad Mohsin; Fouad Jamaani"
year: 2023
doi: "10.1016/j.resourpol.2023.104216"
journal: "Resources Policy"
editorial_quality: "Q1 (Elsevier)"
status: "Exclude"
exclusion_reason: "E6: El modelo predice el PRECIO del crudo Brent (en USD/barril), no la Volatilidad Realizada (RV) ni la varianza condicional."
problem_type: "Point forecast"
horizon: "single-step"
proxy: "Not applicable (price target, not volatility)"
architectural_family: "Convolutional"
architectural_variant: "GroupNet (CNN con group blocks)"
market_asset: "Brent Crude Oil"
---

# Resumen
El estudio propone una novedosa arquitectura de red neuronal convolucional (CNN) llamada GroupNet para pronosticar el **precio** del crudo Brent utilizando como variables predictoras los precios históricos de cinco metales preciosos (Oro, Plata, Platino, Paladio y Rodio) en el contexto de finanzas verdes. Comparan el rendimiento de su modelo contra 7 modelos de Machine Learning y 3 modelos estadísticos tradicionales. **El estudio se excluye del benchmark porque su variable objetivo es el precio del crudo, no la volatilidad.** El título usa la palabra "volatility" de forma confusa, pero la Sección 4.1 declara explícitamente que la variable respuesta es el precio del Brent.

# 1. Método y Datos

**¿Qué se predice exactamente?**
El **precio del crudo Brent** (en USD/barril). A pesar del título, los autores declaran en la Sección 4.1: *"We use Brent Crude oil price as our response variable"*. Las métricas de evaluación (MAE, RMSE, MAPE) son para precios, no para varianza o riesgo. **No se predice volatilidad, ni RV, ni varianza condicional, ni proxy de volatilidad.** Es un caso de **mislabeling** similar a Wu2024 y Wang2020.

**Fuente de los datos:**
- **UCI Machine Learning Repository.** Periodo: 12 de enero de 1990 – 31 de enero de 2020. Los datos incluyen precios históricos de 5 metales preciosos (Oro, Plata, Platino, Paladio, Rodio) y Brent crude oil, entre otras variables.
- **Data availability:** "Data will be made available on request". Los precios son públicos (UCI).

**Partición Temporal:** **El texto indica que *"The dataset is arbitrarily split into training and testing parts"*.** Luego mencionan que usan *"rolling window cross-validation"* para reducir el sesgo, pero **no detallan** cómo aplicaron la ventana móvil ni si la división "arbitraria" implicó aleatorización (shuffle). Si la división fue aleatoria, hay data leakage. Si fue rolling, el diseño es correcto pero no está documentado.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** **No declarado con precisión.** El texto menciona rolling window cross-validation, pero no especifica si el modelo se reentrena en cada ventana o si se entrena una vez y se evalúa sobre ventanas móviles. Tampoco se declara si la predicción es open-loop con actuales o recursiva.

**Normalización:** aplicaron suavizado de datos (smoothing) y selección de características (CfsSubsetEval en WEKA), pero **no reportan** el método de escalado.

**Ingeniería de Características:**
- **Selección de variables (CfsSubsetEval):** aplicaron una técnica de selección de características basada en correlación (CfsSubsetEval en WEKA) para reducir la dimensionalidad. Los 5 metales preciosos se usan como predictores.
- **Suavizado de datos:** aplicaron smoothing para reducir el ruido, pero no se detalla el método.
- **Sin descomposición:** no hay EMD, VMD ni filtros no causales.

**Riesgo de data leakage:** la mención de una división "arbitraria" del dataset es una bandera roja para series de tiempo. Si se aplicó shuffle antes del split, hay fuga de información del futuro al train.

# 2. Arquitectura Final (GroupNet)

El modelo propuesto es una red neuronal convolucional (CNN) con una arquitectura novedosa:
1. **Group Block:** un nuevo tipo de bloque convolucional que divide los kernels en grupos para reducir el número de parámetros y mejorar la eficiencia computacional (inspirado en ShuffleNet e IGCNets).
2. **GroupNet:** la arquitectura completa de la CNN, que consta de varias capas de Group Blocks con capas de Max-Pooling y Dropout (tasa de 0.5).
3. **Capa de Regresión:** en lugar de una capa de clasificación con Softmax, se utiliza una capa densa con activación lineal y una sola neurona para predecir el precio (regresión).
4. **Funciones de Pérdida:** se introducen dos funciones de pérdida personalizadas (Focal Loss y Correlated Loss) para mejorar la capacidad de aprendizaje del modelo.
   $$\text{FL}(p) = -(1-p)^\gamma \log p$$
   $$\text{CE}(p, q) = -q \log p$$
   donde $\gamma$ es el parámetro de enfoque (óptimo = 2.0).

**Hiperparámetros (valores finales):**
- Épocas: 20 (óptimo).
- Learning rate: 0.002 (óptimo).
- Tamaño del grupo (group size): 128 (óptimo).
- Tamaño del kernel: 1×3 (óptimo).
- Dropout: 0.5 (óptimo).
- Función de activación: `tanh` (óptima).
- Focal loss parameter (gamma): 2.0 (óptimo).
- Optimizador: Adam.

# 3. Resultados y Comparaciones

**Modelos comparados (11 en total):**
- **Machine Learning (7):** GPR (Gaussian Processes), MLR (Multiple Linear Regression), MLP (Multi-Layer Perceptron), Lazy LWL, Decision Table (DT), M5Rules, AMT (Alternating Model Tree).
- **Estadísticos (3):** LAD (Least Absolute Deviation), OLS (Ordinary Least Squares), WLS (Weighted Least Squares).
- **Propuesto:** GroupNet (CNN).

**Métricas de Evaluación:**
MAE, MAPE, RMSE, RRSE (%), RAE (%). Se asigna un sistema de puntuación (ranking) que otorga 11 puntos al mejor modelo y 1 al peor en cada métrica. El score total es la suma de los puntos en las 5 métricas.

**Ganador y Rendimiento:**
El modelo **GroupNet (CNN)** es el ganador. En el set de prueba (Tabla 6):
- MAE = 0.3245
- MAPE = 2.4522
- RMSE = 0.4642
- RRSE = 4.2331%
- RAE = 2.7812%
- Score total = 40/55

El segundo mejor es DT (Decision Table) con un score de 35/55. El peor es GPR con un score de 5/55.

**Pruebas de Significancia:**
**NO se reportan pruebas de significancia estadística** (ni Diebold-Mariano, ni MCS, ni DeLong). La comparación se basa únicamente en un sistema de ranking que asigna puntos a cada modelo según su desempeño en 5 métricas. No se puede afirmar que las diferencias sean estadísticamente significativas.

**Sesgo de Reporte:**
Alto. Cinco problemas:
1. **Target inválido para el corpus:** predice precio, no volatilidad. El título usa "volatility" de forma confusa. Es la razón de exclusión.
2. **División "arbitraria" del dataset:** el texto indica que el dataset se divide "arbitrariamente", lo que sugiere aleatorización (shuffle) y, por tanto, data leakage. Aunque mencionan rolling window cross-validation, no detallan cómo se aplicó.
3. **Sin pruebas de significancia:** no hay DM, MCS ni DeLong. La comparación se basa en un ranking subjetivo.
4. **Ausencia de QLIKE:** la métrica estándar para volatilidad es QLIKE. MAE, MAPE, RMSE, RRSE y RAE no son adecuadas para evaluar pronósticos de volatilidad.
5. **Riesgo de sobreajuste no evaluado:** no se reporta la brecha train-test, ni curvas de aprendizaje, ni validación cruzada detallada.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros del CNN.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** reporta el tiempo en la Tabla 8. El CNN propuesto tarda 1.8 segundos en entrenar, siendo el tercero más lento (el más rápido fue OLS con 0.39 s y el más lento GPR con 2.1 s).
- **Hardware:** Intel Core i5-2410M CPU @ 2.70 GHz, 8 GB RAM, 500 GB HDD.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** "Data will be made available on request". Los precios son públicos (UCI).
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast sobre precio).
- **Limitaciones reportadas:** los autores mencionan que futuras investigaciones podrían usar Transfer Learning, Adversarial Learning y algoritmos de optimización como SSA o HHO.

## 5.2 Hiperparámetros reportados (valores finales)
- **CNN (GroupNet):** epochs=20, learning rate=0.002, group size=128, kernel size=1×3, dropout=0.5, activation=tanh, focal loss gamma=2.0, optimizer=Adam.
- **Feature selection:** CfsSubsetEval (WEKA).
- **Smoothing:** aplicado, pero sin detalle del método.
- **Función de pérdida:** Focal Loss + Correlated Loss.
- **Normalización:** no reportada.
- **Batch size:** no reportado.

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | Trial and error (gráficas de sensibilidad) |
| Espacio de búsqueda | Epochs (1-100), LR (0.0001-0.05), group size (16-256), kernel (varios), dropout (varios), activation (6 opciones), gamma (0-5) |
| Valores finales | epochs=20, LR=0.002, group=128, kernel=1×3, dropout=0.5, tanh, gamma=2.0 |
| Optimizador | Adam |
| Learning rate | 0.002 |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **Sí, mencionan rolling window cross-validation**, pero no detallan cómo se aplicó. El texto también dice que el dataset se divide "arbitrariamente", lo que es contradictorio con un esquema rolling.

**¿Sobre qué conjunto se evalúa el fitness?** **NO reportado.** No se aclara si el fitness se calcula sobre el train, sobre un validation set o sobre la serie completa.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **Focal Loss + Correlated Loss.** El modelo minimiza una combinación de ambas funciones de pérdida. No se reporta si se usa validación o early stopping.

### 5.3.4 Feature selection
Se aplicó CfsSubsetEval (WEKA) para seleccionar los predictores. No se reporta si la selección se hizo solo con el train set o con la serie completa.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Los hiperparámetros del CNN están reportados, pero faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura del GroupNet | Parcial (group blocks, max-pooling, dropout) | Medio: falta el número de capas y filtros |
| Función de pérdida | Sí (Focal Loss + Correlated Loss) | Bajo: está declarada |
| Hiperparámetros del CNN | Sí (epochs, LR, group, kernel, dropout) | Bajo: están reportados |
| Normalización | No | Medio: afecta la escala del target |
| Batch size | No | Medio: afecta el entrenamiento |
| Partición Train/Test | Ambiguo ("arbitrarily split") | Alto: no se sabe si hay shuffle |
| Esquema de predicción (open-loop vs recursivo) | No declarado | Medio: no se sabe si el error se propaga |
| Código | No | Alto: no hay referencia ejecutable |
| Semilla | No | Medio: los modelos DL son estocásticos |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Parcialmente, con un riesgo de divergencia alto.** Los supuestos que habría que tomar:
1. **Partición:** asumir split cronológico (no shuffle) y declarar que el texto es ambiguo.
2. **Normalización:** Min-Max o Z-score ajustado solo con train.
3. **Batch size:** asumir un valor típico (e.g., 32 o 64).
4. **Esquema de predicción:** open-loop con actuales.
5. **Semilla:** fijar una semilla para la red.

**Riesgo de divergencia del desempeño reportado:** **alto.** El desempeño reportado está potencialmente inflado por la división "arbitraria" del dataset. Sin claridad sobre la partición, no se puede saber si hay fuga. Además, el target es precio, no volatilidad, lo que lo excluye del corpus de todas formas.

**Riesgo de divergencia de la arquitectura:** **medio.** La arquitectura está descrita a grandes rasgos, pero faltan detalles del GroupNet (número de capas, filtros por capa).

**Riesgo de divergencia del pipeline:** **alto.** La partición y la normalización requieren supuestos.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, partición ambigua ("arbitrarily split"), normalización no declarada, batch size no reportado, arquitectura del GroupNet parcialmente descrita, ausencia de código y ausencia de semilla, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación sería una aproximación, no una réplica. Sin embargo, **este punto es irrelevante para el benchmark porque el estudio se excluye por target inválido (E6).**

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **No.** El estudio predice el **precio** del crudo Brent, no la volatilidad, ni la RV, ni la varianza condicional, ni un proxy de volatilidad. El título usa "volatility" pero la Sección 4.1 declara que el target es el precio.
- **C2. Reporte suficiente:** **Parcial.** Reporta hiperparámetros del CNN, pero no la normalización, el batch size, la partición exacta ni la arquitectura completa del GroupNet. Sin código.
- **C3. Calidad editorial:** **Sí.** Q1 en JCR, Elsevier.
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **Parcial.** El protocolo es reconstruible con supuestos, pero la partición "arbitraria" y la normalización no declarada introducen incertidumbre.
- **E6. Solo precio, sin volatilidad:** **Sí.** El estudio predice precios, no volatilidad. Es la razón principal de exclusión.
- **E7. GARCH como contribución principal:** No aplica.

**Resultado:** **Excluido.** Motivo: **E6**.

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** precio, no volatilidad. El título usa "volatility" de forma confusa. → **excluyente (E6)**.
- **Partición:** "arbitrarily split" sin detalle. Riesgo de data leakage si hubo shuffle. → alto.
- **Predicción:** rolling window cross-validation mencionada pero no detallada. → moderado.
- **Fuga:** posible por partición arbitraria. → alto.
- **Reporte selectivo:** no hay pruebas de significancia; solo ranking. → alto.
- **Inferencia:** sin pruebas de significancia (ni DM, ni MCS, ni DeLong). → alto.
- **Comparación:** 11 modelos, pero la comparación es por ranking subjetivo, no por tests formales. → alto.
- **Reproducibilidad:** sin código, sin semilla, partición ambigua, normalización no declarada. → parcial.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** Precios públicos (UCI). → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** La arquitectura está descrita a grandes rasgos, pero faltan detalles del GroupNet y la partición es ambigua. → **Parcial**.
- **Viabilidad computacional:** **sí.** El CNN es ligero (1.8 s de entrenamiento). → **Sí**.

**Decisión provisional:**
- **Entra al benchmark:** **No.**
- **Tipo de entrada:** **N/A (excluido por E6).**
- **Correcciones necesarias:**
    - **Reemplazar el target de precio por volatilidad realizada (RV)** o varianza condicional. Sin este cambio, el estudio no es comparable con el resto del corpus.
    - **Declarar la partición exacta:** especificar si el split es cronológico (no shuffle) y cómo se aplicó el rolling window cross-validation.
    - **Declarar la normalización** (si la hay) y el momento del ajuste (solo train, o serie completa).
    - **Reportar el batch size.**
    - **Detallar la arquitectura del GroupNet:** número de capas, filtros por capa, tamaño de los group blocks.
    - **Añadir pruebas de significancia estadística** (DM, MCS) para comparar los modelos, no solo un ranking subjetivo.
    - **Añadir QLIKE** como métrica de evaluación, además de MAE, MAPE, RMSE, RRSE y RAE.
    - **Fijar semilla** para los modelos DL.
    - **Liberar el código** o, en su defecto, reportar los valores finales de los hiperparámetros y la semilla para permitir la reimplementación exacta.
- **Brecha principal:** N/A para el benchmark. El estudio se reporta en la revisión como **evidencia de que existe una línea de investigación de price forecasting con metales preciosos como predictores que, aunque usa Deep Learning (CNN), no aborda el target de volatilidad y presenta una partición ambigua del dataset que impide descartar data leakage**. Las limitaciones específicas son: target de precio (E6), partición "arbitrarily split" (riesgo de fuga), ausencia de pruebas de significancia, ausencia de QLIKE, y falta de detalle en la arquitectura y normalización. No compite en la comparación de modelos.