---
id: Ouyang2024
title: "A novel integrated method for improving the forecasting accuracy of crude oil: ESMD-CFastICA-BiLSTM-Attention"
authors: "Zisheng Ouyang; Min Lu; Zhongzhe Ouyang; Xuewei Zhou; Ren Wang"
year: 2024
doi: "10.1016/j.eneco.2024.107851"
journal: "Energy Economics"
editorial_quality: "Q1 (ElSevier)"
status: "Include"
exclusion_reason: "None"
problem_type: "Point forecast"
horizon: "single-step"
proxy: "Conditional variance (GARCH) — for the volatility experiment"
architectural_family: "Hybrid"
architectural_variant: "ESMD-CFastICA-BiLSTM-Attention"
market_asset: "WTI Crude Oil (price) y WTI conditional volatility (GARCH)"
---

# Resumen
El estudio propone un modelo integrado (ESMD-CFastICA-BiLSTM-Attention) para el pronóstico del **precio** y de la **volatilidad condicional** del crudo WTI. El modelo utiliza la descomposición empírica de modo simétrico de punto extremo (ESMD) para descomponer las series, agrupa los componentes con K-means y extrae componentes independientes con FastICA (denominado CFastICA). Finalmente, estos componentes se introducen en un modelo de deep learning compuesto por una red Bidireccional LSTM (BiLSTM) y un mecanismo de atención (Attention) para predecir tanto el precio como la volatilidad. **El estudio se incluye porque aborda volatilidad (aunque condicional, no RV) y usa una partición temporal correcta, pero presenta un look-ahead bias confirmado por descomposición global y un bucle redundante GARCH→LSTM en el experimento de volatilidad.**

# 1. Método y Datos

**¿Qué se predice exactamente?**
El paper realiza **dos experimentos independientes** utilizando la misma arquitectura:
1. **Predicción de Precio:** target = precio diario del WTI (en USD/barril).
2. **Predicción de Volatilidad:** target = varianza condicional estimada mediante un **GARCH(1,1)** sobre los retornos logarítmicos diarios. **No usan Volatilidad Realizada (RV)** de alta frecuencia. La serie de volatilidad se construye como:
   $$\sigma_t^2 = \omega + \alpha_1 y_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$
   donde $y_t$ son los retornos logarítmicos diarios.

**Bucle redundante en el experimento de volatilidad:** el GARCH(1,1) ya produce una predicción de la varianza condicional en $t+1$. Los autores toman esa serie de varianzas predichas, la descomponen con ESMD y la pasan por BiLSTM para "predecirla". Es decir, **están usando Deep Learning para predecir lo que el GARCH ya predijo**. No están prediciendo la volatilidad real del mercado (RV), sino el ajuste de un modelo paramétrico. El LSTM no aporta valor predictivo independiente; solo suaviza y memoriza la salida del GARCH.

**Justificación del GARCH(1,1):** **NO se reporta** ningún análisis de sensibilidad o comparación (GARCH(1,2), EGARCH, GJR-GARCH) para justificar por qué el orden (1,1) es el óptimo. Tampoco muestran pruebas de diagnóstico de residuos (Ljung-Box sobre residuos al cuadrado) para confirmar que el modelo capturó toda la heterocedasticidad.

**Fuente de los datos:**
- **EIA (WTI).** Precios diarios del WTI. Periodo: 4 de abril de 1983 – 12 de diciembre de 2023 (10,218 observaciones diarias).
- **Data availability:** no se declara explícitamente. Los precios son públicos (EIA).

**Partición Temporal:** split estático 80/20. Train: abril 1984 – octubre 2015. Test: octubre 2015 – diciembre 2023. **No hay Rolling Origin ni Walk-forward.** El modelo se entrena una vez y se congela para predecir 8 años.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** **Hold-out estático, esquema de input no declarado.** El modelo se entrena una vez sobre el 80% inicial y se evalúa una vez sobre el 20% final. No hay reentrenamiento entre puntos. No se declara si la predicción es open-loop con actuales o recursiva.

**Normalización:** mencionan que los datos se normalizaron antes de la predicción, pero **no especifican** si el escalador se ajustó solo con el train set o con la serie completa.

**Ingeniería de Características (LOOK-AHEAD BIAS CONFIRMADO):**
- **ESMD + K-means + FastICA sobre la serie completa:** en la Sección 3.2 y la Figura 1, aplican ESMD, K-means y FastICA sobre la **serie completa (1983–2023)** antes de dividir en train/test. El algoritmo FastICA busca maximizar la no-gaussianidad global, por lo que los componentes independientes (ICs) calculados para el train set ya contienen información matemática del futuro (test set). Es el mismo patrón de fuga por descomposición global que en otros estudios del corpus.
- **Generación de la serie GARCH (AMBIGÜEDAD):** no se especifica si el GARCH(1,1) se calibró usando la serie completa (creando un leakage masivo) o solo el train set. Si se usó la serie completa, el GARCH "vio" la crisis de 2020 y la guerra de 2022 para calibrar sus parámetros.
- **Sin variables exógenas:** solo se usa la serie de precios (y la serie GARCH para el experimento de volatilidad).

# 2. Arquitectura Final (ESMD-CFastICA-BiLSTM-Attention)

El modelo tiene tres etapas principales:
1. **Descomposición (ESMD):** la serie (precio o volatilidad) se descompone en múltiples IMFs y un residuo. Para el precio: 10 IMFs. Para la volatilidad: 11 IMFs.
2. **Reconstrucción (CFastICA):**
   - Se aplica K-means a los IMFs para agruparlos por similitud (3 clústeres para precio, 5 para volatilidad).
   - Se aplica FastICA a los clústeres para extraer 3 Componentes Independientes (ICs) no gaussianos.
3. **Predicción (BiLSTM-Attention):**
   - Los 3 ICs se introducen en una red BiLSTM (que procesa la secuencia hacia adelante y hacia atrás).
   - El mecanismo de Attention asigna pesos a los estados ocultos para destacar la información más relevante.
   - La salida es una predicción puntual y una predicción por intervalos (usando una distribución log-normal).

**BiLSTM-Attention (arquitectura):**
- 1 capa BiLSTM, 1 capa Attention, 1 capa Dense de salida.
- 20 nodos por capa, activación `tanh`, 100 épocas, batch size 100, optimizador Adam.
- Grid Search para hiperparámetros.
- Predicción de intervalos: asumen distribución log-normal, calculan media y varianza usando la muestra completa.

# 3. Resultados y Comparaciones

**Modelos comparados (10 en total):**
- **Preprocesamiento:** EMD, EEMD (comparados contra ESMD).
- **Deep Learning:** LSTM puro, BiLSTM puro, SVR, Random Forest (RF).
- **Sin preprocesamiento:** S-BiAt (predicción directa sin ESMD ni CFastICA).
- **Ablación:** ES-BiAt (solo ESMD, sin CFastICA), EE-CF-BiAt (EEMD + CFastICA + BiAt), E-CF-BiAt (EMD + CFastICA + BiAt), ES-CF-BiLSTM, ES-CF-LSTM, ES-CF-SVR, ES-CF-RF.

**Métricas de Evaluación:**
- **Punto:** MAE, RMSE, $R^2$, $D_{stat}$ (direccional).
- **Intervalos:** PICP, PIAW, AWD.
- **Estadística:** Diebold-Mariano (DM) con 4 funciones de pérdida (MAD, MSE, MAPE, Poly).

**Ganador y Rendimiento:**
El modelo **ES-CF-BiAt** es el ganador. Resultados para el precio:
- MAE = 1.2779, RMSE = 2.0835, $R^2$ = 0.9877, $D_{stat}$ = 0.7204.
Resultados para la volatilidad:
- MAE = 0.0017, RMSE = 0.0059, $R^2$ = 0.9252, $D_{stat}$ = 0.5578.

En comparación con su contraparte sin descomposición (S-BiAt), las mejoras en MAE y RMSE para la volatilidad superan el 60% y 140%, respectivamente.

**Pruebas de Significancia:**
Diebold-Mariano (DM) con 4 funciones de pérdida. Los resultados muestran que las mejoras del modelo propuesto son estadísticamente significativas al 1% frente a casi todos los benchmarks, tanto para el precio como para la volatilidad. También evalúan la cobertura de los intervalos (PICP, PIAW, AWD).

**Sesgo de Reporte:**
Alto. Cuatro problemas:
1. **Look-ahead bias por descomposición global:** ESMD, K-means y FastICA se aplican sobre la serie completa antes del split. Los ICs del train contienen información del test. La comparación entre modelos con y sin CFastICA no es justa.
2. **Bucle redundante GARCH→LSTM:** el experimento de volatilidad usa Deep Learning para predecir la salida paramétrica del GARCH, no la volatilidad real del mercado. El LSTM no aporta valor predictivo independiente.
3. **Target no comparable:** el target de volatilidad es la varianza condicional del GARCH, no la RV de alta frecuencia. Los resultados no son comparables con estudios que usan RV diaria.
4. **Falta de validación del GARCH:** no comparan el GARCH(1,1) contra otras especificaciones (EGARCH, GJR-GARCH) ni reportan pruebas de diagnóstico de residuos.
5. **Entrenamiento estático:** el modelo se entrena una vez (2015) y se congela para predecir 8 años (2015–2023). Esto subestima el desempeño real en un entorno de producción.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros de la red.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** no se declara explícitamente. Los precios son públicos (EIA).
- **Métricas de calibración:** no reporta ECE, pero sí evalúa la cobertura de los intervalos de predicción (PICP, PIAW, AWD).
- **Limitaciones reportadas:** los autores admiten que no compararon diferentes métodos de reconstrucción, que no usan modelos de interpretabilidad, y que el modelo es univariado (no incorpora factores macroeconómicos).

## 5.2 Hiperparámetros reportados (valores finales)
- **ESMD:** 10 IMFs para precio, 11 IMFs para volatilidad. Criterio de parada: $\varepsilon = 0.0001\sigma_0$.
- **K-means:** 3 clústeres para precio, 5 para volatilidad (seleccionados por elbow method con WCSS, SSE, SC).
- **FastICA:** 3 componentes independientes (ICs), función de contraste $g(y) = \tanh(by)$ con $b=1$.
- **BiLSTM-Attention:** 20 nodos por capa, activación `tanh`, 100 épocas, batch size 100, Adam, Grid Search.
- **BiLSTM (comparación):** mismos hiperparámetros que BiLSTM-Attention.
- **LSTM (comparación):** mismos hiperparámetros que BiLSTM-Attention.
- **SVR:** kernel RBF, gamma = 'auto'.
- **RF:** 100 estimadores, max_features = 'auto', max_depth = 3, min_samples_split = 2, min_samples_leaf = 1.
- **GARCH(1,1):** parámetros estimados por máxima verosimilitud. **Los valores finales no se reportan.**
- **Función de pérdida:** MSE (para BiLSTM-Attention).

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | Grid Search |
| Espacio de búsqueda | nodes, activation, epochs, batch_size, optimizer |
| Valores finales | 20 nodos, tanh, 100 épocas, batch=100, Adam |
| Optimizador | Adam |
| Learning rate | NO reportado |
| Función de pérdida | MSE |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No se menciona.** El Grid Search se realiza presumiblemente sobre el train set, pero no se especifica si hay conjunto de validación o validación cruzada.

**¿Sobre qué conjunto se evalúa el fitness?** **NO reportado.** No se aclara si el fitness del Grid Search se calcula sobre el train set, sobre un subconjunto del train, o sobre la serie completa.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **MSE** (función de pérdida del BiLSTM-Attention). No se reporta si se usa validación o early stopping.

### 5.3.4 GARCH(1,1)
El modelo GARCH(1,1) se estima por máxima verosimilitud. Los parámetros finales no se reportan. No se reportan pruebas de diagnóstico de residuos.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Los hiperparámetros de la red están reportados, pero faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura del BiLSTM-Attention | Sí (20 nodos, tanh, 100 épocas) | Bajo: está reportada |
| Función de pérdida | Sí (MSE) | Bajo: está declarada |
| Parámetros del GARCH | No | Alto: no se puede replicar la serie de volatilidad |
| Diagnóstico de residuos del GARCH | No | Medio: no se valida la adecuación del modelo |
| Normalización (momento del ajuste) | No | Medio: afecta la escala del target |
| Esquema de predicción (open-loop vs recursivo) | No declarado | Medio: no se sabe si el error se propaga |
| Código | No | Alto: no hay referencia ejecutable |
| Semilla | No | Medio: los modelos DL son estocásticos |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Parcialmente, con un riesgo de divergencia alto.** Los supuestos que habría que tomar:
1. **GARCH(1,1):** estimar sobre el train set (no sobre la serie completa) y reportar los parámetros.
2. **Normalización:** Min-Max o Z-score ajustado solo con train.
3. **Esquema de predicción:** open-loop con actuales.
4. **Semilla:** fijar una semilla para la red.
5. **Corregir el look-ahead bias:** aplicar ESMD/K-means/FastICA de forma *rolling* o solo sobre el train.

**Riesgo de divergencia del desempeño reportado:** **alto.** El desempeño reportado está inflado por el look-ahead bias por descomposición global. Sin la fuga, el desempeño sería peor. Además, el experimento de volatilidad tiene un bucle redundante que no aporta valor predictivo real.

**Riesgo de divergencia de la arquitectura:** **medio.** La arquitectura está descrita (ESMD → CFastICA → BiLSTM-Attention), pero los detalles del GARCH no están reportados.

**Riesgo de divergencia del pipeline:** **alto.** El pipeline incluye descomposición global (fuga), GARCH con calibración ambigua, y entrenamiento estático.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, look-ahead bias por descomposición global, bucle redundante GARCH→LSTM, parámetros del GARCH no reportados, normalización no declarada, esquema de predicción no declarado y ausencia de código, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no una réplica exacta. El benchmark debe tratar la versión corregida como una **variante** del modelo original, no como una réplica exacta, y reportar la brecha entre ambos desempeños como evidencia del impacto de las carencias de reporte.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad del crudo con BiLSTM-Attention como componente central.
- **C2. Reporte suficiente:** **Parcial.** Reporta hiperparámetros de la red, pero no los parámetros del GARCH, ni la normalización, ni el esquema de predicción. Sin código.
- **C3. Calidad editorial:** **Sí.** Q1 en JCR (Energy Economics).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El protocolo es reconstruible con supuestos (partición cronológica, hiperparámetros de la red reportados). La fuga por descomposición es un problema de validez, no de reconstruibilidad.
- **E6. Solo precio, sin volatilidad:** **No aplica.** Predice volatilidad condicional (GARCH), no solo precio.
- **E7. GARCH como contribución principal:** **No aplica.** GARCH es una etapa del pipeline, no la contribución principal.

**Resultado:** **Incluido.**

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** varianza condicional del GARCH (no RV). → alto.
- **Partición:** split estático 80/20, sin rolling origin. → alto.
- **Descomposición:** ESMD + K-means + FastICA aplicados sobre la serie completa antes del split. → **look-ahead bias confirmado (alto)**.
- **Bucle redundante GARCH→LSTM:** el LSTM se entrena para replicar la salida del GARCH. → **alto**.
- **Predicción:** hold-out (entrena una vez, evalúa una vez, sin rolling). Esquema de input no declarado. → alto.
- **Reporte selectivo:** reporta DM con 4 funciones de pérdida. → bajo.
- **Inferencia:** DM con 4 funciones de pérdida. → bajo.
- **Comparación:** 10 modelos, con ablation study. Pero la comparación con/sin CFastICA no es justa (fuga). → alto.
- **Reproducibilidad:** sin código, parámetros del GARCH no reportados, normalización no declarada. → parcial.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** Precios públicos (EIA). → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** La arquitectura está reportada, pero faltan parámetros del GARCH, normalización y esquema de predicción. → **Parcial**.
- **Viabilidad computacional:** **sí.** Los modelos son ligeros. → **Sí**.

**Decisión provisional:**
- **Entra al benchmark:** sí.
- **Tipo de entrada:** **con corrección**.
- **Correcciones necesarias:**
    - **Corregir el look-ahead bias por descomposición global:** aplicar ESMD, K-means y FastICA de forma *rolling* (recalcular en cada origen usando solo datos pasados) o solo sobre el train set. Sin esta corrección, el desempeño reportado no es comparable con el de los modelos que no tienen fuga.
    - **Reemplazar el target de volatilidad:** usar RV de alta frecuencia (5 minutos) en lugar de la varianza condicional del GARCH. El bucle GARCH→LSTM no aporta valor predictivo independiente.
    - **Reportar los parámetros estimados del GARCH(1,1)** (ω, α, β) o, en su defecto, declarar que se estiman solo con el train set.
    - **Añadir pruebas de diagnóstico de residuos del GARCH** (Ljung-Box sobre residuos al cuadrado) para validar la adecuación del modelo.
    - **Declarar la normalización** (si la hay) y el momento del ajuste (solo train, o serie completa).
    - **Declarar el esquema de predicción** (open-loop con actuales o recursivo) y aplicarlo igual a todos los modelos.
    - **Fijar semilla** para los modelos DL.
    - **Añadir QLIKE** como métrica de evaluación, además de MAE, RMSE y $R^2$. QLIKE es estándar en volatilidad y penaliza de forma más adecuada los errores en volatilidades altas.
    - **Añadir un reentrenamiento rolling** (o al menos declarar que el modelo es hold-out y discutir el impacto en el desempeño real).
    - **Comparar el GARCH(1,1) contra otras especificaciones** (EGARCH, GJR-GARCH) para justificar la elección.
    - **Garantizar la equidad de inputs en la comparación:** los modelos con y sin CFastICA deben recibir los mismos inputs (o declarar la diferencia).
- **Brecha principal:** el estudio tiene un target de volatilidad (aunque condicional, no RV) y un esquema de comparación con ablation study. Las limitaciones principales son: (i) look-ahead bias por descomposición global (ESMD/K-means/FastICA sobre la serie completa); (ii) bucle redundante GARCH→LSTM en el experimento de volatilidad; (iii) target no comparable (varianza condicional, no RV); (iv) falta de validación del GARCH (no comparan especificaciones, no reportan diagnóstico de residuos); (v) entrenamiento estático sin rolling; (vi) ausencia de QLIKE; (vii) normalización no declarada; (viii) parámetros del GARCH no reportados. Aporta la familia `Hybrid` con un esquema de descomposición + reconstrucción + BiLSTM-Attention que el benchmark debería replicar con las correcciones declaradas. La brecha entre el desempeño original (con fuga) y el corregido será un hallazgo interesante.