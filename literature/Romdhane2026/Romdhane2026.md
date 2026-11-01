---
id: Romdhane2026
title: "A Hybrid HAR-LSTM-GARCH Model for Forecasting Volatility in Energy Markets."
authors: "Wiem Ben Romdhane; Heni Boubaker"
year: 2026
doi: "10.3390/jrfm19010077"
journal: "Journal of Risk and Financial Management (JRFM)"
editorial_quality: "Q2 (MDPI)"
status: "Include"
exclusion_reason: "None"
problem_type: "Point forecast"
horizon: "single-step"
proxy: "Realized volatility (RV) — proxy based on rolling 22-day squared daily returns, annualized"
architectural_family: "Hybrid"
architectural_variant: "HAR-LSTM-GARCH (three-stage sequential)"
market_asset: "Brent Crude Oil (spot)"
---

# Resumen
El estudio propone una arquitectura híbrida secuencial de tres etapas (HAR-LSTM-GARCH) para pronosticar la volatilidad realizada del crudo Brent. La primera etapa utiliza un modelo Heterogeneous Autoregressive (HAR) para capturar la dinámica lineal. La segunda etapa emplea una red Long Short-Term Memory (LSTM) para modelar los residuos no lineales del modelo HAR, con hiperparámetros optimizados mediante Bayesian Optimization. La tercera etapa aplica un modelo GARCH(1,1) a los residuos del híbrido para modelar la heterocedasticidad condicional de los errores y construir intervalos de predicción. Los resultados reportan que el modelo supera a los benchmarks tradicionales y de Deep Learning puro, validado mediante la prueba Model Confidence Set (MCS). **El estudio se incluye porque predice volatilidad, usa una partición temporal correcta, una descomposición sin look-ahead y pruebas de significancia formales.**

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **Volatilidad Realizada (RV)** diaria del crudo Brent. Construyen la RV a partir del cuadrado de los retornos diarios, aplicando una ventana móvil de 22 días y anualizando:
$$RV^{daily}_t = \sqrt{\frac{252}{22} \sum_{i=1}^{22} r^2_{t-i+1}}$$
donde $r_t = \ln(P_t/P_{t-1}) \times 100$. **No usan datos intradía de alta frecuencia**: ante la ausencia de high-frequency data, usan el retorno diario al cuadrado como proxy de la varianza diaria, y luego lo suavizan con una ventana de 22 días. Es un proxy de volatilidad, no la RV de 5 minutos.

**Fuente de los datos:**
- **Precios spot diarios del crudo Brent:** 4 de enero de 2010 – 30 de diciembre de 2022. **No reporta la base de datos o proveedor específico** (ej. Bloomberg, EIA, Investing). Los precios son públicos, pero la fuente no está declarada.
- **Data availability:** “Data available on request”.

**Partición Temporal:** **70% Train, 15% Validation, 15% Test. Cronológica.** Sin rolling origin a nivel de modelo completo. El LSTM se entrena una vez sobre el train y se evalúa sobre el test, con early stopping monitoreado en validación.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** **Hold-out estático, open-loop con actuales (implícito).** El modelo se entrena una vez sobre el train y se evalúa una vez sobre el test. No hay reentrenamiento entre puntos. No se declara explícitamente si la predicción es open-loop con actuales o recursiva, pero dado que es one-step-ahead y el target es $RV_{t+1}$, lo más probable es open-loop con actuales.

**Normalización:** multiplicaron los retornos por 100 para estandarizar la escala. **No reporta** el uso de MinMaxScaler o StandardScaler sobre las variables de entrada antes de la red LSTM.

**Ingeniería de Características:**
- **Construcción de componentes HAR:** $RV^d_t = RV^{daily}_t$; $RV^w_t = \frac{1}{5}\sum_{i=0}^{4} RV^{daily}_{t-i}$; $RV^m_t = \frac{1}{22}\sum_{i=0}^{21} RV^{daily}_{t-i}$. Todos son rolling window hacia atrás, por lo que son ex-ante correctos.
- **Sin descomposición:** no hay EMD, VMD ni filtros no causales.
- **Sin variables exógenas:** el modelo es univariado; solo usa la serie de RV y sus componentes HAR.
- **Ventana de 22 días:** justificada por “convención de la práctica financiera estándar” (un mes de operaciones = 22 días hábiles). No hay justificación empírica (no hay validación cruzada ni análisis de sensibilidad).

# 2. Arquitectura Final (HAR-LSTM-GARCH)

El modelo tiene tres etapas secuenciales:
1. **Etapa 1 — HAR (OLS):** captura la relación lineal y genera los residuos. La ecuación es:
   $$RV_{t+1} = \beta_0 + \beta_d RV^d_t + \beta_w RV^w_t + \beta_m RV^m_t + \epsilon_{t+1}$$
   donde los parámetros se estiman por OLS. El forecast lineal es $L_{t+1} = \hat{\beta}_0 + \hat{\beta}_d RV^d_t + \hat{\beta}_w RV^w_t + \hat{\beta}_m RV^m_t$. Los residuos son $e_t = RV_t - L_t$.
2. **Etapa 2 — LSTM:** modela los residuos del HAR. La red recibe una secuencia de los últimos $L^*$ residuos y predice el siguiente:
   $$\hat{e}_{t+1} = f_{LSTM}(e_t, e_{t-1}, \dots, e_{t-L^*+1}; \Theta^*)$$
   donde $\Theta^*$ son los hiperparámetros óptimos encontrados por Bayesian Optimization. El forecast híbrido es $H_{t+1} = L_{t+1} + \hat{e}_{t+1}$.
3. **Etapa 3 — GARCH(1,1):** modela la heterocedasticidad condicional de los residuos del HAR-LSTM:
   $$z_t = RV_t - H_t$$
   $$\sigma^2_{z,t} = \omega_z + \alpha_z z^2_{t-1} + \beta_z \sigma^2_{z,t-1}$$
   El forecast de la varianza del error es $\hat{\sigma}^2_{z,t+1}$. El forecast puntual no cambia, pero se usan intervalos de predicción: $[H_{t+1} - 1.96\hat{\sigma}_{z,t+1}, H_{t+1} + 1.96\hat{\sigma}_{z,t+1}]$.

**LSTM (arquitectura):**
- 1 capa LSTM y 1 capa Dense (salida).
- Neuronas: 36 unidades en la capa LSTM (configuración óptima encontrada para el híbrido).
- Activaciones: Sigmoide en las puertas internas, Tanh en la actualización del estado de la celda, Lineal en la capa Dense de salida.
- Entrenamiento: early stopping monitoreando el error del validation set.

# 3. Resultados y Comparaciones

**Modelos comparados (6 en total):**
- **HAR:** modelo base econométrico.
- **LSTM puro:** Deep Learning sin HAR.
- **Stochastic Volatility (SV):** modelo de volatilidad latente.
- **GARCH(1,1):** modelo base de varianza condicional.
- **GJR-GARCH:** GARCH con asimetría.
- **NN-GARCH:** red neuronal simple + GARCH.

**Métricas de Evaluación:**
- **Punto:** RMSE, MAPE, R².
- **Probabilístico:** QLIKE.
- **Estadística:** Model Confidence Set (MCS) con 25,000 simulaciones de bootstrap, evaluado bajo MSE, MAE y QLIKE.

**Ganador y Rendimiento:**
El modelo **HAR-LSTM-GARCH** es el ganador. Resultados out-of-sample (80% train / 20% test):

| Modelo | RMSE | MAPE | R² |
|---|---|---|---|
| **HAR-LSTM-GARCH** | **0.0145** | **0.0342** | **0.9537** |
| HAR | 0.0150 | 0.0373 | 0.9501 |
| LSTM | 0.0158 | 0.0371 | 0.9464 |
| SV | 0.0509 | 0.1893 | 0.6930 |
| GARCH | 0.2056 | 0.8030 | 0.9053 |
| GJR-GARCH | 0.0376 | 0.1379 | 0.8097 |
| NN-GARCH | 0.0230 | 0.0776 | 0.9456 |

**Pruebas de Significancia:**
MCS con 25,000 simulaciones de bootstrap, evaluado bajo MSE, MAE y QLIKE. El modelo propuesto fue **el único que se mantuvo en el conjunto de confianza con p-valor de 1.000 en las tres métricas**. Los benchmarks fueron eliminados secuencialmente (p-valores < 0.05 en su mayoría).

**Sesgo de Reporte:**
Bajo. El estudio tiene una comparación exhaustiva contra 6 baselines fuertes, aplica las mismas condiciones de partición para todos, y utiliza una prueba estadística formal (MCS) para demostrar la supremacía del modelo. La partición es cronológica, el validation set está separado, y el target es volatilidad (no precio). Las limitaciones principales son: (i) el target es un proxy de volatilidad (rolling 22-day squared returns) en lugar de RV de alta frecuencia; (ii) el modelo es univariado y no incorpora variables exógenas; (iii) el LSTM es una caja gris (aunque HAR y GARCH son interpretables); (iv) el target está suavizado (ventana de 22 días), lo que facilita la predicción.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número total de parámetros de la red.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta el tiempo exacto, pero menciona que el framework es computacionalmente intensivo por la optimización bayesiana y el proceso secuencial.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** “Data available on request”.
- **Métricas de calibración:** no reporta evaluación de calibración probabilística (ECE, Brier). Aunque construyen intervalos de predicción con el GARCH, no miden su cobertura empírica.
- **Limitaciones reportadas:** uso de un solo activo (Brent), uso de un proxy de volatilidad (retornos diarios en lugar de datos intra-diarios), alta intensidad computacional y falta de interpretabilidad del componente LSTM (“grey box”). Además, al ser univariado, no incorpora variables macroeconómicas o geopolíticas.

## 5.2 Hiperparámetros reportados (valores finales)
- **HAR (OLS):** $\beta_0 = 0.00361$, $\beta_d = 0.66030$, $\beta_w = 0.45133$, $\beta_m = -0.12439$ (todos significativos al 0.001).
- **LSTM (HAR-LSTM):** 36 unidades, learning rate = 5.51×10⁻⁴, batch size = 62, epochs = 452.
- **LSTM (standalone):** 97 unidades, learning rate = 5.81×10⁻³, batch size = 12, epochs = 198.
- **GARCH(1,1):** $\omega = 0.0036$, $\alpha_1 = 0.1574$, $\beta_1 = 0.6653$, skew = 0.2121, shape = 2.5027 (distribución Skewed Student’s).
- **GJR-GARCH(1,1):** reportado como benchmark.
- **Función de pérdida:** MSE (para el LSTM). No se declara explícitamente en el texto, pero es la métrica que se minimiza en el validation set.

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | Bayesian Optimization con kernel Matern 5/2 |
| Espacio de búsqueda | Units ∈ [10, 100]; LR ∈ [1e-4, 1e-2] (log); Batch ∈ [8, 64]; Epochs ∈ [20, 500] |
| Trials | 20 |
| Fitness | MSE sobre validation set (15%) |
| Valores finales (HAR-LSTM) | 36 unidades, LR=5.51e-4, batch=62, epochs=452 |
| Valores finales (LSTM) | 97 unidades, LR=5.81e-3, batch=12, epochs=198 |
| Early stopping | Monitoreado sobre validation set |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No.** Usan un **conjunto de validación separado (15%)** para la búsqueda bayesiana de hiperparámetros y para early stopping. El paper lo declara explícitamente: “The optimization was configured to minimize the Mean Squared Error (MSE) of the LSTM’s one-step-ahead forecasts on a held-out validation set (constituting 15% of the training data).”

**¿Sobre qué conjunto se evalúa el fitness?** **Sobre el validation set (15%).** El paper lo declara. No hay ambigüedad.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **MSE sobre el validation set.** La función de pérdida es MSE (implícita en la descripción del entrenamiento y en el fitness de la optimización bayesiana).

### 5.3.4 GARCH y GJR-GARCH
La especificación del GARCH se seleccionó mediante un procedimiento de selección de modelos (GARCH(p,q) con p,q ≤ 2; distribuciones Normal, Skewed Normal, Student’s, Skewed Student’s). El criterio fue RMSE. Se seleccionó GARCH(1,1) con Skewed Student’s. Los parámetros se estiman por máxima verosimilitud.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Los hiperparámetros del LSTM están reportados, pero faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura del LSTM | Sí (1 capa, 36 unidades, Dense) | Bajo: está reportada |
| Función de pérdida | Sí (MSE implícita) | Bajo: está declarada |
| Hiperparámetros del LSTM | Sí (36, 5.51e-4, 62, 452) | Bajo: están reportados |
| Espacio de búsqueda | Sí (Units, LR, Batch, Epochs) | Bajo: está reportado |
| Normalización | No (solo menciona multiplicar retornos por 100) | Medio: afecta la escala de los inputs |
| Esquema de predicción (open-loop vs recursivo) | No declarado | Medio: no se sabe si el error se propaga |
| Parámetros del GARCH | Sí (Tabla 2) | Bajo: están reportados |
| Código | No | Alto: no hay referencia ejecutable |
| Semilla | No | Medio: los modelos DL son estocásticos |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Sí, con un riesgo de divergencia medio.** Los supuestos que habría que tomar:
1. **Normalización:** Min-Max o Z-score ajustado solo con train.
2. **Esquema de predicción:** open-loop con actuales.
3. **Semilla:** fijar una semilla para la red.
4. **Early stopping:** paciencia no reportada; se asume un valor típico (e.g., 10-20 épocas).

**Riesgo de divergencia del desempeño reportado:** **medio.** El estudio tiene una partición cronológica correcta, un validation set separado, y pruebas de significancia formales (MCS). El desempeño reportado es creíble. Sin embargo, la falta de normalización declarada y la ausencia de semilla introducen incertidumbre. Además, el target está suavizado (ventana de 22 días), lo que facilita la predicción y puede inflar el R².

**Riesgo de divergencia de la arquitectura:** **bajo.** La arquitectura está descrita con detalle (HAR → LSTM → GARCH, con 36 unidades y hiperparámetros reportados).

**Riesgo de divergencia del pipeline:** **bajo.** Los pasos están claros; la descomposición HAR es rolling, y el GARCH se aplica a los residuos.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, normalización no declarada, esquema de predicción no declarado, ausencia de código y ausencia de semilla, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no una réplica exacta. Sin embargo, **el estudio se incluye en el benchmark** porque cumple los criterios de inclusión (C1, C2, C3, E5, E6), tiene una partición temporal correcta, y usa pruebas de significancia formales.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad del crudo con LSTM como componente central.
- **C2. Reporte suficiente:** **Parcial.** Reporta hiperparámetros del LSTM, parámetros del HAR y GARCH, y el esquema de validación. Sin embargo, no reporta normalización, semilla ni código. La reproducibilidad es parcial.
- **C3. Calidad editorial:** **Sí.** Q2 en JCR (MDPI - Economics, Finance).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El protocolo es reconstruible con supuestos (partición cronológica, validation set separado, hiperparámetros reportados).
- **E6. Solo precio, sin volatilidad:** **No aplica.** Predice RV (proxy).
- **E7. GARCH como contribución principal:** **No aplica.** GARCH es una etapa del híbrido, no la contribución principal.

**Resultado:** **Incluido.**

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** RV proxy basada en rolling 22-day squared returns. → moderado (target suavizado, más fácil que RV diaria; no es RV de alta frecuencia).
- **Partición:** 70/15/15 cronológica, con validation set separado. → bajo.
- **Descomposición:** HAR rolling window, sin look-ahead. → bajo.
- **Predicción:** hold-out con open-loop implícito, esquema no declarado. → moderado.
- **Fuga:** sin evidencia de fuga por descomposición, normalización ni tuning (Bayesian optimization sobre validation). → bajo.
- **Reporte selectivo:** reporta MCS con 25,000 bootstrap, bajo MSE, MAE y QLIKE. → bajo.
- **Inferencia:** MCS formal. → bajo.
- **Comparación:** 6 benchmarks fuertes, con ablation study (HAR, LSTM, HAR-LSTM, HAR-LSTM-GARCH). → bajo.
- **Reproducibilidad:** sin código, sin semilla, normalización no declarada. → parcial.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **parcial.** Precios de Brent públicos, pero la fuente específica no está declarada. → **Tal vez / Sí con proxy**.
- **Reconstructibilidad del pipeline:** **sí, con supuestos.** HAR → LSTM (36 unidades) → GARCH(1,1), con hiperparámetros reportados. → **Sí**.
- **Viabilidad computacional:** **sí.** El modelo es ligero (36 unidades, 20 trials de Bayesian optimization). → **Sí**.

**Decisión provisional:**
- **Entra al benchmark:** sí.
- **Tipo de entrada:** **directo** (con corrección menor: declarar normalización, semilla y esquema de predicción).
- **Correcciones necesarias:**
    - Declarar el esquema de normalización (si lo hay) y el momento del ajuste (solo train, o serie completa).
    - Declarar el esquema de predicción (open-loop con actuales o recursivo) y aplicarlo igual a todos los modelos.
    - Fijar semilla para los modelos DL.
    - Reportar el número de parámetros y FLOPs de la red.
    - Reportar el tiempo de entrenamiento y el hardware utilizado.
    - Declarar la fuente específica de los precios de Brent (Bloomberg, EIA, Investing).
    - Aclarar que el target es un proxy de volatilidad (rolling 22-day squared returns), no RV de alta frecuencia, y discutir el impacto de esta elección en la comparabilidad con otros estudios del corpus.
    - Reportar la cobertura empírica de los intervalos de predicción construidos con el GARCH (PICP, PINAW) para evaluar la calibración.
- **Brecha principal:** el estudio está bien reportado y evita las fugas clásicas del corpus (partición cronológica, validation set separado, Bayesian optimization sobre validation). Las limitaciones son menores: normalización no declarada, ausencia de semilla, ausencia de reporte de complejidad computacional, y target suavizado (rolling 22-day squared returns). La fortaleza principal es el **diseño secuencial HAR → LSTM → GARCH**, que descompone el problema en componentes lineal, no lineal y de heterocedasticidad, y usa MCS para validar significancia. Aporta la familia `Hybrid` con un esquema de tres etapas que el benchmark debería considerar como referencia para modelos que combinan econometría y Deep Learning.