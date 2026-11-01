---
id: Verma2021
title: "Forecasting volatility of crude oil futures using a GARCH–RNN hybrid approach."
authors: "Sauraj Verma"
year: 2021
doi: "10.1002/isaf.1489"
journal: "Intelligent Systems in Accounting, Finance and Management"
editorial_quality: "Q3 (Wiley)"
status: "Exclude"
exclusion_reason: "C3: Revista Q3 (Intelligent Systems in Accounting, Finance and Management), no fundacional. El paper no introduce una arquitectura nueva ni es un trabajo de referencia; es una aplicación estándar de GARCH-LSTM con una contribución incremental. No cumple el criterio de calidad editorial (Q1/Q2 o Q3/Q4 fundacional)."
problem_type: "Point forecast"
horizon: "multi-step"
proxy: "Realized volatility (RV)"
architectural_family: "Hybrid"
architectural_variant: "GG-LSTM (GARCH + GJR-GARCH + LSTM)"
market_asset: "WTI Crude Oil Futures"
---

# Resumen
El estudio propone tres modelos híbridos (GARCH-LSTM, GJR-LSTM y GG-LSTM) para pronosticar la volatilidad del crudo WTI a horizontes de 7, 14, 21 y 28 días. Los modelos combinan los parámetros estimados de GARCH y GJR-GARCH con variables explicativas (oro, plata, platino, log-retornos y log-retornos al cuadrado) como entradas para una red LSTM. Los resultados reportan que GG-LSTM (que concatena parámetros de GARCH y GJR-GARCH) supera a los modelos GARCH tradicionales y a los otros híbridos. El paper también evalúa la robustez cambiando el tamaño de la ventana móvil (14, 21, 28 días) y usa el Model Confidence Set (MCS) para validar significancia estadística. **El estudio se excluye del corpus porque la revista es Q3 y el paper no es fundacional (no cumple C3).**

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **Volatilidad Realizada (RV)**, calculada como la varianza muestral de los log-retornos en una ventana de 21 días **hacia adelante** (Eq. 13):
$$RV_t = \frac{1}{21} \sum_{i=1}^{t+21} (R_i - \bar{R}_t)^2$$
donde $R_t = \log(P_t) - \log(P_{t-1})$. Es decir, el target en $t$ es la varianza de los retornos desde $t+1$ hasta $t+21$. **No es la RV diaria**, sino una varianza suavizada de un mes hacia adelante. Esta elección hace el target más fácil de predecir que la RV diaria y menos útil para gestión de riesgo a 1 día.

**Fuente de los datos:**
- **Bloomberg.** Precios diarios de CL:WTI, Oro (COMEX), Plata (COMEX) y Platino (NYMEX). Periodo: 1 de abril de 2003 – 28 de febrero de 2020 (4,331 observaciones).
  - Train: Abril 2003 – Abril 2016 (3,297 obs).
  - Test: Abril 2016 – Febrero 2020 (1,034 obs).
- **Data availability:** no se declara explícitamente. Los precios son públicos (Bloomberg, EIA, Investing.com).

**Partición Temporal:** división cronológica estricta. Train 2003–2016, Test 2016–2020. **Sin rolling origin a nivel de modelo completo**, pero el LSTM se evalúa con un esquema de **rolling window** para la construcción de inputs (ventana de 14, 21 o 28 días hacia atrás). Esto evita el look-ahead en la evaluación del LSTM, pero **no se reentrena el LSTM en cada paso**: se entrena una vez y se evalúa sobre el test.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** **Hold-out con rolling window de inputs, esquema no declarado.** El LSTM se entrena una vez y se evalúa sobre el test set. La "rolling window" se refiere a la ventana de inputs (14/21/28 días), no a un reentrenamiento rolling. No se declara si la predicción es open-loop con actuales o recursiva.

**Normalización:** **no se reporta**. No se menciona el uso de escaladores ni si se ajustaron solo con el train set.

**Ingeniería de Características (SIN DESCOMPOSICIÓN):**
- **GARCH(1,1)** y **GJR-GARCH(1,1)** se estiman sobre los log-retornos del WTI en el train (2003–2016). Los parámetros estimados se reportan en Tablas 2 y 3.
- **Concatenamiento paralelo de parámetros:** GG-LSTM combina los parámetros de GARCH ($\omega, \alpha, \beta$) y GJR-GARCH ($\omega, \alpha, \beta, \gamma$). **Problema de colinealidad:** GJR-GARCH anida a GARCH cuando $\gamma = 0$, por lo que los parámetros $\omega, \alpha, \beta$ están duplicados. El vector de features resultante tiene 7 números, de los cuales 3 son redundantes y altamente correlacionados.
- **Variables exógenas adicionales:** Oro, Plata, Platino, log-retornos del WTI y log-retornos al cuadrado del WTI.
- **Ambigüedad crítica sobre los parámetros GARCH:** el paper dice que alimenta los parámetros "in a rolling-window fashion", pero **no especifica** si:
  - (a) Re-estima los parámetros del GARCH en cada paso usando solo datos hasta $t$ (correcto pero no reportado).
  - (b) Usa los parámetros estimados una sola vez sobre el train set (2003–2016) como features **constantes** para todo el test set (2016–2020). Dado que las Tablas 2 y 3 reportan una sola estimación con su error estándar, y no hay mención de re-estimación, lo más probable es que sean **estáticos**. Si es así, el LSTM recibe 7 números que no varían en el tiempo, lo cual es prácticamente inútil (solo actúan como biases).

# 2. Arquitectura Final (GG-LSTM)

El modelo **NO es un pipeline secuencial** (GARCH → GJR-GARCH → LSTM). Es un **concatenamiento paralelo de parámetros**:
1. **GARCH y GJR-GARCH en paralelo:** se estiman ambos modelos sobre los mismos retornos del train.
2. **LSTM:** recibe como inputs:
   - Los 7 parámetros (3 duplicados).
   - Precios de Oro, Plata y Platino.
   - Log-retornos del crudo.
   - Log-retornos al cuadrado del crudo.

**Arquitectura del LSTM:**
- 3 capas LSTM con 20, 10 y 10 neuronas.
- 2 capas Dense con 5 y 1 neuronas.
- Optimizador Adam con learning rate = 0.0001.
- 150 épocas.
- Dropout de 0.5 y 0.4 entre las capas LSTM.
- **No se justifica empíricamente** por qué se eligieron estos valores. No hay Grid Search ni Bayesian Optimization.

**Función de pérdida:** MSE (no reportada explícitamente, pero implícita en la descripción del entrenamiento). La evaluación usa HMSE y HMAE, que son funciones de pérdida heterocedásticas-ajustadas:
$$\text{HMSE} = \frac{1}{n}\sum_{t=1}^n (1 - \hat{\sigma}^2_t/\text{RV}_t)^2$$
$$\text{HMAE} = \frac{1}{n}\sum_{t=1}^n |1 - \hat{\sigma}^2_t/\text{RV}_t|$$

# 3. Resultados y Comparaciones

**Modelos comparados (6 en total):**
- **GARCH:** modelo base univariado.
- **GJR-GARCH:** modelo base univariado con efecto apalancamiento.
- **GARCH-LSTM:** híbrido con parámetros GARCH + exógenas.
- **GJR-LSTM:** híbrido con parámetros GJR-GARCH + exógenas.
- **GG-LSTM:** híbrido con parámetros GARCH + GJR-GARCH + exógenas.

**Métricas y Rendimiento:**
HMSE y HMAE. Resultados para la ventana de 21 días (Tabla 4):

| Horizonte | Modelo | HMSE | HMAE |
|---|---|---|---|
| 7 días | GARCH | 0.3554 | 0.4122 |
| 7 días | GJR-GARCH | 0.3755 | 0.4410 |
| 7 días | GARCH-LSTM | 0.2634 | 0.3740 |
| 7 días | GJR-LSTM | 0.2960 | 0.4166 |
| 7 días | **GG-LSTM** | **0.2279** | **0.3163** |
| 14 días | GG-LSTM | 0.2699 | 0.3526 |
| 21 días | GG-LSTM | 0.2845 | 0.3970 |
| 28 días | GG-LSTM | 0.3144 | 0.4301 |

El modelo **GG-LSTM** es el ganador en todos los horizontes. Reduce el HMSE en 35.87% y el HMAE en 23.26% frente a GARCH a 7 días. Frente a GJR-GARCH, la mejora es de 39.30% (HMSE) y 28.27% (HMAE).

**Pruebas de Significancia:**
Usa el **Model Confidence Set (MCS)** con 36 modelos evaluados. El GG-LSTM es el modelo con mayor número de configuraciones estadísticamente significativas (6 de 36). El paper reporta p-valores ajustados.

**Sesgo de Reporte:**
Alto. Cinco problemas:
1. **Comparación estructuralmente injusta:** los GARCH y GJR-GARCH base son **univariados** (solo usan retornos históricos del crudo). Los LSTM son **multivariados** (parámetros GARCH + 3 precios de metales + log-retornos + log-retornos al cuadrado). Es imposible saber si el LSTM es mejor por su arquitectura o porque le dieron mucha más información. La comparación justa sería contra un GARCH-X (que también use las variables exógenas) o un LSTM que solo use los retornos históricos.
2. **Target suavizado:** la varianza hacia adelante de 21 días es un target más suave y fácil de predecir que la RV diaria. Los resultados no son comparables con estudios que usan RV diaria.
3. **Colinealidad en GG-LSTM:** la combinación GARCH + GJR-GARCH duplica los parámetros $\omega, \alpha, \beta$. El vector de features tiene 3 números redundantes.
4. **Ambigüedad sobre los parámetros GARCH:** no se especifica si son estáticos o rolling. Si son estáticos (lo más probable), el LSTM recibe 7 números constantes que no aportan información temporal.
5. **Falta de tuning:** la arquitectura de la red se fija sin justificación empírica.
6. **Inconsistencia en el abstract:** el abstract dice que GG-LSTM es el mejor a 7, 14 y 21 días, pero la Tabla 4 muestra que también es el mejor a 28 días. Es una omisión, no un error.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros del LSTM.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** no se declara explícitamente. Los precios son públicos.
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast).
- **Limitaciones reportadas:** los autores admiten que el estudio se enfoca en la volatilidad del WTI y que futuras investigaciones podrían explorar otros mercados y horizontes.

## 5.2 Hiperparámetros reportados (valores finales)
- **GARCH(1,1):** $\omega = 3 \times 10^{-6}$, $\alpha = 0.058$, $\beta = 0.938$.
- **GJR-GARCH(1,1):** $\omega = 2 \times 10^{-6}$, $\alpha = 0.022$, $\beta = 0.945$, $\gamma = 0.060$.
- **LSTM:** 3 capas LSTM (20, 10, 10), 2 Dense (5, 1), Adam lr=0.0001, 150 épocas, dropout 0.5 y 0.4.
- **Función de pérdida:** MSE (no declarada explícitamente).
- **Rolling window:** 14, 21, 28 días (evaluados como robustness check).

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **No se reporta Grid Search ni Bayesian Optimization.** La arquitectura se fija de forma arbitraria. |
| Espacio de búsqueda | NO reportado. |
| Valores finales | 20-10-10 (LSTM), 5-1 (Dense), lr=0.0001, 150 epochs, dropout 0.5/0.4. |
| Optimizador | Adam |
| Learning rate | 0.0001 |
| Batch size | NO reportado. |
| Épocas | 150 |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No.** No se menciona ningún esquema de validación. Los hiperparámetros se fijan de forma arbitraria.

**¿Sobre qué conjunto se evalúa el fitness?** **NO reportado.** No hay conjunto de validación separado ni criterio de selección declarado.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **NO reportado.** No se especifica la función de pérdida del LSTM. Por el contexto, se asume MSE sobre RV, pero no se declara.

### 5.3.4 Parámetros GARCH
Los parámetros GARCH se estiman una sola vez sobre el train (2003–2016). No se re-estiman en cada paso del test. Si se usan como features constantes, el LSTM recibe 7 números que no varían en el tiempo.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Los hiperparámetros del LSTM están reportados, pero faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura del LSTM | Sí (20-10-10, Dense 5-1) | Bajo: está reportada |
| Función de pérdida | No | Medio: se asume MSE, pero no se declara |
| Batch size | No | Medio: afecta el entrenamiento |
| Parámetros GARCH | Sí (Tabla 2 y 3) | Bajo: están reportados |
| ¿Parámetros GARCH estáticos o rolling? | No | **Alto**: si son estáticos, el LSTM recibe features constantes |
| Normalización | No | Medio: afecta la escala del target |
| Esquema de predicción (open-loop vs recursivo) | No declarado | Medio: no se sabe si el error se propaga |
| Código | No | Alto: no hay referencia ejecutable |
| Semilla | No | Medio: los modelos DL son estocásticos |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Sí, con un riesgo de divergencia medio.** Los supuestos que habría que tomar:
1. **Parámetros GARCH estáticos** (estimados una vez sobre train). Esto es lo más probable según el texto.
2. **Función de pérdida:** MSE sobre RV.
3. **Normalización:** Min-Max o Z-score ajustado solo con train.
4. **Esquema de predicción:** open-loop con actuales.
5. **Semilla:** fijar una semilla para el LSTM.

**Riesgo de divergencia del desempeño reportado:** **medio.** El estudio no tiene fuga evidente (partición cronológica, rolling window de inputs). Sin embargo, la colinealidad en GG-LSTM, la ambigüedad sobre los parámetros GARCH y la comparación injusta con GARCH univariado introducen incertidumbre. El desempeño reportado probablemente se mantiene cualitativamente (GG-LSTM > GARCH-LSTM > GJR-LSTM > GARCH), pero la magnitud exacta puede variar.

**Riesgo de divergencia de la arquitectura:** **bajo.** La arquitectura está reportada con detalle.

**Riesgo de divergencia del pipeline:** **medio.** Los pasos están claros, pero la ambigüedad sobre los parámetros GARCH requiere supuestos.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, función de pérdida no declarada, batch size no reportado, ambigüedad sobre parámetros GARCH estáticos vs rolling, normalización no declarada y ausencia de código, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no una réplica exacta. Sin embargo, **este punto es secundario porque el estudio se excluye por C3 (Q3 no fundacional).**

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad del crudo con LSTM como componente central.
- **C2. Reporte suficiente:** **Parcial.** Reporta hiperparámetros del LSTM y parámetros GARCH, pero no la función de pérdida, el batch size, la normalización, el esquema de predicción ni el código. Además, la ambigüedad sobre parámetros GARCH estáticos vs rolling impide la reconstrucción exacta.
- **C3. Calidad editorial:** **No.** La revista *Intelligent Systems in Accounting, Finance and Management* es **Q3** en Scopus/WoS. El paper no es fundacional: no introduce una arquitectura nueva ni es un trabajo de referencia en la literatura de GARCH-LSTM. Es una aplicación estándar de híbridos GARCH-LSTM. **Falla C3.**
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** Parcial (el protocolo es reconstruible con supuestos, pero no de forma exacta).
- **E6. Solo precio, sin volatilidad:** No aplica. Predice RV.
- **E7. GARCH como contribución principal:** No aplica. GARCH es un componente del híbrido, no la contribución principal.

**Resultado:** **Excluido.** Motivo: **C3** (revista Q3, no fundacional).

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** RV suavizada de 21 días hacia adelante. → moderado (más fácil que RV diaria).
- **Partición:** cronológica 2003–2016 / 2016–2020, sin rolling origin a nivel de modelo. → moderado.
- **Predicción:** hold-out con rolling window de inputs, esquema de input no declarado. → moderado.
- **Fuga:** no hay evidencia de fuga por descomposición, normalización o tuning. → bajo.
- **Reporte selectivo:** reporta MCS, pero no intervalo de confianza para las diferencias. → moderado.
- **Inferencia:** MCS con 36 modelos. → bajo.
- **Comparación:** estructuralmente injusta (GARCH univariado vs LSTM multivariado). → alto.
- **Reproducibilidad:** sin código, sin función de pérdida, sin batch size, sin normalización, ambigüedad sobre parámetros GARCH. → parcial.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** WTI, Oro, Plata y Platino públicos (Bloomberg, EIA, Investing.com). → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** La arquitectura está reportada, pero faltan la función de pérdida, el batch size y la normalización. La ambigüedad sobre parámetros GARCH requiere supuestos. → **Parcial**.
- **Viabilidad computacional:** **sí.** Los modelos son ligeros. → **Sí**.

**Decisión provisional:**
- **Entra al benchmark:** **No.**
- **Tipo de entrada:** **N/A (excluido por C3).**
- **Correcciones necesarias:**
    - Declarar si los parámetros GARCH se estiman una sola vez sobre el train (estáticos) o se re-estiman en cada paso del test (rolling). Aclarar si el LSTM recibe features constantes o variables en el tiempo.
    - Declarar la función de pérdida del LSTM y el batch size. Se asume MSE, pero no se declara.
    - Declarar el esquema de normalización (si lo hay) y el momento del ajuste (solo train, o serie completa).
    - Garantizar la comparación justa: incluir un GARCH-X (con las mismas variables exógenas que el LSTM) o un LSTM univariado (solo con retornos históricos del crudo) para aislar el efecto de la arquitectura frente al efecto de la información adicional.
    - Reportar intervalos de confianza para las diferencias de HMSE y HMAE (bootstrap o DM), no solo p-valores del MCS.
    - Liberar el código o, en su defecto, reportar los valores finales de los hiperparámetros del LSTM (batch size, inicialización, semilla) para permitir la reimplementación exacta.
    - Aclarar la inconsistencia del abstract: dice que GG-LSTM es el mejor a 7, 14 y 21 días, pero la Tabla 4 muestra que también es el mejor a 28 días.
- **Brecha principal:** N/A para el benchmark. El estudio se reporta en la revisión como **evidencia de que existe una línea de investigación de híbridos GARCH-LSTM para volatilidad que, aunque metodológicamente razonable, no alcanza el estándar de calidad editorial del corpus (Q3 no fundacional) y presenta una comparación estructuralmente injusta entre GARCH univariado y LSTM multivariado**. Las limitaciones específicas son: ambigüedad sobre parámetros GARCH estáticos vs rolling, colinealidad en GG-LSTM por duplicación de parámetros, target suavizado (varianza de 21 días hacia adelante, no RV diaria), ausencia de tuning del LSTM, y comparación injusta con GARCH univariado. No compite en la comparación de modelos.