---
id: Kakade2022
title: "Value-at-Risk forecasting: A hybrid ensemble learning GARCH-LSTM based approach."
authors: "Kshitij Kakade; Ishan Jain; Aswini Kumar Mishra"
year: 2022
doi: "10.1016/j.resourpol.2022.102903"
journal: "Resources Policy"
editorial_quality: "Q1 (Elsevier)"
status: "Include"
exclusion_reason: "None"
problem_type: "Point forecast (1-day VaR)"
horizon: "single-step (1-day)"
proxy: "EWMA of squared daily returns (252-day window), labeled as 'realized volatility' — SESGADO"
architectural_family: "Hybrid (Ensemble)"
architectural_variant: "GARCH/eGARCH/tGARCH + LSTM/BiLSTM + ANN meta-learner (FHS-BiLSTM-HYBRID)"
market_asset: "WTI Crude Oil Spot"
---

# Resumen
El estudio propone un modelo de ensamble que combina las predicciones de tres modelos GARCH (GARCH, eGARCH, tGARCH) con redes LSTM y BiLSTM para pronosticar la volatilidad y calcular el Value-at-Risk (VaR) al 95% y 99% del crudo WTI. Evalúan el modelo en dos periodos de crisis extrema (Crisis Financiera de 2008 y pandemia de COVID-19), reportando que el modelo **FHS-BiLSTM-HYBRID** es el mejor. El paper presenta **inconsistencias metodológicas graves**: (i) el proxy de volatilidad está **mal etiquetado como "realized volatility"** y es un estimador **SESGADO** (ventana exponencial de 252 días sobre retornos diarios al cuadrado); (ii) el tuning se hace con Grid Search **sin conjunto de validación reportado**, lo que sugiere fuga de información desde el test; (iii) el protocolo de evaluación es **inconsistente** (GARCH se re-estima con ventana móvil diaria, LSTM/ensemble se entrena una sola vez y se congela durante todo el backtest); (iv) **la comparación entre modelos está confundida por el número de inputs** (el HYBRID recibe 4 series, el LSTM puro recibe 1) y el **ablation study no aísla el efecto del ensemble del efecto de más inputs**; (v) no hay pruebas de significancia **entre modelos** ni varianza entre las 25 ejecuciones. **El estudio se incluye porque aborda el pronóstico de volatilidad (proxy) con deep learning y partición temporal documentada, pero requiere correcciones sustantivas para el benchmark.**

# 1. Método y Datos

**¿Qué se predice exactamente?**
Se predice el **VaR a 1 día** al 95% y 99%. Para ello, se pronostica una **volatilidad condicional** que luego se usa en dos métodos de VaR:
- **Paramétrico (PAR):** $\text{VaR}_\alpha = \mu + \sigma F_\alpha$.
- **Filtered Historical Simulation (FHS):** estandarización de retornos por volatilidad pronosticada + bootstrap.

**EL ESTIMADOR ES SESGADO Y ESTÁ MAL ETIQUETADO.** **NO predicen Volatilidad Realizada (RV).** El paper llama "Realized Volatility" a una **ventana exponencial móvil de 252 días sobre retornos diarios al cuadrado**. Esto **no es RV**: la RV requiere datos intradía de alta frecuencia. El proxy que usan es un **estimador SESGADO** por cuatro fuentes que se acumulan:
1. **Suavizado extremo:** una ventana de 252 días (un año) promedia la volatilidad de todo un año. No captura la dinámica de corto plazo ni los shocks abruptos.
2. **Solo precios de cierre:** ignora toda la varianza intradía y el *overnight gap*.
3. **Mal etiquetado:** llamarlo "realized volatility" induce a error y hace que los errores (RMSE, QLIKE) **no sean comparables** con estudios que usan RV de alta frecuencia.
4. **Parámetro de decaimiento no reportado:** no se reporta el factor $\lambda$ de la ventana exponencial ni el criterio exacto de ponderación.

**Conclusión:** el proxy es **SESGADO** y no comparable con RV. Los resultados de VaR dependen de un target suavizado y mal etiquetado.

**Fuente de los datos:**
- **Precios:** WTI Spot Price, US Energy Information Administration (EIA). Periodo total: 28 de agosto de 1992 – 7 de abril de 2021.
- **Periodo 1 (Crisis Financiera):** entrenamiento 28 ago 1992 – 4 feb 2008; backtesting hasta 6 abr 2009 (305 observaciones).
- **Periodo 2 (COVID-19):** entrenamiento 31 ago 2004 – 5 feb 2020; backtesting hasta 7 abr 2021 (305 observaciones).
- **Data availability:** "Data will be made available on request". Los precios del WTI son públicos (EIA).

**Partición Temporal:**
- **Modelos GARCH:** ventana móvil de 252 días (re-estimación diaria, aunque la redacción "fit the GARCH model every 252 days" es ambigua).
- **Redes neuronales y ensemble:** división **cronológica estática** para entrenamiento. Se entrenan una vez y se congelan para predecir todo el periodo de backtesting (305 días).
- **No se reporta conjunto de validación separado.** Solo hay train y backtesting.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?**
**Hold-out estático para las redes, con GARCH rolling.** Las redes LSTM/BiLSTM y el ensemble se entrenan una vez sobre el train set y se evalúan sobre el backtest completo. No hay reentrenamiento de la red entre puntos. No se declara explícitamente si la predicción es open-loop con actuales o recursiva, ni si el esquema de input es cerrado o abierto.

**Normalización:**
**No se reporta** explícitamente el uso de MinMaxScaler o StandardScaler sobre las variables de entrada antes de la red LSTM. No se especifica si el escalador se ajustó solo con el train o con la serie completa.

**Ingeniería de Características:**
- **Entradas a las redes:** cada una de las 4 redes del ensemble recibe una serie univariada en ventana de 22 días:
  - Red 1: la volatilidad histórica (proxy EWMA 252 días).
  - Red 2: predicciones de GARCH.
  - Red 3: predicciones de eGARCH.
  - Red 4: predicciones de tGARCH.
- **Sin variables exógenas:** no hay features de noticias, sentimiento, redes complejas ni descomposición de señal.
- **Sin selección de features:** no hay selección de variables; se usan las 4 series directamente.

**Asignación de inputs por modelo (asimetría crítica):**
- **GARCH, eGARCH, tGARCH:** reciben **solo la serie de retornos** (modelos univariados).
- **LSTM / BiLSTM puro:** reciben **solo el proxy EWMA** (volatilidad pasada, ventana 22 días) — **1 serie**.
- **LSTM-HYBRID / BiLSTM-HYBRID:** reciben el proxy EWMA **+** las predicciones de GARCH, eGARCH y tGARCH — **4 series**.

**Esto confunde la comparación.** El HYBRID recibe 4 veces más información que el LSTM puro. La mejora del HYBRID sobre el LSTM **no se puede atribuir a la arquitectura de ensamble** (meta-aprendiz ANN) porque también tiene más inputs. Falta el modelo de control con **los mismos 4 inputs pero sin meta-aprendiz** (p. ej., un LSTM que reciba las 4 series concatenadas, o un promedio simple de las 4 redes). Sin ese control, el efecto del ensamble **no está identificado**. La comparación GARCH vs DL es aún más injusta: GARCH ve 1 serie, el DL ve 4. La comparación justa sería un GARCH-X / HAR con las mismas variables exógenas, o un DL con solo retornos como input.

**Tuning de Hiperparámetros:**
**Grid Search** para seleccionar el número de nodos (128, 256, 512), batch size (32, 64), funciones de activación (sigmoid, ReLU), optimizadores (Adam, RMSprop) y learning rate (default 0.001). **CRÍTICO:** No se reporta un conjunto de validación separado. Dado que solo mencionan Train y Backtesting, es altamente probable que los hiperparámetros se hayan optimizado mirando el desempeño en el propio conjunto de prueba, incurriendo en **data leakage**. Los valores finales reportados son: 2 capas ocultas con 256 y 128 nodos, Adam, ReLU, MSE, 100 épocas, dropout 0.2, early stopping con paciencia 5. Cada modelo híbrido se entrena 25 veces independientemente y se promedia el pronóstico.

**Protocolo de evaluación (inconsistencia crítica):**
- Los modelos **GARCH** se **re-estiman con ventana móvil** (dinámicos) durante el backtest.
- Las redes **LSTM/BiLSTM y el ensemble** se **entrenan una sola vez** (estáticos) sobre el train set y se **congelan** para predecir todo el periodo de crisis (305 días).
- Es casi imposible que un modelo estático compita en igualdad de condiciones con uno dinámico en una crisis. Esto sugiere una **mala implementación del GARCH** o un **sesgo en el entrenamiento de la LSTM**, y hace que la comparación no sea apples-to-apples.

# 2. Arquitectura Final (Ensemble GARCH-LSTM/BiLSTM + ANN meta-learner)

El modelo híbrido, denominado **FHS-BiLSTM-HYBRID** (el ganador), tiene la siguiente estructura:

1. **Cuatro redes neuronales independientes:**
   - Una LSTM (o BiLSTM) que recibe la **volatilidad histórica** (el proxy EWMA 252 días) en una ventana de 22 días.
   - Tres LSTM (o BiLSTM) que reciben, cada una, las predicciones de GARCH, eGARCH y tGARCH respectivamente, en ventana móvil.
2. **Meta-aprendiz (ANN):** las salidas de las 4 redes se combinan mediante una **Red Neuronal Artificial (ANN)** que aprende a fusionar las predicciones para acercarse a la volatilidad objetivo.
3. **Cálculo del VaR:** la volatilidad predicha por el ensemble se usa en los métodos PAR y FHS.

**Arquitectura interna de las LSTM/BiLSTM:**
- 2 capas ocultas con 256 y 128 nodos.
- Optimizador Adam, learning rate 0.001 (default).
- Activación ReLU.
- Función de pérdida MSE.
- 100 épocas, dropout 0.2, early stopping con paciencia 5.
- Cada modelo se entrena 25 veces y se promedia.

**Función de pérdida:** MSE. **Reportada explícitamente.**

**Asimetría de inputs en el ensemble:**
- Las 4 redes del ensemble reciben **series distintas**: una recibe el proxy EWMA, y las otras tres reciben las predicciones de GARCH, eGARCH y tGARCH respectivamente.
- El **meta-aprendiz (ANN)** combina las 4 salidas, pero **no se ablate**: no hay comparación contra un promedio simple de las 4 redes ni contra un LSTM único que reciba las 4 series concatenadas.
- **Consecuencia:** no se puede separar el efecto del **ensemble** (meta-aprendiz) del efecto de **tener más inputs**. El ablation study es vertical (GARCH vs eGARCH vs tGARCH, LSTM vs BiLSTM, PAR vs FHS) pero **no aísla el mecanismo de fusión**.

# 3. Resultados y Comparaciones

**Modelos comparados (17 en total):**
- **Tradicionales (VaR):** Historical Simulation (HS), Bootstrap (BHS), Age-weighted (AWHS).
- **Volatilidad GARCH:** GARCH, eGARCH, tGARCH.
- **Deep Learning puro:** LSTM, BiLSTM.
- **Ensamble:** LSTM-HYBRID, BiLSTM-HYBRID.
- **Métodos VaR:** PAR y FHS para cada modelo de volatilidad.

**Métricas y Rendimiento:**
El modelo **FHS-BiLSTM-HYBRID** es el reportado como ganador.
- **Periodo 1:** 0.98% de violaciones al 99% y 4.92% al 95%.
- **Periodo 2:** 1.31% al 99% y 4.26% al 95%.
- Las funciones de pérdida de reguladores y firmas (RQL, RC_1, FS, FABL) fueron las más bajas en ambos periodos y niveles de confianza.

**Pruebas de Significancia:**
- Se usan **UC (Kupiec)** y **CC (Christoffersen)** por modelo. Estas pruebas evalúan si cada modelo está correctamente calibrado, **no si un modelo es significativamente mejor que otro**.
- **No hay pruebas de significancia entre modelos** (no hay Diebold-Mariano ni comparaciones pareadas).
- **No se reporta la varianza ni la desviación estándar** de las 25 ejecuciones independientes. Solo se reporta el promedio.

**Sesgo de Reporte:**
Alto. Siete problemas:
1. **Proxy sesgado y mal etiquetado:** llaman "realized volatility" a una EWMA de 252 días sobre retornos al cuadrado. No es RV. No es comparable con estudios de alta frecuencia.
2. **Tuning con posible fuga:** Grid Search sin conjunto de validación reportado. Es probable que los hiperparámetros se hayan seleccionado mirando el test.
3. **Protocolo inconsistente (static vs rolling):** GARCH se re-estima diariamente; LSTM/ensemble se entrena una vez y se congela. La comparación no es equitativa.
4. **Comparación confundida por número de inputs:** el HYBRID recibe 4 series (proxy + 3 GARCH), el LSTM puro recibe 1 (proxy). La mejora del HYBRID **no se puede atribuir al ensemble** porque también tiene más inputs. Falta el control con inputs igualados (LSTM con 4 series concatenadas, sin meta-aprendiz).
5. **Ablation study incompleto:** no se ablate el meta-aprendiz (ANN) contra un promedio simple de las 4 redes; no se ablate el ensemble contra un LSTM único con los 4 inputs. El efecto del mecanismo de fusión **no está identificado**.
6. **Comparación injusta contra GARCH:** GARCH recibe solo retornos (1 serie); el DL recibe 4 series. La comparación GARCH vs DL es estructuralmente injusta. Falta un GARCH-X / HAR con las mismas variables exógenas.
7. **Sin pruebas entre modelos y sin varianza:** solo hay coverage tests UC/CC por modelo, no comparaciones pareadas con significancia; se reporta el promedio de 25 corridas pero no la dispersión.

**Fortalezas del reporte:**
- Comparación amplia: 17 modelos, 2 periodos de crisis, 2 niveles de confianza, 4 funciones de pérdida.
- Uso de coverage tests formales (UC, CC) y funciones de pérdida de regulador y firma.
- Backtesting en dos periodos de crisis extrema (2008-09 y 2020-21).
- Reporte de múltiples métricas y no solo una.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros de las LSTM/BiLSTM ni del meta-aprendiz.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** "Data will be made available on request". Los precios del WTI son públicos (EIA).
- **Métricas de calibración:** coverage tests de VaR (UC y CC).
- **Limitaciones reportadas:** los autores mencionan que futuras investigaciones deberían enfocarse en VaR a múltiples días y en integrar otras variables explicativas.

## 5.2 Hiperparámetros reportados (valores finales)
- **LSTM/BiLSTM:** 2 capas ocultas (256 y 128 nodos); Adam lr=0.001; ReLU; MSE; 100 épocas; dropout 0.2; early stopping paciencia 5.
- **Ensemble:** ANN meta-aprendiz sobre 4 salidas.
- **GARCH:** distribución t; ventana móvil 252 días. **No se reportan los órdenes (p,q)** de los modelos GARCH, eGARCH y tGARCH.
- **EWMA (proxy):** ventana exponencial 252 días sobre retornos al cuadrado. **No se reporta λ ni el criterio exacto de ponderación.**
- **Normalización:** **NO reportada.**
- **Semilla:** **NO reportada.**
- **Conjunto de validación:** **NO reportado.**

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **Grid Search** |
| Espacio de búsqueda | Nodos: 128, 256, 512; batch: 32, 64; activación: sigmoid, ReLU; optimizador: Adam, RMSprop; lr: default (0.001) |
| Valores finales | 2 capas (256, 128), Adam, ReLU, MSE, 100 épocas, dropout 0.2, early stopping 5 |
| Optimizador | Adam |
| Learning rate | 0.001 |
| Batch size | **No se declara el valor final** (candidatos 32 y 64) |
| Épocas | 100 |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No reportado.** No hay conjunto de validación separado. Solo train y backtesting. Esto es una **bandera roja de fuga**: el Grid Search pudo haber sido guiado por el desempeño en el set de prueba.

**¿Sobre qué conjunto se evalúa el fitness?** **No reportado.** Probablemente sobre el backtest, lo que constituye leakage.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **MSE** (función de pérdida de las LSTM, reportada explícitamente). No se reporta si se usa validación o early stopping sobre un conjunto separado.

### 5.3.4 Modelos econométricos
- **GARCH:** distribución t; ventana móvil 252 días. **No se reportan los órdenes (p,q)** ni el criterio de selección.
- **EWMA:** ventana exponencial 252 días. **No se reporta λ** ni el criterio de ponderación.
- **No se reportan pruebas de diagnóstico de residuos** ni validación de los modelos GARCH.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura LSTM/BiLSTM (capas, nodos) | Sí (2 capas, 256/128) | Bajo: está reportada |
| Función de pérdida | Sí (MSE) | Bajo: está declarada |
| Optimizador y learning rate | Sí (Adam, 0.001) | Bajo: están reportados |
| Batch size final | No (solo candidatos) | Medio: afecta el entrenamiento |
| Dropout y épocas | Sí (0.2, 100) | Bajo: están reportados |
| Normalización (momento del ajuste) | No | Medio: afecta la escala del target |
| Órdenes GARCH (p,q) | No | Medio: afecta las predicciones GARCH |
| λ de la EWMA | No | Medio: afecta el proxy de volatilidad |
| Conjunto de validación / protocolo de tuning | No | Alto: no se sabe si hubo fuga |
| Semilla | No | Medio: los modelos DL son estocásticos |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Parcialmente, con un riesgo de divergencia alto.** Los supuestos que habría que tomar:
1. **GARCH:** GARCH(1,1), eGARCH(1,1), tGARCH(1,1) con distribución t.
2. **EWMA:** λ=0.94 (valor típico de RiskMetrics) o similar.
3. **LSTM/BiLSTM:** batch size=32, 100 épocas, early stopping con paciencia 5.
4. **Normalización:** Min-Max ajustado solo con train.
5. **Protocolo:** walk-forward con re-entrenamiento periódico o static, pero **homogéneo** para todos los modelos.
6. **Semilla:** fijar una semilla.

**Riesgo de divergencia del desempeño reportado:** **alto.** El desempeño reportado está inflado por: (i) posible fuga en el tuning (Grid Search sin validación); (ii) proxy sesgado y mal etiquetado; (iii) protocolo inconsistente (static vs rolling); (iv) confound de inputs (HYBRID=4 series vs LSTM puro=1 serie). Sin estas fugas e inconsistencias, el desempeño sería distinto.

**Riesgo de divergencia de la arquitectura:** **medio.** La arquitectura está descrita a grandes rasgos, pero faltan batch size final, órdenes GARCH, λ de la EWMA y normalización.

**Riesgo de divergencia del pipeline:** **alto.** El protocolo de tuning no está declarado, el protocolo de evaluación es inconsistente y el ablation study no aísla el efecto del ensemble.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, proxy sesgado y mal etiquetado, tuning sin validación reportada, protocolo inconsistente (static vs rolling), inputs desiguales entre modelos, ablation study incompleto, órdenes GARCH no reportados, λ de la EWMA no reportado, normalización no declarada y ausencia de código, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no una réplica exacta. El benchmark debe tratar la versión corregida como una **variante** del modelo original y reportar la brecha entre ambos desempeños.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad (proxy) del crudo con LSTM/BiLSTM como componente central del ensemble.
- **C2. Reporte suficiente:** **Parcial.** Reporta arquitectura, función de pérdida, optimizador y varios hiperparámetros, pero no el batch size final, los órdenes GARCH, el λ de la EWMA, la normalización, el conjunto de validación ni el código. **El ablation study no aísla el efecto del ensemble del efecto de más inputs**, lo que impide verificar la afirmación central del paper.
- **C3. Calidad editorial:** **Sí.** Q1 en JCR (Resources Policy, Elsevier).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El protocolo es reconstruible con supuestos (partición cronológica documentada, arquitectura descrita). La fuga por tuning es un problema de validez, no de reconstruibilidad. Se incluye con reservas.
- **E6. Solo precio, sin volatilidad:** **No aplica.** Predice volatilidad (proxy) para VaR.
- **E7. GARCH como contribución principal:** **No aplica.** GARCH es un insumo del ensemble; la contribución principal es el esquema híbrido GARCH-LSTM/BiLSTM con meta-aprendiz.

**Resultado:** **Incluido** (con reservas).

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** EWMA de 252 días sobre retornos diarios al cuadrado, mal etiquetado como "realized volatility". **SESGADO** (suavizado extremo, solo precios de cierre, λ no reportado). → **alto**.
- **Partición:** features GARCH rolling (252 días); red neuronal y ensemble con hold-out estático. → **moderado**.
- **Predicción:** hold-out para la red, sin rolling retraining; esquema no declarado. → **moderado**.
- **Fuga:** posible fuga por tuning (Grid Search sin validación reportada). Normalización no declarada. → **alto**.
- **Reporte selectivo:** reporta 17 modelos, 2 periodos y 2 niveles, pero **sin varianza entre las 25 ejecuciones**. → **moderado**.
- **Inferencia:** coverage tests UC/CC por modelo, pero **sin pruebas entre modelos** ni control de comparaciones múltiples. → **alto**.
- **Comparación:** **doble confound.** (i) Protocolo inconsistente (GARCH rolling vs LSTM static). (ii) **Inputs desiguales** (GARCH=1 serie, LSTM puro=1 serie, HYBRID=4 series). La afirmación central ("el ensemble mejora") **no está identificada** porque falta el control con inputs igualados. → **alto**.
- **Ablation study:** vertical pero **incompleto**. No aísla el meta-aprendiz (ANN vs promedio simple) ni el ensemble (HYBRID vs LSTM con 4 inputs). El mecanismo de fusión **no está identificado**. → **alto**.
- **Reproducibilidad:** sin código, sin batch size final, sin órdenes GARCH, sin λ de EWMA, sin normalización, sin semilla. → **parcial**.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** WTI Spot Price es público (EIA); los retornos y el proxy EWMA son computables. → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** Arquitectura descrita y varios hiperparámetros reportados, pero faltan normalización, órdenes GARCH, λ de EWMA, batch size final y protocolo de tuning. → **Parcial**.
- **Viabilidad computacional:** **sí.** LSTM/BiLSTM de 256/128 nodos, 4 redes, 25 ejecuciones, 100 épocas. Viable en una GPU estándar. → **Sí**.

**Decisión provisional:**
- **Entra al benchmark:** sí, con reservas.
- **Tipo de entrada:** **con corrección**.
- **Correcciones necesarias:**
    - **(PRIORITARIA) Diseñar un ablation study que aísle el efecto del ensemble del efecto de más inputs:** incluir (i) un LSTM único que reciba las 4 series concatenadas (proxy + 3 GARCH) **sin** meta-aprendiz; (ii) un promedio simple de las 4 redes **sin** meta-aprendiz; (iii) el HYBRID actual. Sin estos controles, la afirmación central ("el ensemble GARCH-LSTM mejora la precisión") **no está identificada**.
    - **(PRIORITARIA) Igualar los inputs en la comparación GARCH vs DL:** incluir un GARCH-X / HAR con las mismas variables exógenas que el DL, o reportar la comparación intra-DL (mismos inputs) como principal.
    - **Corregir la fuga por tuning:** usar un **conjunto de validación separado** o **walk-forward validation** para el Grid Search. Sin esta corrección, el desempeño reportado no es comparable.
    - **Reemplazar el proxy por Volatilidad Realizada (RV)** de alta frecuencia (5 min) o, en su defecto, **declararlo explícitamente como proxy sesgado** y dejar de llamarlo "realized volatility". La EWMA de 252 días sobre retornos diarios al cuadrado no es RV.
    - **Homogeneizar el protocolo de evaluación:** o ambos modelos (GARCH y LSTM/ensemble) son rolling, o ambos son estáticos. La comparación actual (GARCH rolling vs LSTM static) no es equitativa.
    - **Reportar los órdenes (p,q)** de GARCH, eGARCH y tGARCH, y el **λ de la EWMA**.
    - **Reportar la normalización** (si la hay) y el momento del ajuste (solo train).
    - **Reportar el batch size final** y la **semilla**.
    - **Reportar la varianza/desviación estándar** de las 25 ejecuciones independientes.
    - **Añadir pruebas de significancia entre modelos** (Diebold-Mariano, comparaciones pareadas con control de múltiples comparaciones).
    - **Liberar el código** o, en su defecto, reportar todos los valores finales de hiperparámetros y la semilla para permitir la reimplementación exacta.
    - **Añadir QLIKE** como métrica de evaluación de volatilidad, además de las funciones de pérdida de VaR. QLIKE es estándar en volatilidad y penaliza de forma más adecuada los errores en volatilidades altas.
- **Brecha principal:** El estudio aporta la familia `Hybrid (Ensemble)` con un esquema GARCH + LSTM/BiLSTM + ANN que el benchmark debería replicar con las correcciones declaradas. Las limitaciones principales son: **proxy sesgado y mal etiquetado** (EWMA 252 días sobre retornos al cuadrado, llamado erróneamente "realized volatility"), **posible fuga por tuning** (Grid Search sin validación), **protocolo inconsistente** (GARCH rolling vs LSTM static), **comparación confundida por número de inputs** (HYBRID=4 series vs LSTM puro=1 serie), **ablation study incompleto** (no aísla el meta-aprendiz ni el ensemble del efecto de más inputs), **comparación injusta contra GARCH** (1 serie vs 4 series), **sin pruebas de significancia entre modelos** y **sin código**. La afirmación central del paper ("el ensemble GARCH-LSTM mejora la precisión") **no está identificada** por el diseño experimental. La brecha entre el desempeño original (con fuga, proxy sesgado y confound de inputs) y el corregido será un hallazgo interesante, pero el benchmark debe tratar la comparación como **no concluyente** hasta que se realice el ablation faltante.