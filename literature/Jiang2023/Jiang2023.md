---
id: Jiang2023
title: "Forecasting realized volatility of Chinese crude oil futures with a new secondary decomposition ensemble learning approach"
authors: "Wei Jiang; Wanqing Tang; Xiao Liu"
year: 2023
doi: "10.1016/j.frl.2023.104254"
journal: "Finance Research Letters"
editorial_quality: "Q1 (Elsevier)"
status: "Include"
exclusion_reason: "None"
problem_type: "Point forecast"
horizon: "multi-step (1, 5, 20)"
proxy: "Realized volatility (RV) from 5-min intraday returns"
architectural_family: "Hybrid (Decomposition + DL)"
architectural_variant: "VMD-ICEEMDAN-LSTM (descomposición secundaria del residuo VMD + LSTM por componente + suma)"
market_asset: "Chinese crude oil futures (INE)"
---

# Resumen
El estudio propone un modelo de descomposición secundaria y ensamblaje (**VMD-ICEEMDAN-LSTM**) para pronosticar la volatilidad realizada (RV) de los futuros de crudo de China (INE). La serie original de RV se descompone primero con Variational Mode Decomposition (VMD). Luego, el término residual (que aún presenta alta complejidad) se descompone nuevamente utilizando Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (ICEEMDAN). Finalmente, cada subsecuencia se introduce en un modelo LSTM independiente y los resultados se suman para obtener el pronóstico final. Los resultados reportan que VMD-ICEEMDAN-LSTM supera significativamente a LSTM puro, ICEEMDAN-LSTM, VMD-LSTM, VMD-EEMD-LSTM, HAR y ARIMA, con MCS confirmando la superioridad estadística en todos los horizontes. **El estudio se incluye porque es Q1 (Finance Research Letters), predice RV de alta frecuencia, usa partición temporal y pruebas de significancia formales (MCS), y aporta la familia `Hybrid (Decomposition + DL)` con descomposición secundaria. Sin embargo, presenta limitaciones severas: (i) look-ahead bias confirmado por descomposición global sobre la serie completa antes del split; (ii) confound por número de LSTMs — VMD-ICEEMDAN usa ~17 LSTMs, VMD usa ~9, LSTM usa 1 —, por lo que la comparación VMD vs VMD-ICEEMDAN no aísla el efecto de la descomposición secundaria del efecto de tener más modelos; (iii) anomalía numérica en la Tabla 5 (VMD-EEMD tiene MSE 10× peor que VMD-LSTM a H=5, contradiciendo la narrativa); (iv) sin tuning de hiperparámetros de la arquitectura LSTM; (v) K y α de VMD fijados sin justificación; (vi) sin código.** La comparación **entre las 5 variantes de descomposición es relativamente justa** (todas parten de la misma serie RV y tienen el mismo look-ahead bias), pero **contra LSTM puro, HAR y ARIMA está contaminada** por la fuga, y la comparación **VMD vs VMD-ICEEMDAN está confundida** por el número de LSTMs.

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **Volatilidad Realizada (RV) diaria** de los futuros de crudo de China (INE), calculada sumando los cuadrados de los retornos intradía de 5 minutos (Ecuación 1):
$$RV(t) = \sum_{j=1}^{M} r_{t,j}^2$$
donde $M$ es la frecuencia de muestreo (5 minutos). Se evalúan horizontes de 1, 5 y 20 pasos.

**Sobre la validez del target:** **SÍ predicen Volatilidad Realizada (RV) de alta frecuencia.** El target es genuino y comparable con la literatura de RV. Sin embargo, el paper **no especifica** si usaron retornos logarítmicos o simples, ni si aplicaron estimadores robustos para la microestructura (como Realized Kernel o Two-Scale RV) antes de la suma. Con datos de 5 minutos, la microestructura puede introducir ruido; los estimadores robustos son estándar en la literatura. Esta es una **ambigüedad metodológica menor**, no un sesgo del target.

**Fuente de los datos:**
- **Precios intradía:** futuros de crudo de China (INE), base de datos Tushare. Periodo: 26 de marzo de 2018 – 31 de enero de 2023. Muestreo de 5 minutos.
- **Data availability:** "Data will be made available on request". Los datos de futuros de China son accesibles vía Tushare (de pago) o fuentes públicas del exchange (INE).

**Partición Temporal:**
- **Split 7:2:1 (train/test/prediction).** Cronológico. **No se reportan las fechas exactas** de corte, solo la proporción.
- **Rolling time prediction** para el conjunto de predicción. El texto dice: "we employ the rolling time prediction method to test the predictive performance". **No se especifica el tamaño de la ventana móvil** ni si hay re-entrenamiento de la LSTM en cada paso.
- **No hay validation set separado:** el split 7:2:1 es train/test/prediction, no train/val/test. El "test" (20%) se usa para evaluar el desempeño out-of-sample, y el "prediction" (10%) para las predicciones finales. **No se reporta un conjunto de validación para tuning**.

**LOOK-AHEAD BIAS CONFIRMADO (descomposición global):**
La Figura 2 (flujo del modelo) y la Sección 3.2 muestran que los algoritmos de descomposición (VMD e ICEEMDAN) se aplican a la **serie RV original completa (2018-2023)** antes de dividir en train/test/prediction. Luego, los IMFs y el residuo se dividen en las tres particiones. **Esto es look-ahead bias confirmado:** los IMFs calculados para el train set contienen información matemática del test set. El modelo "ve" la forma de la serie futura a través de los modos descompuestos.

**Consecuencia para la comparación:**
- **Entre las 5 variantes de descomposición** (LSTM, ICEEMDAN-LSTM, VMD-LSTM, VMD-EEMD-LSTM, VMD-ICEEMDAN-LSTM): la comparación es **relativamente justa** en el sentido de que todas parten de la misma serie RV y todas tienen el mismo look-ahead bias. La diferencia entre ellas es atribuible a la elección de la descomposición, **salvo por el confound del número de LSTMs** (ver abajo).
- **Contra LSTM puro:** **contaminada**. El LSTM puro NO tiene descomposición, por lo tanto NO tiene la fuga. La comparación VMD-ICEEMDAN-LSTM vs LSTM puro mide, en parte, el efecto de la fuga, no solo el efecto de la descomposición.
- **Contra HAR y ARIMA:** **contaminada** por la misma razón.

**CONFOUND POR NÚMERO DE LSTMs:**
El modelo VMD-ICEEMDAN-LSTM entrena **un LSTM por cada componente**:
- VMD: 8 IMFs + 1 residuo = 9 componentes → 9 LSTMs.
- ICEEMDAN (sobre el residuo de VMD): 8 IMFs + 1 residuo = 9 componentes → 9 LSTMs.
- **Total VMD-ICEEMDAN-LSTM: ~17 LSTMs.**
- **Total VMD-LSTM: 9 LSTMs.**
- **Total ICEEMDAN-LSTM: 9 LSTMs.**
- **Total LSTM puro: 1 LSTM.**

**Esto confunde la comparación VMD-LSTM vs VMD-ICEEMDAN-LSTM:** el segundo tiene ~2× el número de LSTMs del primero. La mejora podría deberse a la **descomposición secundaria** o simplemente a **tener más modelos** (más capacidad de ajuste). El paper no incluye un control con "VMD + 8 LSTMs adicionales sobre los mismos IMFs" (ensemble de LSTMs por IMF) para aislar el efecto. **La afirmación "la descomposición secundaria del residuo mejora la precisión" no está completamente identificada** sin ese control.

**Normalización:**
**No se reporta** qué tipo de escalado usaron ni si fue ajustado solo con el train set. **Ambigüedad con riesgo potencial de fuga.**

**Ingeniería de Características:**
- **Input:** la serie RV descompuesta en componentes. Cada componente se introduce en un LSTM independiente.
- **Sin variables exógenas:** no hay features de noticias, sentimiento, redes complejas ni otras variables financieras.
- **Sin selección de features:** se usan todos los IMFs + residuo.
- **VMD:** K=8 determinado por el método de frecuencia central (Tabla 2). **No se reporta el valor de α** (penalización cuadrática).
- **ICEEMDAN:** **no se reportan sus hiperparámetros** (número de realizaciones de ruido, amplitud del ruido, número de IMFs).

**Tuning de Hiperparámetros:**
- **Arquitectura LSTM:** **NO se reporta** el número de capas ni neuronas ocultas. Solo se reportan **window** y **epoch** por componente (Tabla 4), que varían entre componentes (window ∈ {2,3,4,5}, epoch ∈ {80,90,120,180,240,300,360,420,440,460}).
- **Optimizador:** **NO reportado explícitamente.** Learning rate = 0.001, batch_size = 32 (justificado citando a Keskar et al. 2016).
- **Búsqueda de hiperparámetros:** el paper dice "optimal hyperparameters were selected through experimentation" — **no hay Grid Search, Bayesian Optimization ni validación cruzada**. La selección de window y epoch por componente es **ad hoc**, no sistemática.
- **K de VMD:** 8, por frecuencia central. **α no reportado.**

# 2. Arquitectura Final (VMD-ICEEMDAN-LSTM)

El modelo propuesto, denominado **VMD-ICEEMDAN-LSTM**, tiene la siguiente estructura:

1. **Descomposición VMD:** la serie RV original se descompone en **8 IMFs y 1 residuo** (K=8, por frecuencia central).
2. **Descomposición secundaria ICEEMDAN:** el residuo de VMD se descompone nuevamente en **8 IMFs y 1 residuo** mediante ICEEMDAN.
3. **LSTM por componente:** cada uno de los ~17 componentes (8 IMFs VMD + 8 IMFs ICEEMDAN + 1 residuo ICEEMDAN) se introduce en un **LSTM independiente**. **No se reporta la arquitectura interna** (número de capas, neuronas, dropout).
4. **Reconstrucción:** las predicciones de cada LSTM se suman para obtener el pronóstico final de RV.

**Arquitectura interna de la LSTM:**
- Learning rate 0.001.
- Batch size 32.
- **Número de capas y neuronas ocultas: NO reportado.**
- **Función de pérdida: NO reportada.**
- **Dropout: NO reportado.**
- **Early stopping: NO reportado.**
- **Window y epoch por componente:** reportados en Tabla 4. Ejemplo: IMF1 de VMD (window=3, epoch=420), IMF2 (window=5, epoch=420), ..., Res de VMD (window=2, epoch=80 en ICEEMDAN).

**Ablation del componente de descomposición (parcial):**
- **Sí ablate la elección del método de descomposición:** compara LSTM puro, ICEEMDAN-LSTM, VMD-LSTM, VMD-EEMD-LSTM y VMD-ICEEMDAN-LSTM. **Esto es el ablation central del paper y está bien cubierto en cuanto a la elección del método.**
- **No controla por número de LSTMs:** VMD-ICEEMDAN usa ~17 LSTMs, VMD usa 9, LSTM puro usa 1. La mejora de VMD-ICEEMDAN sobre VMD **no se puede atribuir solo a la descomposición secundaria** porque también tiene más modelos. Falta un control tipo "VMD + ensemble de LSTMs por IMF" (mismo número de LSTMs que VMD-ICEEMDAN, sin descomposición secundaria).
- **No ablate el número de IMFs (K) de VMD:** K=8 se fija sin análisis de sensibilidad.
- **No ablate el valor de α de VMD:** el parámetro de penalización cuadrática se fija sin justificación.
- **No ablate los hiperparámetros de ICEEMDAN:** número de realizaciones de ruido, amplitud del ruido, número de IMFs. Se fijan sin justificación.
- **No ablate la profundidad de la descomposición secundaria:** ¿una descomposición terciaria del residuo de ICEEMDAN mejoraría aún más? ¿Dónde se detiene la mejora? El paper no lo explora.

# 3. Resultados y Comparaciones

**Modelos comparados (7 en total):**
- **Baselines sin descomposición:** LSTM puro, HAR, ARIMA.
- **Descomposición simple:** ICEEMDAN-LSTM (solo ICEEMDAN), VMD-LSTM (solo VMD).
- **Descomposición secundaria:** VMD-EEMD-LSTM (VMD + EEMD sobre el residuo), VMD-ICEEMDAN-LSTM (VMD + ICEEMDAN sobre el residuo).

**Métricas de Evaluación:**
MAE, MSE, HMAE (Heteroskedasticity-adjusted MAE), HMSE (Heteroskedasticity-adjusted MSE).

**Ganador y Rendimiento:**
El modelo **VMD-ICEEMDAN-LSTM** es el ganador en todos los horizontes.
- **1 paso:** MAE = $7.23 \times 10^{-3}$, MSE = $1.05 \times 10^{-4}$, HMAE = 0.438, HMSE = 0.272.
- **5 pasos:** MAE = $2.25 \times 10^{-3}$, MSE = $7.37 \times 10^{-6}$, HMAE = 0.191, HMSE = 0.079.
- **20 pasos:** MAE = $1.70 \times 10^{-3}$, MSE = $4.11 \times 10^{-6}$, HMAE = 0.252, HMSE = 0.312.

**Reducciones reportadas (vs VMD-LSTM, el competidor más cercano):**
- **1 paso:** MAE 7.48→7.23 ($\approx$ 3.3%), MSE 1.10→1.05 ($\approx$ 4.5%), HMAE 0.494→0.438 (11.3%), HMSE 0.672→0.272 (59.5%).
- **5 pasos:** MAE 2.65→2.25 (15.1%), MSE 1.07→0.737 (31.1%), HMAE 0.231→0.191 (17.3%), HMSE 0.108→0.079 (26.9%).
- **20 pasos:** MAE 2.30→1.70 (26.1%), MSE 0.843→0.411 (51.2%), HMAE 0.429→0.252 (41.3%), HMSE 1.808→0.312 (82.7%).

**ANOMALÍA NUMÉRICA EN LA TABLA 5:**
A $H=5$, **VMD-EEMD-LSTM tiene MSE = $7.68 \times 10^{-5}$, que es 10× peor que VMD-LSTM ($1.07 \times 10^{-5}$)** y 10× peor que VMD-ICEEMDAN-LSTM ($7.37 \times 10^{-6}$). Esto **contradice la narrativa del paper** de que la descomposición secundaria siempre mejora: en este caso, VMD-EEMD es **peor que VMD solo** en MSE a $H=5$. El paper no discute esta anomalía. En las otras métricas a $H=5$, VMD-EEMD es mejor que VMD-LSTM (MAE 2.28 vs 2.65, HMAE 0.191 vs 0.231, HMSE 0.080 vs 0.108), lo que hace la anomalía del MSE aún más sospechosa. **Posible typo o error de cómputo.**

**TYPO EN HAR (H=20):**
En la Tabla 5, HAR a $H=20$ tiene MAE = $1.42 \times 10^{-3}$, mientras que a $H=1$ es $1.34 \times 10^{-2}$ y a $H=5$ es $1.34 \times 10^{-2}$. El exponente $10^{-3}$ es inconsistente con los otros horizontes y con el resto de la fila. **Probablemente debería ser $1.42 \times 10^{-2}$.**

**Pruebas de Significancia:**
- **Model Confidence Set (MCS):** VMD-ICEEMDAN-LSTM obtiene p-valor = 1.000 en todas las métricas y horizontes (Tabla 6). En $H=1$, VMD-EEMD-LSTM también pasa parcialmente (p=0.097 en MAE, 0.072 en MSE, 0.020 en HMAE, 0.141 en HMSE). En $H=5$, VMD-EEMD tiene p=0.684 (MAE), 0.231 (MSE), 0.930 (HMAE), 0.844 (HMSE). En $H=20$, VMD-EEMD tiene p=0.540 (MAE), 0.101 (MSE), 0.039 (HMAE), 0.027 (HMSE).
- **No se reporta Wilcoxon signed-rank test** en el paper principal (a diferencia de Jiang2024, que sí lo incluye).

**Sesgo de Reporte:**
Alto. Ocho problemas:
1. **Look-ahead bias confirmado:** descomposición global sobre la serie completa antes del split. Contamina la comparación contra LSTM puro, HAR y ARIMA.
2. **Confound por número de LSTMs:** VMD-ICEEMDAN usa ~17 LSTMs, VMD usa 9, LSTM puro usa 1. La mejora de VMD-ICEEMDAN sobre VMD no se puede atribuir solo a la descomposición secundaria. Falta un control con "VMD + ensemble de LSTMs por IMF".
3. **Anomalía numérica en la Tabla 5:** VMD-EEMD tiene MSE 10× peor que VMD-LSTM a $H=5$, contradiciendo la narrativa.
4. **Typo en HAR (H=20):** MAE = $1.42 \times 10^{-3}$ en lugar de $10^{-2}$.
5. **Sin tuning de hiperparámetros de la arquitectura LSTM:** solo window y epoch por componente, seleccionados ad hoc. Sin capas, neuronas, dropout ni función de pérdida.
6. **K y α de VMD fijados sin justificación:** K=8 por frecuencia central, α no reportado.
7. **Hiperparámetros de ICEEMDAN no reportados:** número de realizaciones de ruido, amplitud del ruido, número de IMFs.
8. **Sin varianza entre ejecuciones:** la LSTM es estocástica, pero no se reporta la dispersión.

**Fortalezas del reporte:**
- Target genuino de RV de alta frecuencia.
- Comparación de 5 variantes de descomposición bajo el mismo protocolo (inputs iguales).
- Evaluación multi-horizonte (1, 5, 20 pasos).
- MCS con Range y Semi-quadratic statistics.
- Robustez con 3 escenarios: 1-min frequency, 6:2:2 ratio, exclusión de COVID-19.
- Diagnóstico de los IMFs con Ljung-Box (correlación serial significativa al 1%).
- Análisis de frecuencia central para K de VMD (Tabla 2).
- Reporte de window y epoch por componente (Tabla 4).
- Reconocimiento explícito de las limitaciones (sin variables exógenas, tuning limitado).

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número total de parámetros de las LSTM.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** "Data will be made available on request". Tushare es de pago, pero los futuros de China son accesibles vía fuentes públicas del exchange (INE).
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast).
- **Limitaciones reportadas:** los autores mencionan que futuras investigaciones deberían integrar más variables (como factores climáticos) y optimizar los hiperparámetros de la red.

## 5.2 Hiperparámetros reportados (valores finales)
- **LSTM:** learning rate 0.001; batch_size 32. **NO se reportan** número de capas, neuronas ocultas, dropout, función de pérdida, early stopping.
- **VMD:** K=8 (frecuencia central). **NO se reporta α** (penalización cuadrática) ni el criterio de parada.
- **ICEEMDAN:** **NO se reportan** número de realizaciones de ruido, amplitud del ruido, número de IMFs.
- **Ventanas y épocas por componente:** reportadas en Tabla 4 (window ∈ {2,3,4,5}, epoch ∈ {80,90,120,180,240,300,360,420,440,460}).
- **Normalización:** **NO reportada.**
- **Validation set:** **NO reportado** (el split es train/test/prediction, no train/val/test).
- **Semilla:** **NO reportada.**
- **HAR:** **NO se reportan los lags** (diario, semanal, mensual) ni el período de estimación.
- **ARIMA:** **NO se reportan los órdenes (p,d,q)** ni el criterio de selección.

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **Ninguno sistemático.** "Optimal hyperparameters were selected through experimentation" — selección ad hoc por componente. |
| Espacio de búsqueda | **NO reportado.** Solo se reportan los valores finales (Tabla 4). |
| Valores finales | window ∈ {2,3,4,5}, epoch ∈ {80,90,120,180,240,300,360,420,440,460}. lr=0.001, batch_size=32. **Capas, neuronas y dropout no reportados.** |
| Optimizador | **NO reportado explícitamente.** |
| Learning rate | 0.001 |
| Batch size | 32 |
| Épocas | 80-460 (por componente) |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No reportado.** El split es train/test/prediction, sin validation set separado. La selección de window y epoch por componente se hace "through experimentation", pero no se documenta el criterio ni el conjunto sobre el que se evalúa.

**¿Sobre qué conjunto se evalúa el fitness?** **NO reportado.** No hay conjunto de validación separado ni criterio de selección declarado.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **NO reportado.** Se asume MSE o MAE sobre cada IMF, pero no se declara.

### 5.3.4 Modelos de referencia y descomposiciones
- **HAR:** lags diario/semanal/mensual asumidos (no reportados). **No se reporta el período de estimación** (rolling vs estático).
- **ARIMA:** órdenes (p,d,q) **no reportados**. **No se reporta el criterio de selección** (AIC, BIC).
- **ICEEMDAN:** hiperparámetros **no reportados** (número de realizaciones de ruido, amplitud del ruido, número de IMFs).
- **VMD:** K=8 por frecuencia central; α **no reportado**.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura VMD-ICEEMDAN-LSTM (flujo) | Sí (descrita en Fig. 2) | Bajo: está descrita |
| Capas y neuronas de la LSTM | No | Alto: no se puede replicar la red |
| Función de pérdida | No | Alto: no se sabe qué se minimiza |
| Dropout, early stopping | No | Medio: afectan la regularización |
| K de VMD | Sí (8, por frecuencia central) | Bajo: está reportado |
| α de VMD | No | Medio: afecta la descomposición |
| Hiperparámetros de ICEEMDAN | No | Medio: afectan la descomposición secundaria |
| Window y epoch por componente | Sí (Tabla 4) | Bajo: están reportados |
| Normalización (tipo y timing) | No | Medio: potencial fuga |
| Validation set | No | Alto: no se sabe si hay fuga por tuning |
| Órdenes de ARIMA y lags de HAR | No | Medio: afectan los baselines |
| Semilla | No | Medio: la LSTM es estocástica |
| Anomalía numérica en Tabla 5 | Posible typo | **Alto: impide verificar los resultados** |
| Typo en HAR (H=20) | Posible typo | Medio: afecta la verificación |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Parcialmente, con un riesgo de divergencia alto.** Los supuestos que habría que tomar:
1. **LSTM:** 1-2 capas, 50-64 neuronas, dropout 0.2, early stopping con paciencia 10, función de pérdida MSE.
2. **VMD:** K=8, α=2000 (valor típico), criterio de parada por convergencia.
3. **ICEEMDAN:** 100-500 realizaciones de ruido, amplitud 0.2 desviaciones estándar, 8-10 IMFs.
4. **Normalización:** Min-Max ajustado solo con train.
5. **Protocolo:** rolling window con re-entrenamiento periódico o estático, pero **homogéneo** para todos los modelos.
6. **Semilla:** fijar una semilla.
7. **HAR:** lags 1, 5, 22 (diario, semanal, mensual).
8. **ARIMA:** selección automática por AIC.

**Riesgo de divergencia del desempeño reportado:** **muy alto.** El desempeño reportado está inflado por: (i) look-ahead bias por descomposición global; (ii) confound por número de LSTMs; (iii) sin tuning de la arquitectura LSTM; (iv) anomalía numérica en la tabla. Sin la fuga y con un control por número de LSTMs, el desempeño sería sustancialmente menor y la comparación VMD vs VMD-ICEEMDAN podría no ser significativa.

**Riesgo de divergencia de la arquitectura:** **alto.** La arquitectura LSTM no está reportada (capas, neuronas, dropout, función de pérdida).

**Riesgo de divergencia del pipeline:** **alto.** Faltan el timing de normalización, el α de VMD, los hiperparámetros de ICEEMDAN, el uso del validation set y los números corruptos de la tabla.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas —look-ahead bias por descomposición global, confound por número de LSTMs, arquitectura LSTM no reportada, función de pérdida no declarada, sin tuning sistemático, K y α de VMD no justificados, hiperparámetros de ICEEMDAN no reportados, normalización no declarada, sin validation set, anomalía numérica en la Tabla 5, typo en HAR (H=20) y ausencia de código—, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no una réplica exacta. El benchmark debe tratar la versión corregida (con descomposición rolling y control por número de LSTMs) como una **variante** del modelo original y reportar la brecha entre ambos desempeños.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de RV de alta frecuencia con LSTM como componente central.
- **C2. Reporte suficiente:** **Parcial.** Reporta el flujo del modelo, K=8 para VMD, window y epoch por componente, learning rate y batch_size, pero no la arquitectura LSTM, la función de pérdida, el α de VMD, los hiperparámetros de ICEEMDAN, la normalización, el validation set ni el código. **La anomalía numérica en la Tabla 5 y el typo en HAR impiden verificar los resultados numéricos.**
- **C3. Calidad editorial:** **Sí.** Q1 en JCR (Finance Research Letters, Elsevier).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El protocolo es reconstruible con supuestos (partición temporal documentada, arquitectura descrita, K reportado). La fuga por descomposición global es un problema de validez, no de reconstruibilidad. Se incluye con reservas.
- **E6. Solo precio, sin volatilidad:** **No aplica.** Predice RV de 5 minutos.
- **E7. GARCH como contribución principal:** **No aplica.** HAR y ARIMA son baselines, no la contribución principal.

**Resultado:** **Incluido** (con reservas).

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** RV de 5 minutos, medida estándar de alta frecuencia. **Sin sesgo del target** (aunque no se reporta si usan retornos logarítmicos o simples ni estimadores robustos). → **bajo**.
- **Partición:** split 7:2:1 cronológico, pero **sin fechas exactas** y **sin validation set separado**. → **moderado**.
- **Predicción:** rolling time prediction para OOS, pero **sin tamaño de ventana ni re-entrenamiento declarado**. → **moderado**.
- **Fuga:** **look-ahead bias CONFIRMADO** por descomposición global sobre la serie completa antes del split. Los IMFs del train contienen información del test. → **alto**.
- **Reporte selectivo:** reporta 7 modelos, 3 horizontes, 4 métricas, MCS, y 3 escenarios de robustez. Pero **sin varianza entre ejecuciones**, con **anomalía numérica en Tabla 5** y **typo en HAR (H=20)**. → **moderado-alto**.
- **Inferencia:** **MCS con Range y Semi-quadratic** formales. **Bien.** Pero las comparaciones están contaminadas por la fuga y el confound. → **moderado**.
- **Comparación:** **inputs iguales entre las 5 variantes de descomposición** (todas parten de la misma serie RV) → **relativamente justa** para el ablation central, **salvo por el confound del número de LSTMs** (VMD-ICEEMDAN usa ~17, VMD usa 9, LSTM puro usa 1). **Contra LSTM puro, HAR y ARIMA** → **contaminada** porque solo los modelos con descomposición tienen la fuga. → **alto**.
- **Ablation:** se comparan **5 variantes de descomposición** → bien cubierto para la elección del método. Pero **no se varía K ni α de VMD**, **no se controla por número de LSTMs**, **no se varía la profundidad de la descomposición secundaria**, y **no se reportan los hiperparámetros de ICEEMDAN**. → **moderado-alto**.
- **Reproducibilidad:** sin arquitectura LSTM, sin función de pérdida, sin α de VMD, sin hiperparámetros de ICEEMDAN, sin normalización, sin validation set, sin código, con anomalía numérica y typo. → **muy parcial**.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** Los futuros de crudo de China son accesibles vía Tushare (de pago) o fuentes públicas del exchange (INE). Los retornos de 5 minutos se pueden construir. → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** Flujo descrito, K=8 reportado y window/epoch por componente reportados, pero faltan arquitectura LSTM, función de pérdida, α de VMD, hiperparámetros de ICEEMDAN y normalización. → **Parcial**.
- **Viabilidad computacional:** **sí.** VMD con K=8 + ICEEMDAN con ~9 componentes + ~17 LSTM, 3 horizontes. Viable en CPU/GPU estándar. → **Sí**.

**Clasificación: R-B (no reproducible por el equipo).** R1=Sí, R2=Tal vez, R3=Sí → no cumple R2=Sí. El estudio se conserva en la revisión como referencia metodológica para la familia `Hybrid (Decomposition + DL)` con descomposición secundaria. Si el equipo decide reimplementarlo con **descomposición rolling** (corrigiendo la fuga) y **control por número de LSTMs**, entra al benchmark como **variante corregida**.

**Decisión provisional:**
- **Entra al benchmark:** sí, con reservas (si se reimplementa con descomposición rolling y control por número de LSTMs).
- **Tipo de entrada:** **con corrección**.
- **Correcciones necesarias:**
    - **(PRIORITARIA) Corregir el look-ahead bias por descomposición global:** aplicar la descomposición de forma **rolling** (recalcular los IMFs en cada origen de pronóstico usando solo datos pasados) o **solo sobre el train set**. Sin esta corrección, el desempeño reportado no es comparable con el de los modelos que no tienen fuga, y la comparación contra LSTM puro, HAR y ARIMA está contaminada.
    - **(PRIORITARIA) Controlar por número de LSTMs:** incluir un modelo de control "VMD + ensemble de LSTMs por IMF" (mismo número de LSTMs que VMD-ICEEMDAN-LSTM, sin descomposición secundaria) para aislar el efecto de la descomposición secundaria del efecto de tener más modelos. Sin este control, la afirmación "la descomposición secundaria del residuo mejora la precisión" **no está identificada**.
    - **(PRIORITARIA) Reportar la arquitectura LSTM:** número de capas, neuronas ocultas, dropout, función de pérdida, early stopping y optimizador.
    - **(PRIORITARIA) Reportar el α de VMD** (penalización cuadrática) y justificar K=8 con un análisis de sensibilidad (variar K y ver el desempeño).
    - **Reportar los hiperparámetros de ICEEMDAN:** número de realizaciones de ruido, amplitud del ruido, número de IMFs.
    - **Reportar el criterio de selección de window y epoch por componente:** ¿sobre qué conjunto se evalúa? ¿Hay validation set?
    - **Reportar la normalización:** tipo (Min-Max, Z-score) y timing (solo train o dataset completo).
    - **Reportar las fechas exactas del split** (train/test/prediction) y el tamaño de la ventana móvil.
    - **Corregir la anomalía numérica en la Tabla 5:** VMD-EEMD tiene MSE 10× peor que VMD-LSTM a $H=5$, contradiciendo la narrativa. Verificar el valor.
    - **Corregir el typo en HAR (H=20):** MAE = $1.42 \times 10^{-3}$ en lugar de $10^{-2}$.
    - **Reportar la varianza/desviación estándar** entre ejecuciones.
    - **Reportar los órdenes de ARIMA** y los lags de HAR.
    - **Reportar la semilla** para la LSTM.
    - **Añadir QLIKE** como métrica de evaluación de volatilidad, además de MAE, MSE, HMAE y HMSE.
    - **Liberar el código** o, en su defecto, reportar todos los valores finales de hiperparámetros y la semilla.
    
- **Brecha principal:** El estudio aporta la familia `Hybrid (Decomposition + DL)` con descomposición secundaria del residuo (VMD + ICEEMDAN) y una comparación sistemática de 5 variantes de descomposición bajo el mismo protocolo. La pregunta central que aborda —¿la descomposición secundaria del residuo de VMD mejora la precisión frente a una sola descomposición?— es relevante para el benchmark porque ninguna otra entrada del corpus la responde. **La brecha es de identificación, no de reporte:** el diseño experimental confunde la descomposición secundaria con tener ~2× más LSTMs (VMD-ICEEMDAN usa ~17, VMD usa 9, LSTM puro usa 1), de modo que la mejora de VMD-ICEEMDAN sobre VMD no se puede atribuir a la descomposición secundaria. El benchmark debe reimplementar el modelo con **descomposición rolling** (para eliminar el look-ahead bias) y **un control con "VMD + ensemble de LSTMs por IMF"** (mismo número de LSTMs, sin descomposición secundaria). Sin ese control, la afirmación central del paper no está identificada y la comparación debe tratarse como **no concluyente**. Las demás carencias (arquitectura LSTM, tuning, K/α, hiperparámetros de ICEEMDAN, normalización, anomalía en Tabla 5) son deficiencias de reporte que afectan la reproducibilidad, pero no la identificación del efecto; se corrigen en la reimplementación y se documentan en la Sección 8.