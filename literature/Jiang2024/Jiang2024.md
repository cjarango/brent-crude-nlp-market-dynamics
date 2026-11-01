---
id: Jiang2024
title: "China Crude Oil Futures Volatility Forecasting Using LSTM Model with Optimal Noise Decomposition."
authors: "Wei Jiang; Wanqing Tang; Huizhi Liu; Yilin Zhou; Xiao Liu"
year: 2024
doi: "10.1155/2024/8021444"
journal: "Discrete Dynamics in Nature and Society"
editorial_quality: "Q3 (Wiley)"
status: "Exclude"
exclusion_reason: "C3: Revista Q3 (Discrete Dynamics in Nature and Society), no fundacional. El paper no introduce una arquitectura nueva ni es un trabajo de referencia citado como comparador en un estudio 2020–2026. Es un estudio comparativo de 6 técnicas de descomposición existentes aplicadas a LSTM. No cumple C3. Consistente con la exclusión de Verma2021 (Q3, no fundacional)."
problem_type: "Point forecast"
horizon: "multi-step (1, 5, 20)"
proxy: "Realized volatility (RV) from 5-min intraday returns"
architectural_family: "Hybrid (Decomposition + DL)"
architectural_variant: "VMD-LSTM (VMD con K=8 + LSTM por IMF + suma)"
market_asset: "Chinese crude oil futures (INE)"
---

# Resumen
El estudio evalúa el impacto de seis algoritmos de descomposición de señales (EMD, EEMD, CEEMDAN, ICEEMDAN, VMD, EWT) combinados con una red LSTM para pronosticar la volatilidad realizada (RV) de los futuros de crudo de China (INE). Los resultados reportan que el modelo **VMD-LSTM** supera significativamente a los modelos tradicionales (ARIMA, HAR) y al LSTM puro, con MCS y Wilcoxon confirmando la superioridad estadística. **El estudio se excluye del corpus por C3: la revista es Q3 (Discrete Dynamics in Nature and Society, Wiley) y el paper no es fundacional — no introduce una arquitectura nueva ni es un trabajo de referencia citado como comparador en un estudio 2020–2026. Es un estudio comparativo de 6 técnicas de descomposición existentes aplicadas a LSTM. La exclusión es consistente con la de Verma2021 (Q3, no fundacional).** Aunque el paper tiene valor metodológico (comparación sistemática de 6 descomposiciones bajo el mismo protocolo y ejemplo claro de look-ahead bias por descomposición global), ese valor se registra en la revisión pero **no justifica la inclusión bajo C3 tal como está escrito**. El análisis técnico completo se conserva para el registro.

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **Volatilidad Realizada (RV) diaria** de los futuros de crudo de China (INE), calculada sumando los cuadrados de los retornos intradía de 5 minutos (Ecuación 1):
$$RV(t) = \sum_{j=1}^{M} r_{t,j}^2$$
donde $M$ es la frecuencia de muestreo (5 minutos). Se evalúan horizontes de 1, 5 y 20 pasos.

**Sobre la validez del target:** **SÍ predicen Volatilidad Realizada (RV) de alta frecuencia.** El target es genuino y comparable con la literatura de RV. Sin embargo, el paper **no especifica** si usaron retornos logarítmicos o simples, ni si aplicaron estimadores robustos para la microestructura (como Realized Kernel o Two-Scale RV) antes de la suma. Con datos de 5 minutos, la microestructura puede introducir ruido; los estimadores robustos son estándar en la literatura. Esta es una **ambigüedad metodológica menor**, no un sesgo del target.

**Fuente de los datos:**
- **Precios intradía:** futuros de crudo de China (INE), base de datos Tushare. Periodo: 26 de marzo de 2018 – 31 de enero de 2023. Muestreo de 5 minutos.
- **Data availability:** "The data used to support the findings of this study will be made available on request". Los datos de futuros de China son accesibles vía Tushare (de pago) o fuentes públicas del exchange (INE).

**Partición Temporal:**
- **Split 7:2:1 (train/validation/test).** Cronológico. **No se reportan las fechas exactas** de corte, solo la proporción.
- **Rolling window para el pronóstico out-of-sample.** El texto dice: "the study utilizes an out-of-sample prediction approach based on rolling windows. This method enables predictions for all time points using the most up-to-date data". **No se especifica el tamaño de la ventana móvil** ni si hay re-entrenamiento de la LSTM en cada paso.
- **El validation set (20%) se usa para "evaluar la generalización"**, pero **no se reporta cómo se usa** (¿early stopping? ¿selección de hiperparámetros? ¿solo monitoreo?). No hay evidencia de que se use para tuning.

**LOOK-AHEAD BIAS CONFIRMADO (descomposición global):**
La Figura 1 (flujo del modelo) y la Sección 3.2 muestran que los 6 algoritmos de descomposición se aplican a la **serie RV original completa (2018-2023)** antes de dividir en train/val/test. Luego, los IMFs y el residuo se dividen en las tres particiones. **Esto es look-ahead bias confirmado:** los IMFs calculados para el train set contienen información matemática del test set. El modelo "ve" la forma de la serie futura a través de los modos descompuestos.

**Consecuencia para la comparación:**
- **Entre las 6 descomposiciones:** la comparación es **justa en inputs** (todas parten de la misma serie RV) y **todas tienen el mismo look-ahead bias**. La comparación interna es válida.
- **Contra LSTM puro:** **contaminada**. El LSTM puro NO tiene descomposición, por lo tanto NO tiene la fuga. La comparación VMD-LSTM vs LSTM puro mide, en parte, el efecto de la fuga, no solo el efecto de la descomposición.
- **Contra HAR y ARIMA:** **contaminada** por la misma razón. HAR y ARIMA no usan descomposición, por lo que no tienen la fuga.

**Normalización:**
**No se reporta** qué tipo de escalado usaron ni si fue ajustado solo con el train set. **Ambigüedad con riesgo potencial de fuga.**

**Ingeniería de Características:**
- **Input:** la serie RV descompuesta en $n$ IMFs + 1 residuo. Cada componente se introduce en un LSTM independiente.
- **Sin variables exógenas:** no hay features de noticias, sentimiento, redes complejas ni otras variables financieras.
- **Sin selección de features:** se usan todos los IMFs + residuo.
- **VMD:** K=8 determinado por el método de frecuencia central (Tabla 2). **No se reporta el valor de α** (penalización cuadrática).

**Tuning de Hiperparámetros:**
**CERO optimización.** Configuración LSTM fija: learning rate = 0.001, batch_size = 32. **No se reporta** el número de capas ni neuronas ocultas. **No hay Grid Search, Bayesian Optimization ni validación cruzada.** Los autores lo reconocen como limitación: "we recognize that the selection of model architecture can significantly influence outcomes in practical applications, posing a potential limitation of our research". Para VMD, fijan K=8 con el método de frecuencia central.

# 2. Arquitectura Final (VMD-LSTM)

El modelo propuesto, denominado **VMD-LSTM**, tiene la siguiente estructura:

1. **Descomposición VMD:** la serie RV original se descompone en **8 IMFs y 1 residuo** (K=8, determinado por frecuencia central).
2. **LSTM por componente:** cada IMF y el residuo se introducen en un **LSTM independiente**. **No se reporta la arquitectura interna** (número de capas, neuronas ocultas, dropout).
3. **Reconstrucción:** las predicciones de cada LSTM se suman para obtener el pronóstico final de RV (Ecuación 24):
   $$\widehat{RV}(t) = \sum_{i=1}^{n} \widehat{IMF}_i(t) + \widehat{R}(t)$$

**Arquitectura interna de la LSTM:**
- Learning rate 0.001.
- Batch size 32.
- **Número de capas y neuronas ocultas: NO reportado.**
- **Función de pérdida: NO reportada.**
- **Dropout: NO reportado.**
- **Early stopping: NO reportado** (aunque hay un validation set, no se declara su uso).

**Ablation del componente de descomposición (parcial):**
- **Sí ablate la elección del método de descomposición:** compara 6 algoritmos (EMD, EEMD, CEEMDAN, ICEEMDAN, VMD, EWT) más LSTM puro. **Esto es el ablation central del paper y está bien cubierto.**
- **No ablate el número de IMFs (K):** K=8 se fija sin análisis de sensibilidad. El paper reconoce que "setting K too small may result in incomplete decomposition and mode mixing, while setting K too large may introduce irrelevant and spurious components", pero no varía K para verificar que 8 es óptimo.
- **No ablate el valor de α de VMD:** el parámetro de penalización cuadrática se fija sin justificación. Los autores lo reconocen como limitación: "meticulous consideration should be given to the selection of the penalty factor and the number K of modal decomposition for VMD during practical applications, as these choices can significantly influence the efficacy of VMD decomposition. Currently, an optimal parameter combination has not been identified."
- **No ablate "sumar predicciones de IMFs" vs "predecir RV directamente con todos los IMFs como features".** El paper usa la estrategia de suma (cada IMF a un LSTM, luego suma), pero no compara con la alternativa de concatenar todos los IMFs como features de un solo LSTM. **Esto deja abierta la pregunta de si el beneficio viene de la descomposición o de la estrategia de ensemble por suma.**

# 3. Resultados y Comparaciones

**Modelos comparados (9 en total):**
- **Baselines sin descomposición:** LSTM puro, HAR, ARIMA.
- **Descomposición + LSTM:** EMD-LSTM, EEMD-LSTM, CEEMDAN-LSTM, ICEEMDAN-LSTM, VMD-LSTM, EWT-LSTM.

**Métricas de Evaluación:**
MAE, MSE, HMAE (Heteroskedasticity-adjusted MAE).

**Ganador y Rendimiento:**
El modelo **VMD-LSTM** es el ganador en todos los horizontes.
- **1 paso:** MAE = $7.5 \times 10^{-2}$, MSE = $1.10 \times 10^{-4}$, HMAE = 0.423.
- **5 pasos:** MAE ≈ $3.66 \times 10^{-2}$ (valor corrupto en la tabla), MSE = $1.08 \times 10^{-5}$, HMAE = 0.140.
- **20 pasos:** MAE ≈ $3.56 \times 10^{-2}$ (valor corrupto en la tabla), MSE = $8.15 \times 10^{-6}$, HMAE = 0.106.

**Reducciones reportadas (vs LSTM puro, 1 paso):**
- VMD-LSTM: 34.58% MAE, 58.11% MSE, 46.10% HMAE.
- EWT-LSTM: 31.39% MAE, 55.68% MSE, 41.04% HMAE.
- EMD-LSTM: 17.71% MAE, 39.04% MSE, 32.75% HMAE.

**A 5 y 20 pasos, las mejoras del VMD-LSTM vs LSTM puro superan el 76% y 81% en MAE**, respectivamente.

**Pruebas de Significancia:**
- **Model Confidence Set (MCS):** VMD-LSTM obtiene p-valor = 1.000 en todas las métricas y horizontes. En $h=1$, EWT-LSTM también pasa parcialmente (p=0.292 en MAE, 0.150 en MSE, 0.002 en HMAE). En $h=5$ y $h=20$, solo VMD-LSTM pasa.
- **Wilcoxon signed-rank test (WSRT):** VMD-LSTM es estadísticamente mejor que EMD, EEMD, CEEMDAN e ICEEMDAN en las tres métricas. Contra EWT-LSTM, es equivalente en MAE y MSE, y mejor en HMAE.

**Sesgo de Reporte:**
Alto. Siete problemas:
1. **Look-ahead bias confirmado:** descomposición global sobre la serie completa antes del split. Contamina la comparación contra LSTM puro, HAR y ARIMA.
2. **Sin tuning de hiperparámetros:** configuración LSTM fija sin búsqueda. Los autores lo reconocen como limitación.
3. **Arquitectura LSTM no reportada:** sin capas, neuronas, dropout ni función de pérdida.
4. **K y α de VMD fijados sin justificación:** K=8 por frecuencia central, α no reportado. Los autores lo reconocen como limitación.
5. **Validation set sin uso declarado:** hay un 20% de validation pero no se reporta si se usa para early stopping, tuning o solo monitoreo.
6. **Números corruptos en la Tabla 3:** varios valores con exponentes faltantes (e.g., "7.50 ×10−" sin exponente, "LSTM-VMD .66 ×10−" sin dígito inicial, "8. 5 ×10−6" con espacio). Esto **impide verificar los resultados numéricos** y es un problema de reporte grave.
7. **Sin varianza entre ejecuciones:** la LSTM es estocástica, pero no se reporta la dispersión (aunque el WSRT mitiga parcialmente este problema).

**Fortalezas del reporte:**
- Target genuino de RV de alta frecuencia.
- Comparación de 6 técnicas de descomposición bajo el mismo protocolo (inputs iguales).
- Evaluación multi-horizonte (1, 5, 20 pasos).
- MCS y Wilcoxon signed-rank test para significancia estadística.
- Diagnóstico de los IMFs con Ljung-Box (correlación serial significativa al 1%).
- Análisis de frecuencia central para K de VMD (Tabla 2).
- Reconocimiento explícito de las limitaciones (sin tuning, sin variables exógenas, K y α de VMD).
- Comparación contra HAR y ARIMA como baselines econométricos.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número total de parámetros de las LSTM.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** "The data used to support the findings of this study will be made available on request". Tushare es de pago, pero los futuros de China son accesibles vía fuentes públicas del exchange (INE).
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast).
- **Limitaciones reportadas:** los autores mencionan que usaron una sola secuencia de RV como entrada (univariada), configuración LSTM fija sin optimización, y que K y α de VMD no están optimizados.

## 5.2 Hiperparámetros reportados (valores finales)
- **LSTM:** learning rate 0.001; batch_size 32. **NO se reportan** número de capas, neuronas ocultas, dropout, función de pérdida, early stopping.
- **VMD:** K=8 (frecuencia central). **NO se reporta α** (penalización cuadrática) ni el criterio de parada.
- **Otras descomposiciones:** **NO se reportan sus hiperparámetros** (número de IMFs para EMD/EEMD/CEEMDAN/ICEEMDAN, número de bandas para EWT, amplitud del ruido para EEMD/CEEMDAN/ICEEMDAN).
- **Normalización:** **NO reportada.**
- **Validation set:** 20% del total. **NO se reporta su uso.**
- **Semilla:** **NO reportada.**
- **HAR:** **NO se reportan los lags** (diario, semanal, mensual) ni el período de estimación.
- **ARIMA:** **NO se reportan los órdenes (p,d,q)** ni el criterio de selección.

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **Ninguno.** Configuración LSTM fija. |
| Espacio de búsqueda | N/A |
| Valores finales | lr=0.001, batch_size=32. **Capas, neuronas y dropout no reportados.** |
| Optimizador | **NO reportado.** |
| Learning rate | 0.001 |
| Batch size | 32 |
| Épocas | **NO reportado.** |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No reportado.** Hay un validation set (20%), pero **no se declara su uso**. No hay búsqueda de hiperparámetros.

**¿Sobre qué conjunto se evalúa el fitness?** **No aplica.** No hay búsqueda de hiperparámetros.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **NO reportado.** Se asume MSE o MAE sobre los IMFs, pero no se declara.

### 5.3.4 Modelos de referencia y descomposiciones
- **HAR:** lags diario/semanal/mensual asumidos (no reportados). **No se reporta el período de estimación** (rolling vs estático).
- **ARIMA:** órdenes (p,d,q) **no reportados**. **No se reporta el criterio de selección** (AIC, BIC).
- **EMD/EEMD/CEEMDAN/ICEEMDAN/EWT:** hiperparámetros **no reportados** (número de IMFs, amplitud del ruido, número de bandas para EWT).
- **VMD:** K=8 por frecuencia central; α **no reportado**.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura VMD-LSTM (flujo) | Sí (descrita en Fig. 1 y Ec. 24) | Bajo: está descrita |
| Capas y neuronas de la LSTM | No | Alto: no se puede replicar la red |
| Función de pérdida | No | Alto: no se sabe qué se minimiza |
| Dropout, early stopping | No | Medio: afectan la regularización |
| K de VMD | Sí (8, por frecuencia central) | Bajo: está reportado |
| α de VMD | No | Medio: afecta la descomposición |
| Hiperparámetros de otras descomposiciones | No | Medio: afectan la comparación |
| Normalización (tipo y timing) | No | Medio: potencial fuga |
| Uso del validation set | No | Alto: no se sabe si hay fuga por tuning |
| Órdenes de ARIMA y lags de HAR | No | Medio: afectan los baselines |
| Semilla | No | Medio: la LSTM es estocástica |
| Números de la Tabla 3 | Corruptos | **Alto: impide verificar los resultados** |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Parcialmente, con un riesgo de divergencia alto.** Los supuestos que habría que tomar:
1. **LSTM:** 1-2 capas, 50-64 neuronas, dropout 0.2, early stopping con paciencia 10, función de pérdida MSE.
2. **VMD:** K=8, α=2000 (valor típico), criterio de parada por convergencia.
3. **Otras descomposiciones:** EMD/EEMD/CEEMDAN/ICEEMDAN con 8-10 IMFs; EWT con 8 bandas; amplitud de ruido 0.2 desviaciones estándar.
4. **Normalización:** Min-Max ajustado solo con train.
5. **Protocolo:** rolling window con re-entrenamiento periódico o estático, pero **homogéneo** para todos los modelos.
6. **Semilla:** fijar una semilla.
7. **HAR:** lags 1, 5, 22 (diario, semanal, mensual).
8. **ARIMA:** selección automática por AIC.

**Riesgo de divergencia del desempeño reportado:** **muy alto.** El desempeño reportado está inflado por: (i) look-ahead bias por descomposición global; (ii) sin tuning de hiperparámetros; (iii) números corruptos en la tabla. Sin la fuga, el desempeño sería sustancialmente menor y la comparación contra LSTM puro/HAR/ARIMA podría incluso revertirse.

**Riesgo de divergencia de la arquitectura:** **alto.** La arquitectura LSTM no está reportada (capas, neuronas, dropout, función de pérdida).

**Riesgo de divergencia del pipeline:** **alto.** Faltan el timing de normalización, el uso del validation set, los hiperparámetros de las descomposiciones y los números de la tabla.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, look-ahead bias por descomposición global, arquitectura LSTM no reportada, función de pérdida no declarada, sin tuning de hiperparámetros, K y α de VMD no justificados, hiperparámetros de las otras descomposiciones no reportados, normalización no declarada, uso del validation set no declarado, números corruptos en la Tabla 3 y ausencia de código, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no una réplica exacta. **Este punto es secundario porque el estudio se excluye por C3.**

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de RV de alta frecuencia con LSTM como componente central.
- **C2. Reporte suficiente:** **Parcial.** Reporta el flujo del modelo, K=8 para VMD, learning rate y batch_size, pero no la arquitectura LSTM, la función de pérdida, el α de VMD, la normalización, el uso del validation set ni el código. **Los números corruptos de la Tabla 3 impiden verificar los resultados numéricos.**
- **C3. Calidad editorial:** **No.** La revista *Discrete Dynamics in Nature and Society* es **Q3** en Scopus/WoS. El paper **no es fundacional**: no introduce una arquitectura nueva ni es un trabajo de referencia citado como comparador en un estudio 2020–2026. Es un estudio comparativo de 6 técnicas de descomposición existentes aplicadas a LSTM. **Falla C3.** La exclusión es consistente con la de Verma2021 (Q3, no fundacional).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El protocolo es reconstruible con supuestos (partición temporal documentada, arquitectura descrita, K reportado). La fuga por descomposición global es un problema de validez, no de reconstruibilidad.
- **E6. Solo precio, sin volatilidad:** **No aplica.** Predice RV de 5 minutos.
- **E7. GARCH como contribución principal:** **No aplica.** HAR y ARIMA son baselines, no la contribución principal.

**Resultado:** **Excluido.** Motivo: **C3** (revista Q3, no fundacional, sin cita documentada como comparador en 2020–2026). Consistente con la exclusión de Verma2021.

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo (para el registro, aunque el paper se excluya):**
- **Target:** RV de 5 minutos, medida estándar de alta frecuencia. **Sin sesgo del target** (aunque no se reporta si usan retornos logarítmicos o simples ni estimadores robustos). → **bajo**.
- **Partición:** split 7:2:1 cronológico, pero **sin fechas exactas** y con **validation set sin uso declarado**. → **moderado**.
- **Predicción:** rolling window para OOS, pero **sin tamaño de ventana ni re-entrenamiento declarado**. → **moderado**.
- **Fuga:** **look-ahead bias CONFIRMADO** por descomposición global sobre la serie completa antes del split. Los IMFs del train contienen información del test. → **alto**.
- **Reporte selectivo:** reporta 9 modelos, 3 horizontes, 3 métricas, MCS y Wilcoxon. Pero **sin varianza entre ejecuciones** y con **números corruptos en la Tabla 3**. → **moderado-alto**.
- **Inferencia:** **MCS y Wilcoxon signed-rank test** formales. **Bien.** Pero las comparaciones están contaminadas por la fuga. → **moderado**.
- **Comparación:** **inputs iguales entre las 6 descomposiciones** (todas parten de la misma serie RV) → **justa** para el ablation central. Pero **contra LSTM puro, HAR y ARIMA** → **contaminada** porque solo los modelos con descomposición tienen la fuga. → **alto**.
- **Ablation:** se comparan **6 descomposiciones + LSTM puro** → bien cubierto para la elección de descomposición. Pero **no se varía K ni α de VMD**, y **no se aísla la estrategia de suma vs. concatenación de IMFs**. → **moderado-alto**.
- **Reproducibilidad:** sin arquitectura LSTM, sin función de pérdida, sin α de VMD, sin hiperparámetros de otras descomposiciones, sin normalización, sin uso del validation set, sin código, con números corruptos. → **muy parcial**.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** Los futuros de crudo de China son accesibles vía Tushare (de pago) o fuentes públicas del exchange (INE). Los retornos de 5 minutos se pueden construir. → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** Flujo descrito y K=8 reportado, pero faltan arquitectura LSTM, función de pérdida, α de VMD, hiperparámetros de las otras descomposiciones y normalización. → **Parcial**.
- **Viabilidad computacional:** **sí.** VMD con K=8 + 9 LSTM (uno por IMF más residuo), 3 horizontes. Viable en CPU/GPU estándar. → **Sí**.

**Clasificación: R-B (no reproducible por el equipo).** R1=Sí, R2=Tal vez, R3=Sí → no cumple R2=Sí. El estudio se conserva en el registro de la revisión como referencia metodológica para el mapeo de técnicas de descomposición, pero **no entra al benchmark por C3**.

**Decisión provisional:**
- **Entra al benchmark:** **No.** Excluido por C3.
- **Tipo de entrada:** **N/A (excluido).**
- **Correcciones necesarias:** **N/A** (no aplica, porque el paper se excluye por C3). Si en el futuro se decidiera **enmendar C3** para admitir "relevancia metodológica" como excepción (primer análisis comparativo sistemático de una familia de métodos bajo un protocolo común, o ejemplo ejemplar de un mecanismo de sesgo), entonces el paper podría incluirse **con corrección** y las correcciones serían:
    - **(PRIORITARIA) Corregir el look-ahead bias por descomposición global:** aplicar la descomposición de forma **rolling** (recalcular los IMFs en cada origen de pronóstico usando solo datos pasados) o **solo sobre el train set**.
    - **(PRIORITARIA) Reportar la arquitectura LSTM:** número de capas, neuronas ocultas, dropout, función de pérdida, early stopping y optimizador.
    - **(PRIORITARIA) Reportar el α de VMD** y justificar K=8 con un análisis de sensibilidad.
    - **Reportar los hiperparámetros de las otras descomposiciones.**
    - **Reportar el uso del validation set.**
    - **Reportar la normalización (tipo y timing).**
    - **Reportar las fechas exactas del split y el tamaño de la ventana móvil.**
    - **Corregir los números corruptos de la Tabla 3.**
    - **Reportar la varianza entre ejecuciones.**
    - **Reportar los órdenes de ARIMA y los lags de HAR.**
    - **Reportar la semilla.**
    - **Añadir QLIKE.**
    - **Liberar el código.**
- **Brecha principal:** **N/A para el benchmark** (excluido por C3). El estudio se reporta en la revisión como **evidencia de que existe una línea de investigación de descomposición + LSTM para RV que, aunque metodológicamente relevante (comparación sistemática de 6 descomposiciones y ejemplo claro de look-ahead bias por descomposición global), no alcanza el estándar de calidad editorial del corpus (Q3 no fundacional)**. Las limitaciones específicas son: **look-ahead bias confirmado por descomposición global**, **arquitectura LSTM no reportada**, **sin tuning de hiperparámetros**, **K y α de VMD no justificados**, **hiperparámetros de las otras descomposiciones no reportados**, **normalización y uso del validation set no declarados**, **números corruptos en la Tabla 3** y **sin código**. La afirmación central del paper ("VMD es la mejor descomposición para LSTM en RV") **está parcialmente identificada** para la comparación entre descomposiciones (todas tienen el mismo sesgo), pero **no está identificada** para la comparación contra modelos sin descomposición (LSTM puro, HAR, ARIMA), donde el look-ahead bias infla artificialmente la mejora. **No compite en la comparación de modelos.**