---
id: Srivastava2023
title: "What do the AI methods tell us about predicting price volatility of key natural resources: Evidence from hyperparameter tuning."
authors: "Mrinalini Srivastava; Amar Rao; Jaya Singh Parihar; Shubham Chavriya; Surendar Singh"
year: 2023
doi: "10.1016/j.resourpol.2022.103249"
journal: "Resources Policy"
editorial_quality: "Q1 (Elsevier)"
status: "Exclude"
exclusion_reason: "E5: El estudio no reporta una partición Train/Test para el modelo híbrido LSTM-GJR-GARCH. El RMSE se calcula in-sample, sobre las mismas 2,733 observaciones usadas para entrenar. Esto invalida cualquier conclusión sobre capacidad predictiva. Adicionalmente, el target es la salida del GJR-GARCH (volatilidad condicional paramétrica), no la volatilidad realizada de alta frecuencia; el LSTM se entrena para replicar la salida del GARCH, lo que constituye un bucle redundante sin valor predictivo."
problem_type: "Point forecast"
horizon: "single-step"
proxy: "Conditional variance (GARCH)"
architectural_family: "Hybrid"
architectural_variant: "LSTM-GJR-GARCH (1,1) with hyperparameter tuning"
market_asset: "Crude Oil, Natural Gas, Copper, Silver, Gold"
---

# Resumen
El estudio propone un modelo híbrido LSTM-GJR-GARCH (1,1) con ajuste de hiperparámetros para pronosticar la volatilidad de cinco recursos naturales: crudo, gas natural, cobre, plata y oro. Utilizan el GJR-GARCH para capturar la heterocedasticidad condicional y el efecto apalancamiento, y luego alimentan sus predicciones de volatilidad a una red LSTM para refinar el pronóstico. Los autores reportan que el ajuste de hiperparámetros (Grid Search) mejora la precisión del pronóstico en un 40%–70% en comparación con el uso del LSTM-GJR-GARCH sin tuning. Además, analizan la relación causal entre el Índice de Riesgo Geopolítico (GPRD) y la conectividad total (TCI) de los recursos naturales. **El estudio se excluye del corpus porque no reporta una partición Train/Test y porque el target es la salida paramétrica del GARCH, no la volatilidad realizada.**

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **volatilidad condicional** (varianza) de los precios de los recursos naturales, modelada mediante GJR-GARCH (1,1). El paper no predice la Volatilidad Realizada (RV) de alta frecuencia, sino que utiliza el GJR-GARCH para modelar la varianza condicional y luego el LSTM para predecir esa serie de varianzas. El objetivo es pronosticar la volatilidad futura (no se especifica explícitamente el horizonte de pronóstico).

**Bucle redundante y sesgo in-sample (crítica metodológica):**
El estudio comete un error metodológico fundamental. El GJR-GARCH (1,1) ya es un modelo de pronóstico que estima la varianza condicional $\sigma^2_{t+1}$ y captura el efecto apalancamiento. Al pasar esa serie de pronósticos a una red LSTM, la red no está aprendiendo a predecir la volatilidad real del mercado, sino que está siendo entrenada para **replicar y suavizar la salida paramétrica del GJR-GARCH**. Además, **no realiza una división Train/Test**. El RMSE se calcula sobre el conjunto de datos completo (2,733 observaciones). Las mejoras del 40%–70% reportadas por el tuning se deben a la capacidad de la LSTM de memorizar los residuos del GARCH en los mismos datos de entrenamiento, no a una mejora en la generalización.

**Justificación del GJR-GARCH (1,1):**
El uso del GJR-GARCH es correcto en el contexto econométrico, ya que captura el efecto apalancamiento (las malas noticias aumentan la volatilidad más que las buenas). Sin embargo, el paper **no compara** el GJR-GARCH contra otras variantes asimétricas (EGARCH, TGARCH, APARCH) para justificar empíricamente por qué el orden (1,1) y esta especificación son las óptimas para cada recurso. Tampoco reporta pruebas de diagnóstico de residuos (Ljung-Box sobre residuos al cuadrado) para validar la adecuación del modelo.

**Fuente de los datos:**
- **Bloomberg.** Precios de cierre diarios del crudo, gas natural, cobre, plata y oro (en USD). Periodo: enero de 2012 – julio de 2022 (2,733 observaciones).
- **Índice de Riesgo Geopolítico (GPRD):** Caldara e Iacoviello (2022). Se usa solo para el análisis de causalidad con la conectividad total (TCI), no como input del modelo de pronóstico.
- **Data availability:** “Data will be made available on request”.

**Partición Temporal:** **NO REPORTADA.** No se especifica una división explícita de Train/Test para el modelo híbrido LSTM-GJR-GARCH. Los resultados de RMSE se reportan para el conjunto de datos completo. No hay rolling origin ni walk-forward.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** **No aplica (no hay evaluación out-of-sample).** El modelo se entrena y se evalúa sobre los mismos datos. No se declara si la predicción es open-loop con actuales o recursiva.

**Normalización:** se aplicó normalización Min-Max a las secuencias de volatilidad pronosticadas por el GJR-GARCH antes de introducirlas al LSTM (Eq. 5). No se especifica si el scaler se ajustó solo con el train set o con la serie completa.

**Ingeniería de Características:**
- **GJR-GARCH (1,1):** se estima sobre los log-retornos de cada recurso. La serie de volatilidad condicional pronosticada se usa como input del LSTM.
- **Sin descomposición:** no hay EMD, VMD ni filtros no causales.
- **Sin variables exógenas en el pronóstico:** el LSTM recibe únicamente la serie de volatilidad pronosticada por el GARCH. El GPRD se usa solo para el análisis de causalidad, no como input.

# 2. Arquitectura Final (LSTM-GJR-GARCH con Tuning)

El modelo es un **ensamble secuencial en serie**:
1. **Etapa 1 (GJR-GARCH):** se estima el modelo GJR-GARCH (1,1) sobre los retornos diarios para obtener la serie de volatilidad condicional (varianza). El modelo captura el efecto apalancamiento (shocks negativos tienen mayor impacto en la volatilidad).
2. **Etapa 2 (LSTM):** la serie de volatilidad pronosticada por el GARCH se normaliza y se utiliza como input para una red LSTM. La LSTM se entrena para predecir la volatilidad futura (no se especifica el horizonte). Se aplica Grid Search para encontrar la mejor combinación de hiperparámetros.

**LSTM (modelo base):** ecuaciones estándar con compuertas de entrada, olvido y salida. **No se reporta** el número de capas LSTM, el número de neuronas por capa, el dropout final ni el learning rate final. Solo se reportan los rangos de búsqueda del Grid Search (Tabla 2).

# 3. Resultados y Comparaciones

**Modelos comparados (3 en total):**
- **GJR-GARCH (1,1):** modelo base econométrico.
- **LSTM-GJR-GARCH (sin tuning):** híbrido con hiperparámetros por defecto.
- **LSTM-GJR-GARCH (con tuning):** híbrido optimizado con Grid Search.

**Métricas y Rendimiento:**
RMSE. El paper reporta mejoras en el RMSE después del tuning:
- **Plata:** Mejora del 77.00% (RMSE antes=0.0683, después=0.0160).
- **Cobre:** Mejora del 52.00% (RMSE antes=0.0478, después=0.0232).
- **Crudo:** Mejora del 48.00% (RMSE antes=0.0271, después=0.0140).
- **Gas:** Mejora del 35.00% (RMSE antes=0.323, después=0.0211).
- **Oro:** Mejora del 30.00% (RMSE antes=0.012, después=0.0083).

**Pruebas de Significancia:**
**NO se reportan pruebas de significancia estadística** para comparar los modelos (ni Diebold-Mariano, ni MCS, ni Clark-West). La comparación se basa únicamente en la reducción del RMSE antes y después del tuning. Tampoco se reporta la varianza entre semillas o ejecuciones.

**Sesgo de Reporte:**
Alto. Cinco problemas:
1. **Target inválido para el corpus:** predice la volatilidad condicional del GARCH, no la volatilidad realizada de alta frecuencia. El LSTM se entrena para replicar la salida paramétrica del GARCH, no para predecir la volatilidad real del mercado.
2. **Evaluación in-sample:** no hay una división Train/Test. El RMSE se calcula sobre los mismos datos con los que se entrenó el modelo. Esto infla artificialmente el rendimiento y no demuestra capacidad predictiva real. Es un caso de **E5**.
3. **Bucle redundante:** usar una red LSTM para predecir la salida del GARCH no aporta valor predictivo real; solo demuestra que el Deep Learning puede memorizar la función paramétrica del GARCH.
4. **Falta de justificación del GJR-GARCH:** no comparan el GJR-GARCH (1,1) contra otras especificaciones (EGARCH, TGARCH, APARCH) para justificar su elección.
5. **Comparación limitada:** solo prueban GJR-GARCH y LSTM-GJR-GARCH. No comparan contra otros modelos de Machine Learning (SVR, RF, XGBoost) ni contra otros Deep Learning (GRU, Transformer, etc.).

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros del LSTM.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** “Data will be made available on request”.
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast).
- **Limitaciones reportadas:** los autores admiten que el estudio se enfoca en la volatilidad de los precios de los recursos naturales y que futuras investigaciones podrían explorar otros métodos de optimización de hiperparámetros y modelos más avanzados.

## 5.2 Hiperparámetros reportados (valores finales)
- **GJR-GARCH (1,1):** parámetros $\omega, \alpha, \beta, \gamma$ estimados para cada recurso. Los valores finales **no se reportan**.
- **LSTM:** **NO reportados.** No se reportan el número de capas, neuronas, dropout final, learning rate final, batch size ni épocas.
- **Grid Search:** rangos reportados en Tabla 2:
  - Batch size: 10, 20, 40, 60, 80, 100.
  - Épocas: 10, 50, 100.
  - Optimizador: SGD, RMSprop, Adagrad, Adadelta, Adam.
  - Dropout: 0.0 a 0.9 (paso de 0.1).
  - Neuronas en la capa oculta: 1, 5, 10, 15, 20, 25, 30.

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | Grid Search |
| Espacio de búsqueda | Batch size (10-100), épocas (10-100), optimizador (SGD, RMSprop, Adagrad, Adadelta, Adam), dropout (0.0-0.9), neuronas (1-30). |
| Valores finales | **NO reportados.** |
| Función de fitness | **NO reportada.** Se asume RMSE, pero no se declara. |
| Conjunto de evaluación del fitness | **NO reportado.** No hay conjunto de validación separado. |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No se menciona.** El Grid Search se realiza presumiblemente sobre el conjunto completo, sin validación cruzada ni hold-out.

**¿Sobre qué conjunto se evalúa el fitness?** **NO reportado.** No se aclara si el fitness del Grid Search se calcula sobre el train set, sobre un subconjunto del train, o sobre la serie completa.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **NO reportado.** No se especifica la función de fitness del Grid Search. Por el contexto, se asume RMSE sobre la serie de volatilidad del GARCH, pero no se declara.

### 5.3.4 GJR-GARCH
Los parámetros del GJR-GARCH se estiman una sola vez sobre la serie completa. No se reportan los valores finales ni las pruebas de diagnóstico de residuos.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **No de forma útil para el benchmark.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Partición Train/Test | No | **Alto**: no hay evaluación out-of-sample |
| Arquitectura del LSTM (capas, neuronas, dropout) | No | Alto: no se puede reconstruir la red exacta |
| Función de pérdida | No | Medio: se asume RMSE, pero no se declara |
| Valores finales del Grid Search | No | Alto: no se puede replicar la configuración óptima |
| Función de fitness del Grid Search | No | Alto: no se puede replicar el criterio de optimización |
| Conjunto de evaluación del fitness | No | Alto: no se sabe si usó train o serie completa |
| Parámetros del GJR-GARCH | No | Medio: no se pueden replicar las estimaciones |
| Diagnóstico de residuos del GJR-GARCH | No | Medio: no se valida la adecuación del modelo |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **No de forma útil para el benchmark.** El problema no es solo la falta de hiperparámetros: es que el estudio no tiene evaluación out-of-sample. Reproducir el estudio implicaría replicar un RMSE in-sample, lo cual no tiene valor para el benchmark. Además, el target es la salida del GARCH, no la volatilidad realizada, lo que lo excluye del corpus de todas formas.

**Riesgo de divergencia del desempeño reportado:** **muy alto.** El RMSE in-sample está inflado por la memorización del GARCH por parte del LSTM. Sin evaluación out-of-sample, no hay forma de saber si el modelo generaliza.

**Riesgo de divergencia de la arquitectura:** **alto.** Sin los valores finales del Grid Search, la reimplementación es una aproximación.

**Riesgo de divergencia del pipeline:** **alto.** El pipeline no tiene partición Train/Test, y el target es la salida del GARCH.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, evaluación in-sample, ausencia de partición Train/Test, valores finales del Grid Search no reportados, función de fitness no declarada, parámetros del GJR-GARCH no reportados y ausencia de código—, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación sería una aproximación, no una réplica. Sin embargo, **este punto es irrelevante para el benchmark porque el estudio se excluye por E5 (evaluación in-sample) y por target inválido (volatilidad condicional del GARCH, no RV).**

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Parcial.** El estudio aborda volatilidad de recursos naturales y usa DL (LSTM), pero el target es la salida del GARCH, no la volatilidad realizada. El LSTM no aporta valor predictivo independiente.
- **C2. Reporte suficiente:** **No.** No reporta partición Train/Test, valores finales del Grid Search, función de fitness, parámetros del GJR-GARCH ni código.
- **C3. Calidad editorial:** **Sí.** Q1 en JCR, Elsevier.
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **Sí.** No hay partición Train/Test. El RMSE es in-sample. El protocolo no permite evaluar capacidad predictiva real.
- **E6. Solo precio, sin volatilidad:** No aplica (predice volatilidad condicional, aunque con target paramétrico).
- **E7. GARCH como contribución principal:** No aplica (GARCH es un componente del híbrido).

**Resultado:** **Excluido.** Motivos: **E5** (evaluación in-sample, sin partición Train/Test) y **target inválido** (volatilidad condicional del GARCH, no RV; bucle redundante).

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** volatilidad condicional del GARCH, no RV de alta frecuencia. El LSTM replica la salida del GARCH. → **excluyente**.
- **Partición:** no reportada. Evaluación in-sample. → **excluyente (E5)**.
- **Predicción:** no aplica (no hay out-of-sample). → **excluyente**.
- **Fuga:** no aplica (no hay train/test, no hay fuga en el sentido tradicional, pero el modelo es inútil para predicción).
- **Reporte selectivo:** solo reporta mejoras de RMSE in-sample. → alto.
- **Inferencia:** sin pruebas de significancia. → alto.
- **Comparación:** 3 modelos, pero todos in-sample. → alto.
- **Reproducibilidad:** sin código, sin valores finales del Grid Search, sin parámetros del GJR-GARCH. → **muy parcial**.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** Precios públicos (Bloomberg, EIA, Investing.com); GPRD público (Caldara-Iacoviello). → **Sí**.
- **Reconstructibilidad del pipeline:** **no.** Falta la partición Train/Test, los valores finales del Grid Search y los parámetros del GJR-GARCH. → **No**.
- **Viabilidad computacional:** **sí.** Los modelos son ligeros. → **Sí**.

**Decisión provisional:**
- **Entra al benchmark:** **No.**
- **Tipo de entrada:** **N/A (excluido por E5 y target inválido).**
- **Correcciones necesarias:**
    - Introducir una partición Train/Test cronológica (o rolling origin) para evaluar la capacidad predictiva out-of-sample. Sin esto, el estudio no puede competir en el benchmark.
    - Declarar los valores finales del Grid Search (capas, neuronas, dropout, learning rate, batch size, épocas) y la función de fitness utilizada.
    - Reportar los parámetros estimados del GJR-GARCH (ω, α, β, γ) para cada recurso, junto con pruebas de diagnóstico de residuos (Ljung-Box sobre residuos al cuadrado).
    - Comparar el GJR-GARCH contra otras variantes asimétricas (EGARCH, TGARCH, APARCH) para justificar la elección de la especificación.
    - Incluir pruebas de significancia estadística (DM, MCS) para comparar los modelos, no solo RMSE puntual.
    - Liberar el código o, en su defecto, reportar los hiperparámetros finales y la semilla para permitir la reimplementación exacta.
- **Brecha principal:** N/A para el benchmark. El estudio se reporta en la revisión como **evidencia de que existe una línea de investigación de híbridos GARCH-LSTM que, aunque usa DL, no aborda la predicción out-of-sample ni la volatilidad realizada, y presenta un bucle redundante donde el LSTM se entrena para replicar la salida del GARCH**. Las limitaciones específicas son: evaluación in-sample sin partición Train/Test (E5), target paramétrico (salida del GARCH, no RV), bucle redundante, ausencia de valores finales del Grid Search, y falta de pruebas de significancia. No compite en la comparación de modelos.