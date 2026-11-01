---
id: Zhang2024a
title: "Geopolitical risks and crude oil futures volatility: Evidence from machine learning."
authors: "Hongwei Zhang; Wentao Wang; Zibo Niu"
year: 2024
doi: "10.1016/j.resourpol.2024.105374"
journal: "Resources Policy"
editorial_quality: "Q1 (Elsevier)"
status: "Include"
exclusion_reason: "None"
problem_type: "Point forecast"
horizon: "multi-step"
proxy: "Realized volatility (RV)"
architectural_family: "Hybrid"
architectural_variant: "CCNTrans (Causal Convolution Network-Transformer)"
market_asset: "WTI Crude Oil Futures"
---

# Resumen
El estudio analiza el poder predictivo de ocho subcategorías de riesgo geopolítico (GPR) sobre la volatilidad realizada (RV) mensual de los futuros de crudo WTI. Propone un modelo CCNTrans (Causal Convolution Network-Transformer) y lo compara contra AR, AR-ALL, LASSO, EN, FNN, LSTM, GARCH y EGARCH. Encuentra que los riesgos asociados a guerra y terrorismo son los predictores más significativos, que su impacto es time-varying y asimétrico entre expansiones y recesiones, y que la incorporación de GPR mejora la precisión de los pronósticos y genera beneficios económicos en estrategias de portafolio. El estudio usa rolling origin con reentrenamiento mensual, sin descomposición y con pruebas de significancia formales (MCS, Diebold-Mariano, Clark-West).

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **Volatilidad Realizada (RV)** mensual del precio de los futuros de WTI:
$$RV_t = \sum_{k=1}^{K} r_{t,k}^2$$
donde $r_{t,k}$ son los retornos logarítmicos diarios dentro del mes $t$. Se usa $\ln(RV_t)$ para aproximar normalidad. **La RV se construye a partir de retornos diarios, no de datos intradía de 5 minutos.** Es una RV de baja frecuencia. Se predicen horizontes de 1, 3, 6 y 12 meses.

**Fuente de los datos:**
- **WTI futures (NYMEX):** abril 1984 – julio 2022. Precios diarios para construir la RV mensual.
- **Predictores:** 8 subcategorías de GPR de Caldara y Iacoviello (2022): Beginning of War, Escalation of War, Military Buildups, Nuclear Threats, Peace Threats, Terror Acts, Terror Threats, War Threats. Son índices públicos disponibles en https://www.matteoiacoviello.com/gpr.htm.
- **Data availability:** “Data will be made available on request”. Los precios de WTI son públicos (EIA, Investing.com); los índices GPR son públicos.

**Partición Temporal:** **rolling origin con re-calibración mensual**. El modelo se re-entrena cada mes y se predice el mes siguiente. Los conjuntos de train y validation se expanden progresivamente. El 90% del train se usa para entrenamiento y el 10% para validación. El Apéndice A lo describe explícitamente: “we engage in a rolling scenario, systematically re-calibrating the entire model on a monthly basis and using it for out-of-sample predictions in the subsequent month. With each re-calibration, the training set is forward by one month, while upholding a fixed-size rolling sample for validation.” **La cronología se respeta explícitamente:** el Apéndice A dice “we preserve the chronological order of predicted data by avoiding any disruption to the samples”.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** **Rolling origin con reentrenamiento mensual, open-loop con actuales.** El esquema es:
1. Para cada mes $t$ del período out-of-sample, se entrena el modelo con datos hasta $t-1$.
2. Se predice el mes $t$ usando las observaciones reales hasta $t-1$ (open-loop con actuales).
3. Se avanza un mes, se re-entrena, se predice el siguiente.
4. No hay propagación de errores porque cada predicción es one-step-ahead con historia real.

Este esquema simula un escenario real de predicción, donde el modelo se actualiza con la información disponible en cada momento.

**Normalización:** no se especifica, pero al ser rolling no afecta la validez de la evaluación.

**Ingeniería de Características:** **limpia, sin descomposición.** No hay ICEEMDAN, VMD, EMD ni filtros no causales. Se usan las 8 subcategorías de GPR directamente como variables exógenas. No hay look-ahead bias por descomposición. Los GPR son índices publicados con rezago, pero el paper no discute si el rezago de publicación es suficiente para evitar fuga. Se asume que los índices están disponibles en $t$ para predecir $t+1$, pero el paper no lo declara explícitamente.

**Uso de variables exógenas:** las 8 subcategorías de GPR se usan como predictores en AR-ALL, LASSO, EN, FNN, LSTM y CCNTrans. Los benchmarks AR, GARCH y EGARCH son univariados (sin GPR). Esta asimetría se discute en la sección de sesgo de reporte.

# 2. Arquitectura Final (CCNTrans)

**CCNTrans:** Transformer con convoluciones causales. La convolución causal captura información local; el multi-head attention captura dependencias globales. El paper argumenta que esto mitiga la insensibilidad al contexto local del Transformer vanilla (cita a Li et al., 2019).

**Inputs:** $\ln(RV_t)$ con 3 rezagos + las 8 subcategorías de GPR. Los 3 rezagos se seleccionan por AIC y SC.

**Arquitectura:** **no se detalla** el número de capas, cabezas de atención, dimensión de embeddings ni dimensión del feed-forward. Solo se reportan hiperparámetros de entrenamiento (dropout=0.1, batch=32, epochs=100, patience=5, Adam default).

**Ecuaciones reportadas:**
- Self-attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$
- Multi-head: $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$, con $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- Pero no se reportan los valores de $h$, $d$, ni el número de capas.

# 3. Resultados y Comparaciones

**Modelos comparados (10 en total):**
- AR (benchmark sin GPR)
- AR-ALL (con las 8 GPR)
- LASSO
- EN (Elastic Net)
- FNN
- LSTM
- CCNTrans
- GARCH
- EGARCH

Para evaluar el poder predictivo de cada subcategoría, se compara CCNTrans con una sola GPR contra CCNTrans sin GPR (9 variantes).

**Métricas y Rendimiento:**
MSE, MAE, MCS (Range y SemiQ), $R^2_{OOS}$, MSFE-adjusted (Clark-West) y Diebold-Mariano. El CCNTrans obtiene p-valores de 1.000 en MCS, superando a todos los benchmarks. En la comparación de subcategorías, los riesgos de guerra y terrorismo dominan. Por ejemplo, Terror Acts obtiene p-valores de 1.000 (MCS) a 1 mes; War Threats domina a 3 y 6 meses.

**Pruebas de Significancia:**
El estudio reporta **MCS con p-valores, Diebold-Mariano, $R^2_{OOS}$ con MSFE-adjusted (Clark-West)**. La mayoría de subcategorías de GPR mejoran significativamente el pronóstico de RV.

**Sesgo de Reporte:**
Moderado. Tres problemas:
1. **Asimetría de inputs en la comparación principal:** el benchmark AR y los GARCH/EGARCH son univariados (solo usan rezagos de $\ln(RV_t)$ o retornos históricos, **sin GPR**). Los modelos ML (AR-ALL, LASSO, EN, FNN, LSTM, CCNTrans) reciben 8 variables adicionales (las subcategorías de GPR). La narrativa principal compara CCNTrans contra AR (sin GPR), lo que hace que la mejora parezca mayor de lo que es. La comparación honesta sería CCNTrans vs AR-ALL, que también usa las 8 GPR. Ahí la mejora es más modesta.
2. **La Tabla 4 (análisis por subcategoría) sí es justa:** cada modelo CCNTrans difiere del benchmark CCNTrans por exactamente una variable (una subcategoría de GPR). Esto permite aislar el efecto marginal de cada tipo de riesgo geopolítico.
3. **El test DM (Tabla B.1) sí es justo:** todos los modelos comparados (AR-ALL, LASSO, EN, FNN, LSTM, CCNTrans) usan las 8 GPR. La comparación entre ellos es equitativa.

**Fortalezas del reporte:** el estudio reporta MCS, DM, $R^2_{OOS}$ y Clark-West, además de un análisis de subcategorías que aísla el efecto marginal de cada variable.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros del CCNTrans.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta. El rolling mensual con re-entrenamiento de 10 modelos durante 38 años (456 meses) implica un costo computacional alto, pero no se cuantifica.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** “Data will be made available on request”. Los precios de WTI son públicos; los índices GPR son públicos.
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast).
- **Limitaciones reportadas:** los autores mencionan que la frecuencia mensual es insuficiente y que futuras investigaciones deberían usar datos intradía de alta frecuencia para construir RV diaria. También mencionan que el análisis por país (consumidor vs productor) es una extensión pendiente.

## 5.2 Hiperparámetros reportados (valores finales)
- **AR:** $k=3$ rezagos, seleccionado por AIC y SC.
- **LASSO:** $\lambda \in (10^{-4}, 10^{-1})$, búsqueda en validation.
- **EN:** $\lambda \in (10^{-4}, 10^{-1})$, $\alpha = 0.5$ fijo.
- **FNN, LSTM, CCNTrans:** $\lambda \in (10^{-5}, 10^{-2})$, dropout=0.1, batch=32, epochs=100, patience=5, Adam default.
- **CCNTrans (arquitectura interna):** **NO reportada.** No se reporta el número de capas, cabezas de atención, dimensión de embeddings ni dimensión del feed-forward.
- **Función de pérdida:** **NO reportada.** Por el contexto (regresión de $\ln(RV)$), se asume MSE, pero no se declara.
- **GARCH/EGARCH:** especificación no reportada (órdenes $p,q$ no reportados).

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | Búsqueda en validation set (grid o random, no especificado) |
| Espacio de búsqueda (LASSO/EN) | $\lambda \in (10^{-4}, 10^{-1})$, $\alpha=0.5$ |
| Espacio de búsqueda (FNN/LSTM/CCNTrans) | $\lambda \in (10^{-5}, 10^{-2})$ |
| Dropout | 0.1 |
| Batch size | 32 |
| Epochs | 100 |
| Patience (early stopping) | 5 |
| Optimizador | Adam (default) |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No.** Usan un **conjunto de validación separado** (10% del train) para la búsqueda de hiperparámetros. El Apéndice A dice: “On the validation set, we perform hyperparameter optimization on each machine learning model used in our predictive analysis to approximate the best model architecture.” La validación es rolling: se mantiene un tamaño fijo de muestra de validación y se avanza con el train.

**¿Sobre qué conjunto se evalúa el fitness?** **Sobre el validation set (10% del train).** El paper lo declara explícitamente.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **NO reportado con precisión.** El paper no especifica la función de pérdida del CCNTrans. Por el contexto (regresión de $\ln(RV)$), se asume **MSE** sobre $\ln(RV)$, pero no se declara. Tampoco se especifica si la búsqueda de hiperparámetros minimiza MSE, MAE u otra métrica en validation.

### 5.3.4 GARCH y EGARCH
No se reportan los órdenes $(p,q)$ ni el criterio de selección. El paper los usa como benchmarks pero no detalla su especificación. Para reproducir, se tendría que hacer un análisis propio de los modelos GARCH/EGARCH.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **No de forma exacta.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura del CCNTrans (capas, cabezas, dimensiones) | No | Alto: no se puede reconstruir la red exacta |
| Función de pérdida | No | Medio: se asume MSE, pero no se declara |
| Órdenes GARCH/EGARCH | No | Medio: no se pueden replicar los benchmarks |
| Código | No | Alto: no hay referencia ejecutable |
| Semilla | No | Medio: los modelos DL son estocásticos; sin semilla, no se reproduce exactamente |
| Normalización | No reportada | Bajo: al ser rolling, no afecta la validez |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Sí, con un riesgo de divergencia medio.** Los supuestos que habría que tomar:

1. **CCNTrans:** Transformer estándar (p. ej., 2-4 capas, 4-8 cabezas, dimensión 64-128) con convoluciones causales (kernel 3, dilatación 1,2,4). Se declara como supuesto.
2. **Función de pérdida:** MSE sobre $\ln(RV)$.
3. **GARCH/EGARCH:** GARCH(1,1) y EGARCH(1,1) estándar.
4. **Semilla:** fijar una semilla para los modelos DL.
5. **Búsqueda de hiperparámetros:** grid search sobre los rangos reportados, con el validation set (10% del train) como criterio.
6. **Índices GPR:** usar los índices públicos de Caldara-Iacoviello, verificando que la versión descargada corresponde a la utilizada en el paper (pueden haber revisiones).

**Riesgo de divergencia del desempeño reportado:** **medio.** El esquema rolling origin es correcto, no hay look-ahead por descomposición, y el validation set está separado. La falta de arquitectura interna del CCNTrans introduce incertidumbre, pero el desempeño no está inflado por fuga. El CCNTrans probablemente mantiene su ventaja relativa sobre AR-ALL, aunque la magnitud exacta puede variar.

**Riesgo de divergencia de la arquitectura:** **medio.** El pipeline (rolling origin, sin descomposición, 8 GPR) es reconstruible. La arquitectura interna del CCNTrans se asume.

**Riesgo de divergencia del pipeline:** **bajo.** Los pasos están claros; el rolling origin es mecánico.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas —arquitectura del CCNTrans no detallada, función de pérdida no reportada, órdenes GARCH/EGARCH no reportados y ausencia de código—, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no el mismo modelo ni la misma arquitectura. El benchmark debe tratar la versión corregida como una **variante** del modelo original, no como una réplica exacta, y reportar la brecha entre ambos desempeños como evidencia del impacto de las carencias de reporte. La comparación con el artículo se limita a la **familia arquitectónica** (Transformer + convolución causal + GPR), no al desempeño puntual ni a la configuración exacta.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad del crudo con CCNTrans (Transformer + convolución causal) como componente central.
- **C2. Reporte suficiente:** **Tal vez.** Reporta hiperparámetros de entrenamiento y rangos de búsqueda, pero no la arquitectura interna del CCNTrans, ni la función de pérdida, ni los órdenes GARCH/EGARCH. Sin código.
- **C3. Calidad editorial:** **Sí.** Q1 en JCR, Elsevier.
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El pipeline es reconstruible con supuestos estándar (Transformer canónico, MSE, GARCH(1,1)). La partición rolling está reportada.
- **E6. Solo precio, sin volatilidad:** **No aplica**. Predice RV.
- **E7. GARCH como contribución principal:** **No aplica**. GARCH es benchmark, no contribución principal.

**Resultado:** Incluido.

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Partición:** rolling origin con re-calibración mensual, validación separada (10%), orden cronológico preservado explícitamente. → bajo.
- **Predicción:** rolling con reentrenamiento, open-loop con actuales. → bajo.
- **Fuga:** sin descomposición, sin normalización global, GPR como variables exógenas. No hay fuga evidente. → bajo.
- **Reporte selectivo:** reporta MCS, DM, $R^2_{OOS}$, MSFE-adjusted, y análisis por subcategoría. Múltiples métricas y pruebas. → bajo.
- **Inferencia:** MCS, DM, Clark-West con p-valores. → bajo.
- **Comparación:** asimetría de inputs en la comparación principal (CCNTrans vs AR sin GPR), pero comparaciones justas en Tabla 4 y Tabla B.1. → moderado.
- **Reproducibilidad:** sin código, sin arquitectura interna del CCNTrans, sin función de pérdida explícita, sin órdenes GARCH/EGARCH. → parcial.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** WTI futures público (EIA, Investing.com); índices GPR públicos (Caldara-Iacoviello). → **Sí**.
- **Reconstructibilidad del pipeline:** **sí, con supuestos.** Rolling origin, 8 GPR, CCNTrans aproximado con Transformer canónico, GARCH(1,1)/EGARCH(1,1). → **Sí**.
- **Viabilidad computacional:** **Tal vez.** Rolling mensual con reentrenamiento de 10 modelos durante 38 años (456 meses) es costoso pero viable con GPU moderna. → **Tal vez**.

**Decisión provisional:**
- **Entra al benchmark:** sí.
- **Tipo de entrada:** con corrección menor.
- **Correcciones necesarias:**
    - Especificar la arquitectura del CCNTrans (capas, cabezas, dimensiones) o, en su defecto, reimplementar con un Transformer canónico declarado como supuesto.
    - Declarar la función de pérdida (MSE sobre $\ln(RV)$) y el criterio de búsqueda de hiperparámetros.
    - Especificar los órdenes GARCH/EGARCH o reimplementar con GARCH(1,1)/EGARCH(1,1) estándar.
    - Fijar semilla para los modelos DL.
    - **Garantizar la equidad de inputs en la comparación principal:** todos los modelos deben recibir el mismo conjunto de variables. Si se comparan modelos con y sin GPR, reportar ambas comparaciones por separado y no atribuir la mejora al modelo cuando se debe a la información adicional.
- **Brecha principal:** la falta de especificación de la arquitectura del CCNTrans y de la función de pérdida, la ausencia de código, la asimetría de inputs en la comparación principal y la falta de especificación de los benchmarks GARCH/EGARCH. El esquema rolling origin es correcto y las pruebas de significancia son formales. La reimplementación es una aproximación, no una réplica exacta, y los resultados no son directamente comparables con los reportados.