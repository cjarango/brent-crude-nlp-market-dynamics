---
id: Jiang2026
title: "How uncertain are oil prices? Bayesian regularized distribution neural network for forecasting oil prices"
authors: "He Jiang; Yawei Dong; Shaolong Sun; Shouyang Wang"
year: 2026
doi: "10.1080/00036846.2025.2468937"
journal: "Applied Economics"
editorial_quality: "Q2 (Routledge)"
status: "Exclude"
exclusion_reason: "E6: Predice la distribución del precio del crudo (Price Distribution) en lugar de la Volatilidad Realizada (RV) o la varianza."
problem_type: "Probabilistic price forecast (point + density)"
horizon: "multi-step (1, 7, 15)"
proxy: "N/A — el target es el precio de cierre, no una proxy de volatilidad"
architectural_family: "Probabilistic DL (Mixture Density Network)"
architectural_variant: "BRDNN (PatchTST backbone + GMM output + Bayesian regularization + offset)"
market_asset: "Brent and WTI Crude Oil Futures"
---

# Resumen
El estudio propone un marco de pronóstico probabilístico llamado **BRDNN (Bayesian Regularized Distribution Neural Network)**. Este modelo integra arquitecturas de Deep Learning (DNN, RNN, CNN, TCN, LSTNet, NBEATS, NHITS, PatchTST, TFT) con una capa de salida basada en una mezcla de distribuciones Gaussianas (GMM) y regularización bayesiana. El objetivo es predecir la **distribución de probabilidad del precio del petróleo** para cuantificar la incertidumbre. **Se excluye del benchmark por E6: la variable objetivo es el nivel de precio del crudo (Brent y WTI), no la volatilidad realizada, la varianza ni una proxy equivalente.** El paper está bien construido para su propio objetivo (pronóstico probabilístico del precio), pero es **irrelevante para el benchmark de volatilidad**. Aun así, se documentan sus limitaciones metodológicas para el registro: (i) posible look-ahead bias por limpieza (Isolation Forest + interpolación de Lagrange) antes del split; (ii) normalización con timing no reportado; (iii) hiperparámetros parciales (sin capas, neuronas ni learning rate); (iv) sin pruebas de significancia entre modelos; (v) ablation incompleto (no aísla el componente GMM ni la regularización bayesiana); (vi) sin código.

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **distribución probabilística del precio de cierre** de los futuros de crudo Brent y WTI (en dólares por barril). El modelo genera una mezcla de Gaussianas (pesos $\pi_k$, medias $\mu_k$, varianzas $\sigma_k$) más un offset $b$ que ajusta el sesgo global. Se generan intervalos de predicción al 80% y 90% y densidades predictivas. **No se predice volatilidad realizada (RV), varianza ni proxy equivalente.** El target es el **nivel de precio**, no una medida de dispersión.

**Sobre la relevancia para el benchmark (E6):** El paper pertenece a la literatura de **pronóstico probabilístico del precio** (*density forecasting*), no a la de **pronóstico de volatilidad**. La incertidumbre se captura a través de la dispersión de la distribución predictiva, pero esa dispersión **no es un target de volatilidad**: es un subproducto del modelo de precio. Para el benchmark de volatilidad, este paper **no aporta una familia arquitectónica comparable** porque su variable objetivo es distinta. La exclusión por E6 es limpia y no requiere correcciones.

**Fuente de los datos:**
- **Precios:** Brent y WTI Crude Oil Futures, Investing.com. Periodo: enero 2011 – octubre 2023. Brent: 3,345 observaciones; WTI: 3,346 observaciones.
- **Data availability:** "The dataset is downloaded from the platform investing which is available at https://www.investing.com/". Los precios son públicos.

**Partición Temporal:**
- **In-sample (80%):** enero 2011 – abril 2021.
- **Out-of-sample (20%):** abril 2021 – octubre 2023.
- **Split cronológico.** Correcto para evitar look-ahead en la partición.

**Preprocesamiento y limpieza (riesgo de fuga):**
- **Isolation Forest** para eliminar valores atípicos.
- **Interpolación de Lagrange** para valores faltantes.
- **El paper no especifica si la limpieza se hizo antes o después del split.** Si se hizo sobre el dataset completo (lo más común en la práctica), el modelo podría "conocer" la forma de la serie futura para eliminar anomalías o rellenar huecos, incurriendo en **look-ahead bias**. Esta es una **bandera roja metodológica**.

**Normalización:**
Se menciona Z-score en la literatura citada, pero **no se reporta explícitamente** si el escalador se ajustó solo con el train set o con el dataset completo. **Ambigüedad con riesgo potencial de fuga.**

**Tuning de Hiperparámetros:**
- **Backbones probados (9, aunque el texto dice "8 classic models"):** DNN, RNN, CNN, TCN, LSTNet, NBEATS, NHITS, PatchTST, TFT.
- **Capa de salida:** GMM con softmax para pesos, softplus para varianzas, activación lineal para medias, más un offset $b$ con prior normal.
- **Estimación:** regularización bayesiana (inferencia sobre los pesos $W$).
- **No se reporta un conjunto de validación separado** para la búsqueda de hiperparámetros. Se aplica la inferencia bayesiana directamente. **No se reportan los hiperparámetros de la regularización bayesiana** (prior precision $\alpha$, $\beta$).
- **No se reportan capas, neuronas, learning rate ni batch size** de los backbones.

**Sobre la comparación entre backbones (check de inputs igualados):**
- Todos los backbones BRDNN reciben **la misma serie de precios** como input. La comparación **dentro del bloque BRDNN es justa** en inputs.
- **Pero la comparación contra QRA, BNN, MQRNN, DeepAR y DDNN-JSU no declara explícitamente que estos reciban los mismos inputs.** Se asume que sí (la serie de precios), pero no se verifica. Esta es una **ambigüedad de comparación**.

# 2. Arquitectura Final (PatchTST-BRDNN + offset)

El modelo propuesto, denominado **PatchTST-BRDNN**, tiene la siguiente estructura:

1. **Backbone:** PatchTST (Transformer con patch embedding). El paper probó 9 backbones y PatchTST fue el mejor en la mayoría de indicadores.
2. **Capa de salida probabilística:** Mixture Density Network con $K$ componentes Gaussianos. Los pesos $\pi_k$ se obtienen con softmax; las varianzas $\sigma_k$ con softplus; las medias $\mu_k$ con activación lineal.
3. **Offset $b$:** un término adicional con prior normal que ajusta el sesgo global de la mezcla. Se entrena un output unit extra en la última capa oculta para $\mu_b$. El offset se suma a la media de cada componente Gaussiano.
4. **Regularización bayesiana:** los pesos $W$ se estiman vía inferencia bayesiana (posterior $p(W|X_{1:T})$), no vía máxima verosimilitud. El objetivo es evitar sobreajuste y mejorar la generalización.
5. **Predicción probabilística:** se usa una cópula para calcular la probabilidad condicional de $x_t$ dado $X_{1:t}$, y se integra sobre el offset $b$ para obtener la distribución predictiva a $h$ pasos.

**Función de pérdida:** **NO reportada explícitamente.** Se describe el likelihood de la GMM (Eq. 12) y la regularización bayesiana, pero no se declara la función objetivo exacta que se minimiza durante el entrenamiento (¿log-likelihood negativa? ¿ELBO? ¿MSE + regularización?).

**Ablation del componente GMM y de la regularización bayesiana (gap):**
- El paper **sí ablate el backbone** (9 opciones) y **sí ablate el offset** (with vs without).
- **Pero no ablate el componente GMM** (¿qué pasa si se usa una sola Gaussiana en lugar de una mezcla?) ni **la regularización bayesiana** (¿qué pasa si se usa maximum likelihood en lugar de inferencia bayesiana?).
- La comparación contra BNN no es un ablation: BNN es un enfoque distinto (incertidumbre sobre los pesos, no mezcla de Gaussianas en la salida).
- **Consecuencia:** no se puede atribuir la mejora al componente GMM, a la regularización bayesiana o al offset por separado. El ablation es **incompleto**.

# 3. Resultados y Comparaciones

**Modelos comparados:**
- **Deterministas + BRDNN (9 backbones):** DNN-BRDNN, RNN-BRDNN, CNN-BRDNN, TCN-BRDNN, LSTNet-BRDNN, NBEATS-BRDNN, NHITS-BRDNN, PatchTST-BRDNN, TFT-BRDNN.
- **Probabilísticos de referencia:** QRA (Quantile Regression Averaging), BNN (Bayesian Neural Networks), MQRNN (Multi-Horizon Quantile RNN), DeepAR, DDNN-JSU.

**Métricas de Evaluación:**
- **Probabilísticas:** CRPS (Continuous Ranked Probability Score), PICP (Prediction Interval Coverage Probability), PINAW (Prediction Interval Normalized Average Width), MPICD (Mean Prediction Interval Center Deviation), CWC (Coverage Width-based Criterion).
- **Financieras (backtesting):** Sharpe Ratio (SR), Total Return (TR), Annualized Return (AR).

**Ganador y Rendimiento:**
El modelo **PatchTST-BRDNN** es el ganador, especialmente en Brent.
- **1 día, 90%:** CRPS = 1.0376 (Brent), 1.0616 (WTI); PICP ≈ 0.8980 (Brent), 0.9104 (WTI); CWC = 0.0950 (Brent), 0.2270 (WTI).
- **1 día, 80%:** PINAW = 0.0491 (Brent), 0.0796 (WTI).
- **7 y 15 días:** PICP = 0.8652 y 0.7890 (Brent, 90%).
- Frente a QRA, BNN, MQRNN, DeepAR, DDNN-JSU: PatchTST-BRDNN reduce CRPS en ~14% (Brent) y ~12% (WTI) en 1 día.

**Pruebas de Significancia:**
- **No se reporta ninguna prueba de significancia formal** (ni Diebold-Mariano, ni Model Confidence Set, ni Clark-West). La comparación se basa en métricas puntuales de CRPS, PICP, PINAW, MPICD y CWC.
- **No se reporta la varianza ni la desviación estándar** entre ejecuciones. El modelo es estocástico (inferencia bayesiana, inicialización de pesos), pero no se reporta dispersión.
- Las diferencias entre PatchTST-BRDNN y TFT-BRDNN (CRPS 1.0376 vs valores cercanos) podrían estar dentro del ruido, pero no se puede saber sin una prueba formal.

**Sesgo de Reporte:**
Moderado-alto. Seis problemas:
1. **Posible look-ahead por limpieza:** Isolation Forest + interpolación de Lagrange antes del split (no se especifica el timing).
2. **Normalización con timing no reportado:** riesgo potencial de fuga.
3. **Hiperparámetros parciales:** sin capas, neuronas, learning rate, batch size ni parámetros de la regularización bayesiana.
4. **Sin pruebas de significancia entre modelos:** solo métricas puntuales.
5. **Sin varianza entre ejecuciones:** el modelo es estocástico pero no se reporta dispersión.
6. **Ablation incompleto:** se ablate el backbone y el offset, pero no el componente GMM ni la regularización bayesiana. La comparación contra BNN no es un ablation.

**Fortalezas del reporte:**
- Comparación amplia: 9 backbones + 5 modelos probabilísticos de referencia.
- Evaluación multi-horizonte (1, 7, 15 días) y multi-nivel (80%, 90%).
- Métricas probabilísticas estándar (CRPS, PICP, PINAW, MPICD, CWC).
- Backtesting financiero con Sharpe Ratio, Total Return y Annualized Return.
- Ablation del offset (with vs without) y del backbone.
- Evaluación en dos mercados (Brent y WTI).
- Estrategia de trading con KDJ como filtro adicional.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número total de parámetros de PatchTST-BRDNN ni de los backbones.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** "The dataset is downloaded from the platform investing which is available at https://www.investing.com/". Los precios son públicos.
- **Métricas de calibración:** PICP, PINAW, MPICD, CWC (cobertura y ancho de intervalos). **No reporta ECE ni Brier score.**
- **Limitaciones reportadas:** los autores mencionan que se podría mejorar la eficiencia computacional y que se deberían integrar más variables macroeconómicas o eventos políticos.

## 5.2 Hiperparámetros reportados (valores finales)
- **Backbone:** PatchTST. **No se reportan capas, dimensiones, número de heads, patch length, etc.**
- **Capa de salida:** GMM con $K$ componentes. **No se reporta $K$.**
- **Offset:** prior normal con precisión $\beta$. **No se reporta $\beta$.**
- **Regularización bayesiana:** **no se reportan los priors** ($\alpha$, $\beta$) ni el método de inferencia (VI, MCMC, Laplace).
- **Función de pérdida:** **NO reportada.**
- **Normalización:** Z-score mencionado. **NO se reporta el timing.**
- **Isolation Forest / Lagrange:** **NO se reporta si se aplican antes o después del split.**
- **Semilla:** **NO reportada.**
- **Conjunto de validación:** **NO reportado.**

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **Ninguno explícito.** Se prueban 9 backbones y se elige el mejor por métricas out-of-sample. |
| Espacio de búsqueda | 9 backbones: DNN, RNN, CNN, TCN, LSTNet, NBEATS, NHITS, PatchTST, TFT. **No se reportan los hiperparámetros internos de cada backbone.** |
| Valores finales | PatchTST-BRDNN. **No se reportan capas, neuronas, LR, batch size ni $K$.** |
| Optimizador | **No reportado.** |
| Learning rate | **No reportado.** |
| Batch size | **No reportado.** |
| Épocas | **No reportado.** |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No reportado.** No hay conjunto de validación separado. Se comparan los 9 backbones directamente sobre el out-of-sample (abril 2021 – octubre 2023). **Esto es una bandera roja de selección de modelo sobre el test.**

**¿Sobre qué conjunto se evalúa el fitness?** **Sobre el out-of-sample.** La selección del mejor backbone (PatchTST) se hace mirando el desempeño en el test. **Esto constituye una fuga de selección de modelo.**

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **NO reportado.** Se describe el likelihood de la GMM y la regularización bayesiana, pero no se declara la función objetivo exacta (¿log-likelihood negativa? ¿ELBO? ¿MSE + regularización?).

### 5.3.4 Modelos de referencia
- **QRA, BNN, MQRNN, DeepAR, DDNN-JSU:** **no se reportan sus hiperparámetros** ni si se ajustaron con el mismo protocolo. **No se declara explícitamente que reciban los mismos inputs.**
- **No se reportan pruebas de significancia** entre estos modelos y PatchTST-BRDNN.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **No, en la práctica.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Backbone PatchTST (capas, heads, patch length) | No | Alto: no se puede replicar la arquitectura |
| Número de componentes GMM ($K$) | No | Alto: afecta la flexibilidad del modelo |
| Prior de la regularización bayesiana ($\alpha$, $\beta$) | No | Alto: afecta la estimación |
| Método de inferencia bayesiana (VI, MCMC, Laplace) | No | Alto: afecta la estimación |
| Función de pérdida | No | Alto: no se sabe qué se minimiza |
| Offset prior ($\beta$) | No | Medio: afecta el ajuste |
| Normalización (timing) | No | Medio: potencial fuga |
| Isolation Forest / Lagrange (timing) | No | Medio: potencial fuga |
| Hiperparámetros de los backbones | No | Alto: no se pueden replicar |
| Hiperparámetros de los modelos de referencia | No | Medio: comparación no verificable |
| Semilla | No | Medio: los modelos son estocásticos |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **No, sin un esfuerzo sustancial.** Los supuestos que habría que tomar son demasiados:
1. **PatchTST:** configuración canónica (3 capas, 4 heads, patch length 16, etc.).
2. **GMM:** $K=3$ o $K=5$ (valor típico).
3. **Regularización bayesiana:** prior $\alpha=1$, $\beta=1$; inferencia por VI o Laplace.
4. **Función de pérdida:** log-likelihood negativa de la GMM + término de regularización.
5. **Normalización:** Z-score ajustado solo con train.
6. **Isolation Forest / Lagrange:** aplicar solo con train (o declarar que se aplican al dataset completo y corregir la fuga).
7. **Semilla:** fijar una semilla.
8. **Modelos de referencia:** reimplementar QRA, BNN, MQRNN, DeepAR, DDNN-JSU con sus hiperparámetros canónicos.

**Riesgo de divergencia del desempeño reportado:** **muy alto.** El desempeño reportado está afectado por: (i) posible fuga por limpieza y normalización; (ii) selección del backbone sobre el test; (iii) hiperparámetros no reportados; (iv) sin pruebas de significancia. Sin estas correcciones, el desempeño sería distinto.

**Riesgo de divergencia de la arquitectura:** **alto.** La arquitectura PatchTST-BRDNN está descrita conceptualmente, pero faltan todos los hiperparámetros clave.

**Riesgo de divergencia del pipeline:** **alto.** Faltan el timing de limpieza y normalización, la función de pérdida, el método de inferencia bayesiana y los hiperparámetros de los backbones.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas —posible fuga por limpieza y normalización, selección del backbone sobre el test, hiperparámetros no reportados, función de pérdida no declarada, sin pruebas de significancia y ausencia de código—, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación sería una **aproximación muy gruesa**, no una réplica. **Pero esto es irrelevante para el benchmark de volatilidad porque el paper se excluye por E6.**

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **No.** El paper usa deep learning (PatchTST, BRDNN) pero **no pronostica volatilidad, varianza realizada ni proxy equivalente**. Pronostica la distribución del precio.
- **C2. Reporte suficiente:** **No.** Faltan la función de pérdida, los hiperparámetros de los backbones, el número de componentes GMM, los priors bayesianos, el método de inferencia, el timing de limpieza y normalización, y el código.
- **C3. Calidad editorial:** **Sí.** Q2 en JCR (Applied Economics, Routledge).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **Sí, parcialmente.** El protocolo es difícilmente reconstruible por la falta de hiperparámetros clave (GMM $K$, priors bayesianos, función de pérdida, capas del backbone).
- **E6. Solo precio, sin volatilidad:** **Sí.** El target es el precio de cierre, no la volatilidad, la varianza realizada ni una proxy equivalente. **Este es el criterio de exclusión aplicable.**
- **E7. GARCH como contribución principal:** **No aplica.**

**Resultado:** **Excluido por E6.**

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo (para el registro, aunque el paper se excluya):**
- **Target:** precio de cierre, no volatilidad. **No aplica al benchmark de volatilidad.** → **N/A**.
- **Partición:** cronológica 80/20 con fechas explícitas. → **bajo para la partición**.
- **Limpieza:** Isolation Forest + interpolación de Lagrange antes del split (timing no reportado). → **alto (posible look-ahead)**.
- **Normalización:** Z-score con timing no reportado. → **moderado (posible fuga)**.
- **Selección del backbone:** 9 backbones comparados sobre el test. → **alto (selección sobre el test)**.
- **Reporte selectivo:** reporta 9 backbones + 5 modelos de referencia, multi-horizonte y multi-nivel. Pero **sin varianza entre ejecuciones**. → **moderado**.
- **Inferencia:** **sin pruebas de significancia entre modelos.** Solo métricas puntuales. → **alto**.
- **Comparación:** los 9 backbones BRDNN reciben los mismos inputs (la serie de precios), pero **no se declara que QRA, BNN, MQRNN, DeepAR y DDNN-JSU reciban los mismos inputs**. → **moderado**.
- **Ablation:** se ablate el backbone y el offset, pero **no el componente GMM ni la regularización bayesiana**. La comparación contra BNN no es un ablation. → **alto (ablation incompleto)**.
- **Reproducibilidad:** sin código, sin función de pérdida, sin hiperparámetros clave, sin priors bayesianos, sin semilla. → **muy parcial**.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** Precios de Brent y WTI públicos (Investing.com). → **Sí**.
- **Reconstructibilidad del pipeline:** **no.** Faltan demasiados hiperparámetros y la función de pérdida. → **No**.
- **Viabilidad computacional:** **sí.** PatchTST + GMM es viable en una GPU estándar. → **Sí**.

**Clasificación: R-B (no reproducible por el equipo).** R1=Sí, R2=No, R3=Sí → no cumple R2=Sí. El estudio se conserva en el registro como referencia de la familia `Probabilistic DL (Mixture Density Network)`, pero **no entra al benchmark** por E6 y por R-B.

**Decisión provisional:**
- **Entra al benchmark:** **No.** Excluido por E6.
- **Tipo de entrada:** **N/A** (excluido).
- **Correcciones necesarias:** **N/A** (no aplica, porque el target no es volatilidad). Si en el futuro se quisiera adaptar al benchmark de volatilidad, habría que **cambiar el target a RV** y rehacer el estudio. Eso sería un paper distinto, no una corrección de este.
- **Brecha principal:** El estudio aporta la familia `Probabilistic DL (Mixture Density Network)` con un esquema PatchTST-BRDNN + GMM + regularización bayesiana + offset. **Esta familia no está representada en el benchmark de volatilidad** porque el paper predice precio, no volatilidad. Si el equipo quisiera incorporar esta familia, necesitaría un paper que aplique BRDNN (o una Mixture Density Network similar) a la **volatilidad realizada** como target. Ese paper no es este. La brecha entre el objetivo del paper (densidad del precio) y el objetivo del benchmark (volatilidad) es **de target, no de arquitectura**. Por eso la exclusión es limpia y no requiere correcciones.

**Nota final sobre el check permanente (inputs igualados y ablation):**
- **Inputs igualados:** dentro del bloque BRDNN (9 backbones), todos reciben la misma serie de precios. La comparación es justa **en inputs**. Pero la comparación contra QRA, BNN, MQRNN, DeepAR y DDNN-JSU **no declara explícitamente que reciban los mismos inputs**, lo que es una ambigüedad.
- **Ablation:** se ablate el backbone (9 opciones) y el offset (with vs without). **No se ablate el componente GMM** (¿una sola Gaussiana vs mezcla?) ni **la regularización bayesiana** (¿maximum likelihood vs inferencia bayesiana?). La comparación contra BNN no es un ablation porque BNN es un enfoque distinto. **El ablation es incompleto.**
- **Consecuencia:** la afirmación central del paper ("BRDNN mejora la precisión probabilística") **está parcialmente identificada** para el backbone y el offset, pero **no está identificada** para el componente GMM ni para la regularización bayesiana. Como el paper se excluye por E6, esta limitación se registra pero no afecta al benchmark de volatilidad.