---
id: Gadhi2024
title: "Improving Volatility Forecasting: A Study through Hybrid Deep Learning Methods with WGAN."
authors: "Adel Hassan A. Gadhi; Shelton Peiris; David E. Allen"
year: 2024
doi: "10.3390/jrfm17090380"
journal: "Journal of Risk and Financial Management (JRFM)"
editorial_quality: "Q2 (MDPI)"
status: "Include"
exclusion_reason: "None"
problem_type: "Point forecast (volatility proxy)"
horizon: "single-step (implícito)"
proxy: "Rolling 5-day standard deviation of daily log returns, annualized — SESGADO (no es RV)"
architectural_family: "Hybrid (GARCH + DL + Generative)"
architectural_variant: "GARCH-ANN / GARCH-LSTM-ANN / GARCH-BLSTM-ANN + WGAN-GP data augmentation"
market_asset: "Brent Crude Oil Spot (y temperatura de Sídney)"
---

# Resumen
El estudio evalúa el uso de una **Wasserstein GAN con gradient penalty (WGAN-GP)** para aumentar los datos de entrenamiento y mejorar el pronóstico de volatilidad. Se aplican modelos híbridos que combinan GARCH con redes neuronales (ANN, LSTM, BLSTM) y se comparan con sus versiones aumentadas con WGAN-GP. La evaluación se realiza sobre dos conjuntos de datos: los retornos del precio spot del crudo Brent y la temperatura de Sídney (Australia). Los resultados reportan mejoras en las métricas de error al incorporar WGAN-GP (R² de 0.980 para ANN-WGAN en Brent). **El estudio se incluye porque es Q2 (JRFM, MDPI), predice volatilidad (proxy) con DL, y aporta la familia `Hybrid (GARCH + DL + Generative)` con WGAN-GP data augmentation. Sin embargo, presenta limitaciones metodológicas severas**: (i) el proxy es una desviación estándar móvil de 5 días sobre retornos diarios, un estimador SESGADO y extremadamente ruidoso, no comparable con RV; (ii) look-ahead bias CONFIRMADO por normalización Min-Max antes del split (el pseudocódigo del Apéndice A.1 lo dice explícitamente); (iii) partición 80/20 sin orden temporal declarado (no hay shuffle=False ni fechas de corte); (iv) comparación no aislada: los modelos con WGAN no incluyen GARCH y los modelos con GARCH no incluyen WGAN, por lo que no se puede separar el efecto del WGAN del efecto del GARCH; (v) sin pruebas de significancia entre modelos; (vi) sin tuning; (vii) sin varianza entre ejecuciones; (viii) MAPE de 16,864% en temperatura (error de cálculo); (ix) sin código. El ablation es parcial: compara con/sin GARCH, con/sin WGAN, con/sin LSTM/BLSTM, pero no aísla el WGAN-GP de un aumento de datos trivial (bootstrap, ruido gaussiano).

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **volatilidad**, definida como la desviación estándar de los retornos logarítmicos diarios en una ventana móvil de 5 días, anualizada (Apéndice A.1, paso 8):
$$V_t = \text{std}(\log\_return[t-4:t]) \cdot \sqrt{252}$$
Para el crudo Brent, se calcula a partir de los retornos; para la temperatura, se usa la serie de temperatura directamente (sin ventana móvil explícita). **No se especifica** el uso de estimadores de rango de alta frecuencia (Garman-Klass, Parkinson) ni de RV intradía.

**EL ESTIMADOR ES SESGADO.** **NO predicen Volatilidad Realizada (RV).** La volatilidad objetivo es una **desviación estándar móvil de 5 días sobre retornos diarios**, un estimador **SESGADO e inconsistente** por cinco fuentes que se acumulan:
1. **Solo precios de cierre:** ignora toda la varianza intradía y el *overnight gap*. La RV de 5 minutos captura la variación dentro del día, que en el crudo es una fracción sustancial de la varianza total.
2. **Ventana extremadamente corta (N=5):** la desviación estándar muestral con 5 observaciones tiene una **varianza de muestreo enorme**. El estimador es extremadamente ruidoso y su error estándar es grande. No es un estimador eficiente de la volatilidad del período.
3. **Target mismatch (suavizado):** estima la volatilidad **promedio de los últimos 5 días**, no la volatilidad condicional en $t+1$. En un mercado con clustering, este promedio subestima la volatilidad cuando viene un spike y la sobreestima cuando viene una calma.
4. **Sesgo por la raíz cuadrada (Jensen):** al aplicar la raíz cuadrada se obtiene un estimador sesgado a la baja de la desviación estándar.
5. **Anualización por $\sqrt{252}$ sobre una ventana de 5 días:** amplifica el ruido de la ventana corta.

**Conclusión:** el proxy es **SESGADO** y no comparable con RV. Los resultados de pronóstico dependen de un target ruidoso y mal especificado.

**Fuente de los datos:**
- **Crudo Brent:** precios diarios de Investing.com. Periodo: 5 de junio de 2012 – 27 de junio de 2022 (2,638 observaciones).
- **Temperatura:** lecturas de Sídney, Australia, de Wunderground.com. Periodo: enero 2013 – diciembre 2023 (3,992 observaciones).
- **Data availability:** "The datasets generated and/or analyzed during the current study are available from the corresponding author on request". Los precios son públicos (Investing.com); la temperatura es pública (Wunderground).

**Partición Temporal (posible fuga):**
- **Split estático 80/20 (train/test).** **No se especifica** si se respetó el orden cronológico. El texto y el pseudocódigo (Apéndice A.1) **no mencionan** `shuffle=False`, fechas de corte, Time Series Split, Rolling Origin ni Walk-forward. **Tampoco hay conjunto de validación separado.**
- **Esta omisión es crítica:** si el split es aleatorio (o si no se documenta el orden), hay fuga de información futura hacia el train. No se puede descartar.

**LOOK-AHEAD BIAS CONFIRMADO (normalización antes del split):**
El pseudocódigo del Apéndice A.1 lo dice **explícitamente**:
- **Paso 10:** "Normalize the data using MinMaxScaler".
- **Paso 11:** "Split the data into training (80%) and testing (20%) sets".

**Esto es look-ahead bias confirmado:** la normalización Min-Max se aplica sobre el **100% del dataset** antes de dividir. Las estadísticas del test set (mínimo y máximo) influyen en el escalado del train set. **El modelo "ve" el rango del futuro.** Este es un error metodológico grave y explícito en el pseudocódigo.

**WGAN-GP data augmentation (posible fuga adicional):**
- El pseudocódigo del WGAN-GP (Apéndice A.4) dice "Input: Training data", lo cual es **correcto** si se entrena solo con train.
- Pero el texto principal (Sección 3.5) dice "We first train a WGAN-GP model on our historical data", lo cual es **ambiguo**: ¿"historical data" = train set o dataset completo?
- Si el WGAN-GP se entrena con el dataset completo, las muestras sintéticas incorporan información del test set, y la fuga es **doble** (normalización + WGAN).

**Consecuencia para la comparación:**
- **Los modelos con WGAN reciben datos aumentados** (sintéticos + reales). Los modelos sin WGAN reciben solo datos reales. La comparación mide el efecto del **aumento de datos**, no solo el de la arquitectura WGAN-GP.
- **Los modelos con GARCH reciben la volatilidad condicional del GARCH como input** (además de los retornos). Los modelos sin GARCH no la reciben. La comparación mide el efecto del **GARCH como feature**, no solo el de la arquitectura híbrida.
- **No hay un control con "datos aumentados por bootstrap" o "ruido gaussiano"** para aislar el efecto del WGAN-GP del efecto de tener más datos. **La afirmación "el WGAN-GP mejora la precisión" no está identificada.**

**Ingeniería de Características:**
- **Input:** retornos logarítmicos diarios (Ecuación 1) + (en los modelos GARCH) la volatilidad condicional del GARCH(1,1).
- **Sin variables exógenas:** no hay features de noticias, sentimiento, redes complejas ni otras variables financieras.
- **Sin selección de features:** se usan los retornos y, en su caso, la volatilidad GARCH.
- **GARCH(1,1):** estimado por máxima verosimilitud sobre el train set. **No se reportan** los órdenes (p,q) más allá de (1,1), ni el criterio de selección, ni pruebas de diagnóstico de residuos.

**Tuning de Hiperparámetros:**
**CERO optimización sistemática.** El paper reconoce: "We used fixed architectures for our neural networks. Further research could explore optimal architecture selection through techniques like neural architecture search". **No hay Grid Search, Bayesian Optimization ni validación cruzada.** Los hiperparámetros de la ANN/LSTM/BLSTM se fijan sin justificación (y muchos no se reportan).

**Hiperparámetros reportados (parciales):**
- **ANN:** 1 capa de entrada, 2 capas ocultas, 1 capa de salida. **No reporta** el número exacto de neuronas ni la función de activación.
- **LSTM:** 2 capas LSTM (32 y 64 unidades) + 2 capas Dense (32 con ReLU, 1 de salida). Optimizador Adam, pérdida MSE. **No reporta** learning rate, batch size, epochs ni early stopping.
- **WGAN-GP:** `critic_steps=5`, `lambda_gp=10`, `epochs=10000`, `noise_dim` no reportado. **Parcialmente reportado.**
- **GARCH(1,1):** parámetros estimados reportados en Tabla 6 (crudo) y Tabla 11 (temperatura).

# 2. Arquitectura Final (GARCH + DL + WGAN-GP)

El estudio compara **16 configuraciones** de modelos:

1. **GARCH(1,1) puro:** modelo base univariado.
2. **DL puro (sin GARCH, sin WGAN):** ANN, LSTM-ANN, BLSTM-ANN.
3. **Híbridos GARCH (sin WGAN):** GARCH-ANN, GARCH-LSTM-ANN, GARCH-BLSTM-ANN.
4. **DL + WGAN (sin GARCH):** ANN-WGAN, LSTM-ANN-WGAN, BLSTM-ANN-WGAN.
5. **Híbridos GARCH + WGAN:** solo para temperatura (GARCH-ANN-WGAN, GARCH-LSTM-ANN-WGAN, GARCH-BLSTM-ANN-WGAN). **Para crudo Brent, NO se reportan modelos GARCH+WGAN.** Esta es una **inconsistencia crítica**: el paper afirma que "the proposed hybrid models with WGAN especially those incorporating GARCH... are more accurate", pero **no reporta los resultados de GARCH+WGAN para crudo**.

**Arquitectura interna de la ANN (parcial):**
- 1 capa de entrada, 2 capas ocultas, 1 capa de salida.
- **Número de neuronas: NO reportado.**
- **Función de activación: NO reportada** (se menciona ReLU solo para la capa Dense de la LSTM).

**Arquitectura interna de la LSTM:**
- 2 capas LSTM (32 y 64 unidades) + 2 Dense (32 con ReLU, 1 de salida).
- Optimizador Adam, pérdida MSE.
- **Learning rate, batch size, epochs: NO reportados.**

**WGAN-GP:**
- Generador y crítico (discriminador) con `critic_steps=5`, `lambda_gp=10`, `epochs=10000`.
- **No se reporta** la arquitectura del generador ni del crítico (número de capas, neuronas).
- **No se reporta** el `noise_dim`.
- **No se reporta** cuántas muestras sintéticas se generan.
- **No se reporta** si el WGAN-GP se entrena solo con train o con el dataset completo.

**Sobre el check permanente (ablación):**
- **Sí compara con/sin GARCH:** ANN vs GARCH-ANN, LSTM-ANN vs GARCH-LSTM-ANN, BLSTM-ANN vs GARCH-BLSTM-ANN. **Bien cubierto.**
- **Sí compara con/sin WGAN:** ANN vs ANN-WGAN, LSTM-ANN vs LSTM-ANN-WGAN, BLSTM-ANN vs BLSTM-ANN-WGAN. **Bien cubierto.**
- **Sí compara con/sin LSTM/BLSTM:** ANN vs LSTM-ANN vs BLSTM-ANN. **Bien cubierto.**
- **No ablate el tipo de aumento de datos:** no hay control con "datos aumentados por bootstrap" o "ruido gaussiano" para aislar el efecto del WGAN-GP del efecto de tener más datos. **La afirmación "el WGAN-GP mejora" no está identificada.**
- **No ablate el número de muestras sintéticas:** no se varía el tamaño del aumento de datos.
- **No ablate los hiperparámetros del WGAN-GP:** el paper reconoce que "the impact of different WGAN-GP hyperparameters on forecasting performance was not extensively explored".

# 3. Resultados y Comparaciones

**Modelos comparados (16 configuraciones):**
- **Baselines:** GARCH(1,1), ANN, LSTM-ANN, BLSTM-ANN.
- **Híbridos GARCH:** GARCH-ANN, GARCH-LSTM-ANN, GARCH-BLSTM-ANN.
- **DL + WGAN:** ANN-WGAN, LSTM-ANN-WGAN, BLSTM-ANN-WGAN.
- **Híbridos GARCH + WGAN (solo temperatura):** GARCH-ANN-WGAN, GARCH-LSTM-ANN-WGAN, GARCH-BLSTM-ANN-WGAN.

**Métricas de Evaluación:**
R², MSE, MAE, RMSE, MAPE.

**Ganador y Rendimiento (crudo Brent, sin WGAN, Tabla 7):**
- **GARCH-ANN:** R² = 0.734, MSE = 0.203, MAE = 0.295, RMSE = 0.450, MAPE = 14.497%.
- **GARCH-LSTM-ANN:** R² = 0.725, MSE = 0.206, MAE = 0.293, RMSE = 0.453, MAPE = 13.872%.
- **GARCH-BLSTM-ANN:** R² = 0.731, MSE = 0.204, MAE = 0.294, RMSE = 0.452, MAPE = 23.453%.
- **Inconsistencia:** el paper dice que GARCH-BLSTM-ANN es el mejor sin WGAN, pero **GARCH-ANN tiene el R² más alto (0.734)** y el MSE más bajo (0.203). GARCH-LSTM-ANN tiene el MAE más bajo (0.293) y el MAPE más bajo (13.872%). **No hay un ganador claro**, y el paper no discute esta inconsistencia.

**Ganador y Rendimiento (crudo Brent, con WGAN, Tabla 8):**
- **ANN-WGAN:** R² = 0.980, MSE = 0.004, MAE = 0.049, RMSE = 0.065, MAPE = 0.317%.
- **LSTM-ANN-WGAN:** R² = 0.977, MSE = 0.050, MAE = 0.050, RMSE = 0.069, MAPE = 0.263%.
- **BLSTM-ANN-WGAN:** R² = 0.977, MSE = 0.005, MAE = 0.054, RMSE = 0.070, MAPE = 0.284%.
- **Sospechosamente alto:** el R² de 0.980 y el MSE de 0.004 para ANN-WGAN son **50× mejores** que el mejor modelo sin WGAN (GARCH-ANN, MSE = 0.203). Esto es **altamente sospechoso** y probablemente un **artefacto de la fuga de datos** (normalización antes del split y/o WGAN-GP entrenado con el dataset completo).

**Ganador y Rendimiento (temperatura, sin WGAN, Tabla 9):**
- **BLSTM-ANN:** R² = 0.9993, MSE = 0.032, MAE = 0.141, RMSE = 0.178, MAPE = 0.229%.
- **ANN:** R² = 0.9966, MSE = 0.229, MAE = 0.026, RMSE = 0.479, MAPE = inf% (MAPE infinito, error de cálculo).
- **GARCH-ANN:** R² = 0.7657, MSE = 11.303, MAE = 2.571, RMSE = 3.362, MAPE = 3.981%.
- **Inconsistencia:** BLSTM-ANN tiene el mejor R² (0.9993) sin GARCH ni WGAN. Pero el paper no lo destaca.

**Ganador y Rendimiento (temperatura, con WGAN, Tabla 10):**
- **ANN-WGAN:** R² = 0.999, MSE = 0.000, MAE = 0.013, RMSE = 0.016, MAPE = 0.018%.
- **GARCH-ANN-WGAN:** R² = 0.999, MSE = 1.318×10⁻⁷, MAE = 0.000, RMSE = 0.011, MAPE = **16,864.385%**.
- **MAPE absurdo:** el MAPE de 16,864% para GARCH-ANN-WGAN es **imposible** y sugiere un **error de cálculo** (probablemente división por cero o valores normalizados negativos). El paper no lo discute.

**Pruebas de Significancia:**
- **No se reporta ninguna prueba de significancia formal** (ni Diebold-Mariano, ni Model Confidence Set, ni Clark-West). La comparación se basa únicamente en los valores puntuales de R², MSE, MAE, RMSE y MAPE.
- **No se reporta la varianza ni la desviación estándar** entre ejecuciones. Los modelos son estocásticos (inicialización de pesos, WGAN-GP), pero no se reporta dispersión.

**Sesgo de Reporte:**
Muy alto. Diez problemas:
1. **Look-ahead bias confirmado:** normalización Min-Max antes del split (Apéndice A.1, pasos 10–11). El modelo "ve" el rango del futuro.
2. **Partición sin orden temporal declarado:** no hay `shuffle=False`, no hay fechas de corte, no hay Time Series Split. No se puede descartar fuga de información futura.
3. **WGAN-GP con timing ambiguo:** "historical data" puede ser train o dataset completo. Si es completo, la fuga es doble.
4. **Comparación no aislada (WGAN vs GARCH):** los modelos con WGAN no incluyen GARCH y los modelos con GARCH no incluyen WGAN. No se puede separar el efecto del WGAN del efecto del GARCH.
5. **Sin control de aumento de datos trivial:** no hay bootstrap ni ruido gaussiano para aislar el efecto del WGAN-GP del efecto de tener más datos.
6. **R² de 0.980 sospechoso:** probablemente artefacto de la fuga, no una mejora genuina.
7. **MAPE de 16,864% en temperatura:** error de cálculo, no discutido.
8. **Sin pruebas de significancia:** solo métricas puntuales.
9. **Sin varianza entre ejecuciones:** los modelos son estocásticos.
10. **Sin código.**

**Fortalezas del reporte:**
- Comparación de 16 configuraciones (aunque no todas estén bien aisladas).
- Uso de WGAN-GP para data augmentation (familia no representada en el corpus).
- Evaluación en dos dominios (financiero y ambiental) como validación cruzada de dominio.
- Reporte de parámetros GARCH (Tabla 6 y 11).
- Reconocimiento explícito de limitaciones (arquitecturas fijas, hiperparámetros del WGAN-GP no explorados, generalización a otros activos).
- Pseudocódigo de los componentes (Apéndice A), aunque revela la fuga.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número total de parámetros de la ANN, LSTM, BLSTM ni WGAN-GP.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.** Los autores indican que los datos están disponibles bajo petición, pero no mencionan la disponibilidad del código.
- **Data availability:** "The datasets generated and/or analyzed during the current study are available from the corresponding author on request". Los precios de Brent son públicos (Investing.com); la temperatura es pública (Wunderground).
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast).
- **Limitaciones reportadas:** arquitecturas fijas (sin neural architecture search), hiperparámetros del WGAN-GP no explorados, generalización a otros activos pendiente.

## 5.2 Hiperparámetros reportados (valores finales)
- **ANN:** 1 capa de entrada, 2 capas ocultas, 1 capa de salida. **NO se reportan** neuronas ni función de activación.
- **LSTM:** 2 capas LSTM (32 y 64 unidades) + 2 Dense (32 con ReLU, 1 de salida). Adam, MSE. **NO se reportan** learning rate, batch size, epochs ni early stopping.
- **BLSTM:** **no se reporta** la arquitectura interna (se asume similar a LSTM pero bidireccional).
- **WGAN-GP:** `critic_steps=5`, `lambda_gp=10`, `epochs=10000`. **NO se reportan** `noise_dim`, arquitectura del generador/crítico, número de muestras sintéticas, ni si se entrena solo con train o con dataset completo.
- **GARCH(1,1):** parámetros estimados reportados en Tabla 6 (crudo) y Tabla 11 (temperatura). **NO se reportan** pruebas de diagnóstico de residuos.
- **Normalización:** Min-Max. **CONFIRMADO que se aplica antes del split** (Apéndice A.1, pasos 10–11).
- **Semilla:** **NO reportada.**
- **Conjunto de validación:** **NO reportado.**

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **Ninguno sistemático.** Arquitecturas fijas. |
| Espacio de búsqueda | **NO reportado.** |
| Valores finales | ANN (2 capas ocultas, sin detalle); LSTM (32/64 + 32/1); WGAN-GP (critic_steps=5, lambda_gp=10, epochs=10000). |
| Optimizador | Adam (para las redes) |
| Learning rate | **NO reportado.** |
| Batch size | **NO reportado.** |
| Épocas | 100 (Apéndice A.3) / 10000 (WGAN-GP) |
| Función de pérdida | MSE (redes) / WGAN-GP loss (generador) |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No reportado.** No hay conjunto de validación separado ni validación cruzada.

**¿Sobre qué conjunto se evalúa el fitness?** **No aplica.** No hay búsqueda de hiperparámetros.

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **MSE** (función de pérdida de las redes, Apéndice A.3). **Reportada explícitamente.**

### 5.3.4 Modelos de referencia
- **GARCH(1,1):** parámetros reportados, pero **sin diagnóstico de residuos**.
- **ANN, LSTM-ANN, BLSTM-ANN:** **sin hiperparámetros completos** (learning rate, batch size, epochs).
- **WGAN-GP:** **sin arquitectura del generador/crítico**, sin `noise_dim`, sin número de muestras sintéticas.
- **No se declara explícitamente** si todos los modelos reciben el mismo conjunto de features (los híbridos con GARCH reciben la volatilidad condicional; los modelos con WGAN reciben datos aumentados).

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **No, en la práctica.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura ANN (neuronas, activación) | No | Alto: no se puede replicar la red |
| Arquitectura LSTM (capas, unidades) | Sí (32/64 + 32/1) | Bajo: está reportada |
| Learning rate, batch size, epochs | No | Alto: afecta el entrenamiento |
| Early stopping | No | Medio: afecta la regularización |
| Arquitectura WGAN-GP (generador/crítico) | No | Alto: no se puede replicar el generador |
| `noise_dim`, número de muestras sintéticas | No | Alto: afecta el aumento de datos |
| Timing del WGAN-GP (train vs dataset completo) | Ambiguo | **Alto: posible fuga doble** |
| Normalización (timing) | Sí (antes del split, Apéndice A.1) | **Alto: fuga confirmada** |
| Orden temporal del split | No declarado | **Alto: posible fuga futura** |
| GARCH (órdenes, diagnóstico) | Parcial (1,1; sin diagnóstico) | Medio: afecta las predicciones GARCH |
| Semilla | No | Medio: los modelos son estocásticos |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **No, sin un esfuerzo sustancial.** Los supuestos que habría que tomar son demasiados:
1. **ANN:** 2 capas ocultas con 64 y 32 neuronas, ReLU.
2. **LSTM/BLSTM:** ya reportados (32/64 + 32/1), Adam lr=0.001, batch_size=32, 100 épocas, early stopping con paciencia 10.
3. **WGAN-GP:** generador y crítico con 2 capas ocultas de 128 neuronas, `noise_dim=100`, 10,000 muestras sintéticas, entrenado solo con train.
4. **Normalización:** Min-Max ajustado **solo con train** (corrigiendo la fuga).
5. **Split:** cronológico 80/20 (asumiendo que no se respetó, corregirlo).
6. **Semilla:** fijar una.

**Riesgo de divergencia del desempeño reportado:** **muy alto.** El desempeño reportado está inflado por: (i) look-ahead bias por normalización antes del split; (ii) posible fuga por orden temporal no declarado; (iii) posible fuga doble si el WGAN-GP se entrenó con el dataset completo; (iv) sin tuning sistemático. Sin estas fugas, el R² de 0.980 probablemente caería a niveles cercanos a los de los modelos sin WGAN (R² ~0.7). **La brecha será enorme.**

**Riesgo de divergencia de la arquitectura:** **alto.** Faltan la arquitectura de la ANN, el learning rate, el batch size, la arquitectura del WGAN-GP y el `noise_dim`.

**Riesgo de divergencia del pipeline:** **muy alto.** El timing de la normalización y del WGAN-GP es ambiguo o está confirmado como fuga.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, look-ahead bias confirmado por normalización antes del split, orden temporal del split no declarado, timing del WGAN-GP ambiguo, arquitectura de la ANN no reportada, learning rate/batch size/epochs no reportados, arquitectura del WGAN-GP no reportada, `noise_dim` no reportado, sin semilla y ausencia de código, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación sería una **aproximación muy gruesa**, no una réplica. El benchmark debe tratar la versión corregida (con split cronológico y normalización solo con train) como una **variante** del modelo original y reportar la brecha entre ambos desempeños.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad (proxy) del crudo con ANN/LSTM/BLSTM y WGAN-GP como componentes centrales.
- **C2. Reporte suficiente:** **Parcial.** Reporta la arquitectura de la LSTM y los parámetros del GARCH, pero no la arquitectura de la ANN, el learning rate, el batch size, los epochs, la arquitectura del WGAN-GP, el `noise_dim`, el timing del WGAN-GP, el orden temporal del split ni el código. **La fuga por normalización está confirmada en el pseudocódigo.**
- **C3. Calidad editorial:** **Sí.** Q2 en JCR/Scopus (Journal of Risk and Financial Management, MDPI).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El protocolo es reconstruible con supuestos (partición 80/20, arquitectura LSTM reportada, GARCH reportado). La fuga es un problema de validez, no de reconstruibilidad. Se incluye con reservas.
- **E6. Solo precio, sin volatilidad:** **No aplica.** Predice volatilidad (proxy).
- **E7. GARCH como contribución principal:** **No aplica.** GARCH es un componente del híbrido, no la contribución principal.

**Resultado:** **Incluido** (con reservas).

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** desviación estándar móvil de 5 días sobre retornos diarios, anualizada. **SESGADO** (solo precios de cierre, ventana extremadamente corta y ruidosa, target mismatch, Jensen, anualización amplifica ruido). → **alto**.
- **Partición:** split 80/20 **sin orden temporal declarado** (no hay `shuffle=False`, no hay fechas de corte). → **alto (posible fuga futura)**.
- **Normalización:** Min-Max **antes del split** (confirmado en Apéndice A.1). → **alto (fuga confirmada)**.
- **WGAN-GP:** timing ambiguo ("historical data" puede ser train o dataset completo). → **moderado-alto (posible fuga doble)**.
- **Predicción:** single-step implícito, sin rolling. → **moderado**.
- **Reporte selectivo:** reporta 16 configuraciones, 2 dominios, 5 métricas. Pero **sin varianza entre ejecuciones** y con **MAPE de 16,864%** (error). → **alto**.
- **Inferencia:** **sin pruebas de significancia entre modelos.** Solo métricas puntuales. → **alto**.
- **Comparación:** **no aislada.** Los modelos con WGAN no incluyen GARCH; los modelos con GARCH no incluyen WGAN. La comparación no separa el efecto del WGAN del efecto del GARCH. **Sin control de aumento de datos trivial** (bootstrap, ruido gaussiano). → **alto**.
- **Ablation:** compara con/sin GARCH, con/sin WGAN, con/sin LSTM/BLSTM. **Bien cubierto para esas dimensiones**, pero **no aísla el WGAN-GP de un aumento de datos trivial** ni varía el número de muestras sintéticas ni los hiperparámetros del WGAN-GP. → **moderado-alto**.
- **Reproducibilidad:** sin arquitectura de la ANN, sin learning rate/batch size/epochs, sin arquitectura del WGAN-GP, sin `noise_dim`, sin orden temporal del split, sin semilla, sin código, con fuga confirmada. → **muy parcial**.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** Precios de Brent públicos (Investing.com); temperatura pública (Wunderground). → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** Arquitectura LSTM y GARCH reportados, pero faltan ANN, learning rate, batch size, epochs, WGAN-GP, `noise_dim`, orden temporal y normalización. → **Parcial**.
- **Viabilidad computacional:** **sí.** ANN/LSTM/BLSTM con 2 capas + WGAN-GP con 10,000 épocas. Viable en GPU estándar. → **Sí**.

**Clasificación: R-B (no reproducible por el equipo).** R1=Sí, R2=Tal vez, R3=Sí → no cumple R2=Sí. El estudio se conserva en la revisión como referencia de la familia `Hybrid (GARCH + DL + Generative)` con WGAN-GP data augmentation. Si el equipo decide reimplementarlo con **split cronológico** y **normalización solo con train** (corrigiendo la fuga) y los supuestos declarados, entra al benchmark como **variante corregida**.

**Decisión provisional:**
- **Entra al benchmark:** sí, con reservas (si se reimplementa con split cronológico y normalización solo con train).
- **Tipo de entrada:** **con corrección**.
- **Correcciones necesarias:**
    - **(PRIORITARIA) Corregir el look-ahead bias por normalización:** ajustar el Min-MaxScaler **solo con el train set** (no con el dataset completo). Sin esta corrección, el desempeño reportado no es comparable.
    - **(PRIORITARIA) Declarar y garantizar el orden temporal del split:** usar split cronológico 80/20 con fechas de corte explícitas. Documentar `shuffle=False`. Sin esta corrección, no se puede descartar fuga de información futura.
    - **(PRIORITARIA) Declarar el timing del WGAN-GP:** entrenar el generador **solo con el train set** (no con el dataset completo). Sin esta corrección, la fuga puede ser doble.
    - **(PRIORITARIA) Aislar el efecto del WGAN-GP del aumento de datos trivial:** incluir un control con "datos aumentados por bootstrap" o "ruido gaussiano" (mismo número de muestras sintéticas, sin WGAN-GP). Sin este control, la afirmación "el WGAN-GP mejora" **no está identificada**.
    - **(PRIORITARIA) Reportar la arquitectura de la ANN:** número de neuronas en las capas ocultas y función de activación.
    - **(PRIORITARIA) Reportar la arquitectura del WGAN-GP:** número de capas y neuronas del generador y del crítico, `noise_dim`, número de muestras sintéticas.
    - **Reportar learning rate, batch size y epochs** de la ANN, LSTM y BLSTM.
    - **Reportar el número de capas del BLSTM** (se asume similar a LSTM, pero no se declara).
    - **Reportar el early stopping** (si se usa).
    - **Reportar la semilla** para las redes y el WGAN-GP.
    - **Reportar la varianza/desviación estándar** entre ejecuciones.
    - **Añadir pruebas de significancia entre modelos** (Diebold-Mariano, Model Confidence Set).
    - **Corregir el MAPE de 16,864% en temperatura** (error de cálculo).
    - **Reportar los resultados de GARCH+WGAN para crudo Brent** (actualmente solo se reportan para temperatura, lo cual es una inconsistencia).
    - **Reportar los órdenes del GARCH** (se asume (1,1), pero no se declara) y pruebas de diagnóstico de residuos.
    - **Añadir QLIKE** como métrica de evaluación de volatilidad, además de R², MSE, MAE, RMSE y MAPE.
    - **Liberar el código** o, en su defecto, reportar todos los valores finales de hiperparámetros y la semilla.

- **Brecha principal:** El estudio aporta la familia `Hybrid (GARCH + DL + Generative)` con WGAN-GP data augmentation, que **no está representada en el corpus**. **La brecha es de identificación y de validez:** el look-ahead bias por normalización antes del split (confirmado en el pseudocódigo) y la posible fuga por orden temporal no declarado inflan el desempeño a niveles sospechosamente altos (R² de 0.980). Además, la comparación **no aísla el efecto del WGAN-GP del efecto del GARCH** (los modelos con WGAN no incluyen GARCH; los modelos con GARCH no incluyen WGAN) ni del **efecto de tener más datos** (no hay control con bootstrap o ruido gaussiano). El benchmark debe reimplementar el modelo con **split cronológico**, **normalización solo con train** y **WGAN-GP entrenado solo con train**, e incluir **un control con aumento de datos trivial** para aislar el efecto del WGAN-GP. Sin esas correcciones, la afirmación central del paper ("el WGAN-GP mejora la precisión") **no está identificada** y la comparación debe tratarse como **no concluyente**. Las demás carencias (arquitectura de la ANN, learning rate, batch size, arquitectura del WGAN-GP, `noise_dim`, semilla, varianza entre ejecuciones, MAPE absurdo en temperatura) son deficiencias de reporte que afectan la reproducibilidad, pero no la identificación del efecto; se corrigen en la reimplementación y se documentan en la Sección 8.