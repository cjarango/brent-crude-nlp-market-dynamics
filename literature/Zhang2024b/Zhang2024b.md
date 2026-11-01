---
id: Zhang2024b
title: "Interval prediction of crude oil spot price volatility: An improved hybrid model integrating decomposition strategy, IESN and ARIMA."
authors: "Jinliang Zhang; Ziyi Liu"
year: 2024
doi: "10.1016/j.eswa.2024.124195"
journal: "Expert Systems With Applications"
editorial_quality: "Q1 (Pergamon)"
status: "Include"
exclusion_reason: "None"
problem_type: "Interval forecast"
horizon: "single-step"
proxy: "Squared returns (r²)"
architectural_family: "Reservoir"
architectural_variant: "ICEEMDAN-FE-FIG-ARIMA-WOA-IESN"
market_asset: "WTI y Brent Crude Oil Spot"
---

# Resumen
El estudio propone un modelo híbrido para la predicción por intervalos de la volatilidad del precio spot del crudo WTI y Brent. El pipeline incluye: (1) descomposición ICEEMDAN de la volatilidad; (2) reconstrucción de componentes con Fuzzy Entropy (FE); (3) Fuzzy Information Granulation (FIG) para extraer los límites inferior y superior de cada componente; (4) predicción de los componentes de alta frecuencia con un Echo State Network mejorado por Whale Optimization Algorithm (WOA-IESN) y de los de baja frecuencia con ARIMA; (5) suma de intervalos. Los autores reportan que el modelo supera a 17 benchmarks (M1–M17) en métricas de cobertura y ancho de intervalo (PICP, PINAW, PINRW, AW, IME, PICWC).

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **volatilidad proxy** del precio spot, calculada como el cuadrado del retorno logarítmico:
$$r_t = 100 \cdot \ln(x_t / x_{t-1}), \quad \text{Vol}_t = r_t^2$$
Los retornos al cuadrado son un **proxy de volatilidad legítimo** (estimador insesgado de la varianza condicional bajo condiciones débiles), pero extremadamente ruidoso: es una sola observación diaria, no RV de alta frecuencia ni varianza condicional GARCH. Se predice el **intervalo** [Low, Up] de la volatilidad en $t+1$.

**Fuente de los datos:**
- **WTI y Brent spot prices:** EIA (https://www.eia.gov) y WIND database (https://www.wind.com.cn/).
- **Precio de futuros de crudo:** WIND. Se usa como variable de entrada al IESN.
- **Tres frecuencias:** diaria (2 ene 2003 – 3 ene 2023, 5,150 obs), semanal (1,045 obs), mensual (245 obs).
- **Data availability:** los autores declaran “the data that has been used is confidential”. Los spot son públicos vía EIA; los futuros vía WIND son de pago.

**Partición Temporal:** split estático con conteos explícitos en la Sección 3.3. Para WTI diario: **824 train / 206 test**. Semanal: **168 train / 41 test**. Mensual: **40 train / 9 test**. **No hay conjunto de validación separado.** Sobre si se respetó la cronología: el texto indica “the front 824 data are used as the training sets, and the back 206 data are used as the test sets”, lo que sugiere que la división es cronológica (primeras observaciones para train, últimas para test). Sin embargo, el paper **no declara explícitamente** que se respetó el orden cronológico ni proporciona fechas exactas de corte. El test se usa una sola vez para evaluar, pero no se especifica si se usó para tuning.

**Nota sobre la granularidad de FIG:** la granulación con ventana de 5 días reduce las 5,150 observaciones diarias a 1,030 ventanas de 5 días. Cada ventana tiene (Low, R, Up). El split es **824 ventanas train / 206 ventanas test**, no 824 días / 206 días. El test cubre 206 × 5 ≈ 1,030 días. El paper no lo dice explícitamente, pero se deduce de la aritmética: 5,150 / 5 = 1,030, y 824 + 206 = 1,030.

**¿La predicción es rolling o hold-out? ¿Open-loop o recursiva?** El paper **no usa rolling origin ni walk-forward**, y **no declara** si la predicción es open-loop o recursiva. Lo que se puede afirmar:
1. **El modelo se entrena una sola vez** sobre las 824 ventanas del train y sus pesos quedan congelados. No hay reentrenamiento entre ventanas.
2. **La descomposición ICEEMDAN/FE/FIG se aplica una sola vez** sobre la serie completa antes del split, y los componentes resultantes se usan tal cual para train y test. No hay recálculo en cada origen.
3. **Se generan 206 predicciones one-step-ahead** sobre el test set, una por cada ventana de 5 días del test. Cada predicción usa el mismo modelo con los mismos pesos.

**Ambigüedad no resuelta: open-loop vs recursivo.** El paper **no especifica** cómo se construyen los inputs para cada una de las 206 predicciones:
- **Open-loop (con actuales):** para cada ventana $i$ del test, el modelo recibe las **observaciones reales** de las ventanas anteriores y predice el Low/Up de la ventana $i$. Cada predicción es independiente y usa historia real. Es el estándar para $h=1$.
- **Closed-loop (recursivo):** para cada ventana $i$, el modelo recibe sus **propias predicciones** de las ventanas anteriores y predice la ventana $i$. Propaga el error.

Dado que el paper predice $h=1$ (single-step) y describe el test como “the back 206 data are used as the test sets”, **lo más probable es open-loop con actuales**, pero **el paper no lo declara**. La diferencia importa: en open-loop el desempeño refleja la capacidad de predecir un paso con historia real; en recursivo refleja la capacidad de encadenar predicciones, y suele ser peor. Para el benchmark, se debe declarar explícitamente el esquema y aplicarlo igual a todos los modelos.

**Normalización:** **no se reporta** si se aplicó Min-Max, Z-score ni en qué momento se ajustó. El paper no menciona normalización en la metodología.

**Ingeniería de Características:** La Fig. 2 muestra la descomposición ICEEMDAN de **los 5,150 puntos completos**, no solo del train. El texto confirma: “the WTI spot volatility data is decomposed into several sub-series by ICEEMDAN” y luego “the components in the 1st interval with similar component complexity are reconstructed”. Todo el pipeline de ICEEMDAN → FE → FIG se aplica sobre la **serie completa antes del split**. Los componentes del train contienen información matemática del test. La descomposición tampoco es *rolling*: se aplica una sola vez sobre la serie completa.

**Uso de futuros como input:** El texto dice “historical data and futures prices are chosen as the influencing factors of volatility”. **No especifica** si el futuro es contemporáneo o rezagado. Si es contemporáneo, hay una fuente adicional de información no disponible en tiempo real para predecir $t+1$.

# 2. Arquitectura Final

**Etapa 1 — ICEEMDAN:** descompone la volatilidad en 10 IMFs + residuo (Fig. 2).
**Etapa 2 — FE:** calcula la fuzzy entropy de cada IMF y agrupa en 3 nuevas secuencias:
- New sequence 1: IMF1 + IMF2 (alta frecuencia).
- New sequence 2: IMF3 + … + IMF6 (baja frecuencia).
- New sequence 3: IMF7 + … + IMF10 (residuo/tendencia).

**Etapa 3 — FIG:** granulación triangular con ventana de 5 días sobre cada nueva secuencia para extraer Low, R (media) y Up. Función de afiliación triangular:

$$A(x, a, s, b) = \begin{cases} 0, & x \le a \\ \frac{x-a}{s-a}, & a < x \le s \\ \frac{b-x}{b-s}, & s < x \le b \\ 0, & x > b \end{cases}$$

donde $a$ = límite inferior, $b$ = límite superior, $s$ = media.

**Etapa 4 — Pronóstico:**
- **WOA-IESN:** predice Low y Up de la new sequence 1 (alta frecuencia). WOA optimiza los parámetros del reservoir. ESN canónico: input → reservoir (activación no especificada, presumiblemente tanh) → output (lineal). Inputs: historical data + futures prices (forma exacta no especificada).
- **ARIMA:** predice Low y Up de las new sequences 2 y 3 (baja frecuencia y residuo). Órdenes $(p,d,q)$ no reportados.

**Etapa 5 — Construcción del intervalo:** suma de los intervalos de cada componente:

$$\text{Intervalo final} = \left[\sum_p \text{Low}_p, \sum_p \text{Up}_p\right]$$

donde $p$ indexa las nuevas secuencias (1, 2, 3).


**Fig. 1** del paper muestra el flujo completo (four phases: Sample obtaining → Data decomposition and reconstruction → Component feature categorization → Volatility prediction interval construction).

# 3. Resultados y Comparaciones

**Modelos comparados (17 en total):**
- M1: EMD-FE-FIG-BP
- M2: EEMD-FE-FIG-ARIMA-BP
- M3: VMD-FE-FIG-ARIMA-LSTM
- M4: CEEMDAN-FE-FIG-ARIMA-ESN
- **Propuesto:** ICEEMDAN-FE-FIG-ARIMA-IESN
- M5–M8: variantes sin ARIMA (ELM, BP-ELM, BP-SVM, SVM-ELM).
- M9–M12: variantes con EMD, EEMD, VMD, CEEMDAN (mismo resto del pipeline).
- M13–M17: variantes sin ARIMA ni IESN (LSTM, ESN, BP, LSTM+ARIMA, ESN+ARIMA).
- **GARCH no incluido:** los autores argumentan que “GARCH family model focuses on point forecasting” y por eso no lo incluyen. Justificación débil: GARCH puede producir intervalos vía simulaciones o distribuciones condicionales.

**Métricas y Rendimiento:**
El estudio reporta seis métricas de evaluación de intervalos: PICP, PINAW, PINRW, AW, IME y PICWC.
- **WTI diario (modelo propuesto):** PICP=0.9029, PINAW=0.1367, PINRW=0.1716, AW=8.7881, IME=0.0015, PICWC=0.8205.
- **Brent diario (modelo propuesto):** PICP=0.9854, PINAW=0.4829, PINRW=0.6388, AW=7.4138, IME=0.0002, PICWC=2.8972.
- **Comparación:** el modelo propuesto obtiene el mejor PICWC en todos los escenarios. Frente a M4 (CEEMDAN), la mejora en PICWC es de 0.4875 en WTI diario.

**Pruebas de Significancia:**
**NO se reportan** pruebas de significancia estadística (ni Diebold-Mariano, ni MCS, ni bootstrap de intervalos). La comparación se basa únicamente en valores puntuales de las métricas.

**Sesgo de Reporte:**
Alto. Siete problemas:
1. **Look-ahead bias por ICEEMDAN/FE/FIG:** toda la descomposición, reconstrucción y granulación se aplica antes del split train/test, contaminando el train con información del futuro.
2. **Predicción hold-out sin reentrenamiento y con esquema de input no declarado:** el modelo se entrena una vez y genera las 206 predicciones del test con los mismos pesos. No hay rolling origin, walk-forward ni reentrenamiento entre ventanas. La descomposición tampoco se recalcula en cada origen. Además, **no se declara si la predicción es open-loop (con actuales) o recursiva (con predicciones propias)**. El desempeño reportado refleja un escenario hold-out, no un esquema de predicción dinámico, y la ambigüedad del esquema de input impide saber si el error se propaga o no.
3. **Uso de futuros como input:** la justificación económica es razonable, pero no se especifica si el futuro es contemporáneo (lo cual sería circular) o rezagado.
4. **Sin pruebas de significancia:** el modelo “gana” por valores puntuales de PICWC, pero no se demuestra que las diferencias sean estadísticamente significativas.
5. **Exclusión de GARCH sin justificación sólida.**
6. **Sin varianza entre ejecuciones:** un solo split, una sola corrida, sin semillas.
7. **Equidad de inputs no verificable:** el modelo propuesto usa “historical data + futures prices”, pero el paper **no especifica si los benchmarks M1–M17 también usan precios de futuros**. Si los benchmarks no los usan, el modelo propuesto recibe información adicional que los demás no tienen, lo que haría que la comparación no sea justa. El paper solo compara el modelo propuesto con y sin futuros (experimento comparativo 3), pero **no compara benchmarks con y sin futuros**. La equidad de la comparación no se puede verificar con la información reportada.

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros de ESN, ARIMA ni del pipeline completo.
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.** Los datos son declarados confidenciales.
- **Data availability:** “the data that has been used is confidential”. Los spot son públicos vía EIA; los futuros vía WIND son de pago.
- **Métricas de calibración:** reporta PICP, PINAW, PINRW, AW, IME y PICWC. No usa QLIKE ni ECE.
- **Limitaciones reportadas:** los autores no reportan limitaciones explícitas en la sección de conclusiones. Mencionan como trabajos futuros la aplicación a otros activos energéticos.

## 5.2 Hiperparámetros reportados (valores finales)
- **ICEEMDAN:** SNR=0.2, número de ruidos=50, iteraciones máximas=5,000.
- **FE:** embedding dimension=1, similarity tolerance=0.2, weight=2.
- **FIG:** ventana de 5 días (ciclo semanal del mercado).
- **PICWC:** $\mu=0.8$, $\eta=15$, $\alpha=0.1$, $\beta=6$.
- **ESN (valores finales):** **NO reportados.** El paper no publica los valores finales de reservoir size, input scale, spectral radius ni sparsity que seleccionó WOA.
- **ARIMA:** órdenes $(p,d,q)$ **NO reportados**.
- **Activaciones del ESN:** **NO reportadas** (se asume tanh en reservoir y lineal en salida, pero no se declara).
- **Feedback del ESN ($W_{back}$):** **NO reportado** si se usa o no.
- **Método de cálculo de $W_{out}$:** **NO reportado** (se asume ridge regression o pseudo-inversa, pero no se declara).

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Algoritmo | Whale Optimization Algorithm (WOA) |
| Hiperparámetros del WOA | population = 15, iteraciones máximas = 20 |
| Parámetros optimizados | 4: reservoir size, input scale, spectral radius, sparsity |
| Espacio de búsqueda | reservoir size $\in [10, 300]$; input scale $\in [0.01, 1]$; spectral radius $\in [0.01, 1]$; sparsity $\in [0.01, 1]$ |
| Valores finales seleccionados | **NO reportados** |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No.** El paper **no menciona validación cruzada** en ningún lugar. **No hay conjunto de validación separado.** La búsqueda se hace con WOA sobre un conjunto que **no se especifica**.

**¿Sobre qué conjunto se evalúa el fitness?** **NO reportado.** El paper solo dice “the error is used as the fitness of the WOA according to the characteristics of the ESN”. No aclara si el error se calcula sobre el conjunto de entrenamiento (824 ventanas), el conjunto de prueba (206 ventanas), la serie completa (1,030 ventanas) o algún subconjunto del train. Esta ambigüedad es crítica: si el fitness se calculó sobre el test, hay fuga directa; si se calculó sobre la serie completa, hay fuga por descomposición y por uso del test; si se calculó sobre el train, es defendible pero subóptimo (no hay validación separada para detectar sobreajuste). **El paper no permite determinar cuál de los tres escenarios se usó.**

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **NO reportado con precisión.** El paper dice “the error is used as the fitness” pero **no especifica**:
- ¿Qué métrica de error? (¿MSE? ¿RMSE? ¿MAE? ¿PICWC? ¿PINAW? ¿otra?)
- ¿Sobre qué variable? (¿Low? ¿Up? ¿Ambos? ¿El intervalo [Low, Up]? ¿La volatilidad puntual $r_t^2$? ¿El ancho del intervalo?)
- ¿Es un error promedio, máximo, o una combinación ponderada?

El IESN predice **Low** y **Up** de la new sequence 1, y cada uno es un modelo separado (o una salida de 2 dimensiones, pero no se especifica). Si WOA optimiza cada modelo por separado, la variable objetivo sería el error de predicción de Low y el error de predicción de Up, respectivamente. Si optimiza ambos conjuntamente, la variable objetivo sería una combinación de ambos errores. **El paper no lo dice.**

La métrica de evaluación reportada es **PICWC** (que combina cobertura y ancho), pero **no hay evidencia de que PICWC sea la variable objetivo que WOA optimiza**. Es solo la métrica con la que se evalúa el desempeño final. Podría ser que WOA optimice MSE sobre Low/Up y luego se evalúe con PICWC, o que optimice directamente PICWC. **No se puede saber.**

### 5.3.4 ARIMA
Los órdenes $(p,d,q)$ **no se reportan**, ni el criterio de selección. El paper no dice si usó `auto.arima`, inspección de ACF/PACF, criterios de información u otro método. Para reproducir, se tendría que hacer un análisis propio de los modelos ARIMA (por ejemplo, inspección de ACF/PACF o selección por AIC/BIC), pero no se puede saber si ese análisis coincide con el que hicieron los autores.

### 5.3.5 ESN
El paper reporta los rangos de búsqueda pero no los valores finales. La arquitectura del ESN (número de capas, tipo de conexiones, función de activación del reservoir, uso de feedback) **no se detalla**. No es posible reconstruir el ESN exacto sin estos datos.

## 5.4 Limitaciones detectadas por el revisor
1. **Look-ahead bias por descomposición:** corregible aplicando ICEEMDAN/FE/FIG de forma *rolling* (recalcular en cada origen usando solo datos pasados).
2. **Predicción hold-out sin reentrenamiento:** el modelo se entrena una vez y genera las 206 predicciones del test con los mismos pesos. No hay rolling origin ni walk-forward. Para el benchmark, se debe usar rolling origin o expanding window para que la evaluación sea realista.
3. **Esquema de predicción no declarado:** no se especifica si la predicción es open-loop (usa observaciones reales como input) o recursiva (usa predicciones propias como input). Dado que $h=1$, lo más probable es open-loop, pero no se declara. Para el benchmark, se debe declarar explícitamente y aplicar el mismo esquema a todos los modelos.
4. **Sin validación:** no hay conjunto de validación separado. El test (206 ventanas) se usa para evaluar; no se especifica si se usó para tuning. El WOA necesita un conjunto para evaluar el fitness, y no se declara cuál.
5. **Variable objetivo de WOA no especificada:** no se sabe qué error se optimiza, ni sobre qué variable (Low, Up, ambos, PICWC, otro).
6. **Futuros contemporáneos:** si el input es contemporáneo, hay fuga adicional. Debe rezagarse.
7. **Sin significancia:** imposible afirmar superioridad con solo métricas puntuales.
8. **GARCH excluido:** benchmark econométrico relevante ausente.
9. **Valores finales no reportados:** sin los valores finales del ESN ni los órdenes ARIMA, la reimplementación exacta no es posible; solo una aproximación con supuestos.
10. **Equidad de inputs no verificable:** no se especifica si los benchmarks M1–M17 usan precios de futuros. Si no los usan, el modelo propuesto tiene una ventaja informacional sobre ellos, lo que sesga la comparación a su favor.

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **No de forma exacta.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura del ESN (capas, conexiones, feedback) | No | Alto: no se puede reconstruir la red exacta |
| Activaciones del ESN | No | Medio: se asume tanh/lineal, pero no se declara |
| Método de cálculo de $W_{out}$ | No | Medio: se asume ridge, pero no se declara |
| Valores finales del ESN (seleccionados por WOA) | No | Alto: no se puede replicar la configuración óptima |
| Variable objetivo de WOA (qué error, sobre qué variable) | No | Alto: no se puede replicar el criterio de optimización |
| Conjunto sobre el que se evalúa el fitness del WOA | No | Alto: no se sabe si usó train, test o serie completa |
| Método de validación (¿CV? ¿hold-out?) | No | Alto: no hay validación; no se puede saber cómo se seleccionaron los hiperparámetros |
| Órdenes ARIMA $(p,d,q)$ | No | Alto: no se puede replicar el componente lineal |
| Criterio de selección del ARIMA | No | Medio: se asume AIC/BIC, pero no se declara |
| Semilla del WOA | No | Medio: WOA es estocástico; sin semilla, no se reproduce |
| Normalización (si la hay) | No | Medio: afecta la escala del target y de los inputs |
| Estructura exacta de los inputs al ESN | No | Alto: “historical data + futures prices” es ambiguo |
| Estrategia de predicción (¿rolling? ¿hold-out?) | Hold-out (implícito) | Alto: entrena una vez, 206 predicciones con el mismo modelo congelado, sin rolling ni recálculo de la descomposición |
| Esquema de predicción (open-loop vs recursivo) | No declarado | Alto: no se sabe si el error se propaga entre ventanas o si cada predicción usa historia real |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Sí, con un riesgo de divergencia alto.** Los supuestos que habría que tomar:

1. **ESN canónico:** input → reservoir (tanh) → output (lineal), sin feedback, con ridge regression para $W_{out}$.
2. **WOA:** population=15, iteraciones=20. **Fitness:** se asume MSE sobre train (el paper no especifica; se declara como supuesto). **Variable objetivo:** se asume MSE sobre Low y MSE sobre Up por separado (el paper no especifica; se declara como supuesto).
3. **Validación:** no hay validación en el paper. Para el benchmark, se debe añadir un conjunto de validación separado o usar validación cruzada temporal para seleccionar hiperparámetros. Esto es una corrección necesaria.
4. **Predicción:** el paper usa hold-out (entrena una vez, predice las 206 ventanas del test con el mismo modelo, sin reentrenar ni recalcular la descomposición) y no declara si es open-loop o recursivo. Para el benchmark, se debe usar rolling origin o expanding window con recálculo de la descomposición en cada origen, y declarar explícitamente el esquema de input.
5. **ARIMA:** el paper no reporta los órdenes $(p,d,q)$ ni el criterio de selección. Para reproducir, se tendría que hacer un análisis de los modelos ARIMA (por ejemplo, inspección de ACF/PACF o selección por AIC/BIC), pero el paper no reporta cómo lo hicieron ellos.
6. **Futuros:** rezagar 1 período para evitar fuga contemporánea.
7. **ICEEMDAN/FE/FIG:** aplicar de forma *rolling* o solo sobre train.
8. **FIG:** ventana de 5 días, función triangular como se describe.
9. **ICEEMDAN:** 10 IMFs (el paper muestra 10 componentes, pero no especifica el criterio de parada).
10. **Semilla:** fijar una semilla para el WOA (no reportada).

**Riesgo de divergencia del desempeño reportado:** **alto.** El PICWC de 0.8205 está inflado por tres factores: (i) look-ahead bias de la descomposición, (ii) esquema hold-out sin reentrenamiento ni recálculo de la descomposición en cada origen (que en series de tiempo suele inflar el desempeño), (iii) ausencia de validación para detectar sobreajuste. A esto se suma la **ambigüedad del esquema de input** (open-loop vs recursivo): si el paper usó open-loop con actuales, el desempeño es más optimista que si hubiera usado recursivo.

**Riesgo de divergencia de la arquitectura:** **medio.** El pipeline (ICEEMDAN → FE → FIG → WOA-ESN + ARIMA → suma) es reconstruible, pero los detalles del ESN y del ARIMA se tienen que asumir.

**Riesgo de divergencia del pipeline:** **bajo.** Los pasos están claros; la corrección de la fuga es mecánica (rolling decomposition). La corrección del esquema de predicción (hold-out → rolling, open-loop/recursivo declarado) es mecánica pero cambia el desempeño.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas, variable objetivo de WOA no especificada, conjunto de fitness no declarado, ausencia de validación, esquema de predicción (open-loop vs recursivo) no declarado, valores finales de hiperparámetros no reportados, arquitectura del ESN no detallada y órdenes ARIMA no reportados, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación es una **aproximación** al modelo original, no el mismo modelo ni la misma arquitectura. El benchmark debe tratar la versión corregida como una **variante** del modelo original, no como una réplica exacta, y reportar la brecha entre ambos desempeños como evidencia del impacto de las carencias de reporte. La comparación con el artículo se limita a la **familia arquitectónica** (reservoir + descomposición + granulación + ARIMA), no al desempeño puntual ni a la configuración exacta.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad del crudo con ESN (deep learning) como componente central.
- **C2. Reporte suficiente:** **Tal vez.** Reporta parámetros parciales (ICEEMDAN, FE, FIG, WOA, rangos de ESN), pero no la arquitectura exacta del ESN, ni los órdenes ARIMA, ni los valores finales del reservoir, ni la normalización, ni la variable objetivo de WOA, ni el conjunto de fitness, ni el esquema de predicción. Sin código.
- **C3. Calidad editorial:** **Sí.** Q1 en JCR, Elsevier.
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El pipeline es reconstruible con supuestos estándar (ESN canónico, análisis propio de ARIMA, fitness asumido sobre train, predicción asumida hold-out open-loop). La partición está reportada.
- **E6. Solo precio, sin volatilidad:** **No aplica**. Predice volatilidad (proxy $r_t^2$).
- **E7. GARCH como contribución principal:** **No aplica**. GARCH no es contribución principal; se excluye como benchmark con justificación débil, pero no es el núcleo.

**Resultado:** Incluido.

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Partición:** cronológica por conteos (front/back), pero sin declaración explícita de orden ni fechas de corte, y sin validación. → ambiguo.
- **Predicción:** hold-out (entrena una vez, 206 predicciones con el mismo modelo congelado, sin rolling ni recálculo de la descomposición). Esquema de input (open-loop vs recursivo) no declarado. → alto.
- **Fuga:** confirmada por descomposición global (Fig. 2 = 5,150 puntos). → alta.
- **Fuga por futuros:** no verificable si es contemporáneo o rezagado. → riesgo.
- **Reporte selectivo:** un solo split, una sola corrida, sin varianza. → alto.
- **Inferencia:** sin pruebas de significancia. → alto.
- **Comparación:** 17 benchmarks, pero todos variantes del mismo pipeline y sin GARCH; además, no se verifica que los benchmarks usen los mismos inputs (futuros) que el modelo propuesto. → alto.
- **Reproducibilidad:** sin código, hiperparámetros parciales, sin valores finales del ESN ni del ARIMA, sin detalles del ESN, sin variable objetivo de WOA, sin conjunto de fitness, sin método de validación, sin esquema de predicción. → parcial.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **parcial.** WTI/Brent spot vía EIA son públicos. Futuros vía WIND son de pago; existe proxy público (NYMEX, Investing.com, EIA). → **Tal vez / Sí con proxy**.
- **Reconstructibilidad del pipeline:** **sí, con supuestos.** ESN canónico (input → reservoir tanh → output lineal, ridge regression), análisis propio de ARIMA (selección de órdenes por AIC/BIC o inspección de ACF/PACF), FIG ventana=5. La fuga es corregible. El esquema de predicción hold-out se corrige a rolling en el benchmark. → **Sí**.
- **Viabilidad computacional:** **Tal vez.** ICEEMDAN con 50 ruidos y 5,000 iteraciones sobre 5,150 puntos es costoso pero viable; la versión *rolling* multiplica el costo. WOA con pop=15 e iter=20 es ligero. → **Tal vez**.

**Decisión provisional:**
- **Entra al benchmark:** sí.
- **Tipo de entrada:** con corrección.
- **Correcciones necesarias:**
    - ICEEMDAN/FE/FIG de forma *rolling* o solo sobre train.
    - Predicción con rolling origin o expanding window, con recálculo de la descomposición en cada origen, y esquema de input declarado explícitamente (open-loop con actuales o recursivo), aplicado igual a todos los modelos.
    - Rezagar el precio de futuros para evitar fuga.
    - Especificar arquitectura del ESN (capas, activaciones, feedback) y hacer un análisis propio de ARIMA con criterio declarado.
    - Declarar la variable objetivo de WOA (qué error, sobre qué variable) y el conjunto de fitness (train o validación).
    - Añadir un conjunto de validación separado o usar validación cruzada temporal para seleccionar hiperparámetros.
    - Declarar los valores finales del ESN o, en su defecto, reoptimizar con WOA bajo un protocolo reproducible (misma semilla, mismo fitness sobre validación).
    - Añadir pruebas de significancia en el benchmark.
    - Garantizar que todos los modelos del benchmark usen el mismo conjunto de inputs (incluidos o excluidos los futuros) para asegurar la equidad de la comparación.
- **Brecha principal:** la falta de especificación de la variable objetivo de WOA, la ausencia de validación, la predicción hold-out sin rolling y con esquema de input no declarado, la falta de especificación del ESN y ARIMA, la ausencia de valores finales de hiperparámetros, la ausencia de pruebas de significancia y la equidad de inputs no verificable. El desempeño reportado (PICWC=0.8205) no se usa como referencia: se reimplementa la arquitectura y se reporta la brecha entre el original (con fuga, hold-out, sin validación) y el corregido (rolling, con validación, con tests). **La reimplementación no es el mismo modelo ni la misma arquitectura que el artículo: es una variante aproximada, y los resultados no son directamente comparables con los reportados.**