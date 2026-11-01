---
id: AlFattah2019
title: "Artificial Intelligence Approach for Modeling and Forecasting Oil-Price Volatility"
authors: "Saud M. Al-Fattah"
year: 2019
doi: "10.2118/195584-PA"
journal: "SPE Reservoir Evaluation & Engineering"
editorial_quality: "Q2 (JCR 2024 - Engineering, Petroleum)"
status: "Include"
exclusion_reason: "None"
problem_type: "Point forecast (monthly volatility, multivariate regression)"
horizon: "single-step (1-month)"
proxy: "Squared monthly log returns (r²) — TARGET INADECUADO: estimador extremadamente ruidoso de σ² (SE ≈ 1.41σ²), con sesgo pequeño hacia arriba por μ² (~1% para WTI mensual), e inconsistente con la definición del paper (intro: 'standard deviation'; fórmula: r²). No es RV."
architectural_family: "MLP (Multilayer Perceptron)"
architectural_variant: "GANNATS (GA + MLP + Time Series)"
market_asset: "WTI Crude Oil"
---

# Resumen
El estudio desarrolla el modelo **GANNATS** (Genetic Algorithm + Artificial Neural Network + Time Series) para pronosticar la volatilidad mensual del crudo WTI. La arquitectura central es un **MLP (Multilayer Perceptron) de tres capas** (20 neuronas de entrada, 14 en la capa oculta, 1 de salida), entrenado con el algoritmo Quasi-Newton y función de activación sigmoide. Un **Algoritmo Genético (GA)** se usa para seleccionar las 20 variables exógenas más importantes de un conjunto inicial de 58 (fundamentales de oferta/demanda, macroeconomía). Los resultados reportan un RMSE de 0.5247 y un DPI (Directional Predictability Indicator) del 87.9% en el set de prueba. **El estudio se incluye porque es Q2 (SPE Reservoir Evaluation & Engineering), predice volatilidad (proxy) del crudo con una red neuronal, usa partición cronológica con validación y aporta la familia `MLP (Multilayer Perceptron)` con selección de features por GA**. 

Sin embargo, presenta limitaciones severas: (i) el target es r² (un solo retorno mensual al cuadrado), un **target inadecuado**: es un estimador **extremadamente ruidoso** de σ² (su error estándar, ≈ 1.41σ², es mayor que el propio parámetro que estima), con un **sesgo pequeño** hacia arriba por μ² (~1% para WTI mensual), y es **inconsistente con la definición del propio paper** (la introducción dice "standard deviation", pero la fórmula usa r²); además, no es RV ni una desviación estándar móvil; (ii) no hay comparación cuantitativa contra baselines estándar (GARCH, LSTM); (iii) la normalización y el GA pueden haberse aplicado sobre el dataset completo antes del split (fuga potencial, no confirmada); (iv) sin pruebas de significancia entre modelos (solo Jarque-Bera de residuos); (v) sin código; (vi) sin learning rate reportado; (vii) datos mensuales (220 observaciones), lo cual limita la capacidad de capturar dinámicas de alta frecuencia.** El ablation es parcial: cubre la profundidad de la red, el algoritmo de entrenamiento, la función de activación y la selección de features, pero **no ablate el target (r² vs desviación estándar vs RV), la frecuencia de datos, el tamaño del conjunto de features ni incluye baselines externos**. El check permanente de **inputs igualados** se cumple dentro de las comparaciones internas del paper (todas las variantes reciben las mismas 20 variables), pero **no contra baselines externos** (que no se incluyen).

# 1. Método y Datos

**¿Qué se predice exactamente?**
La **volatilidad mensual** del crudo WTI, definida en la Ecuación 2 como el **retorno logarítmico mensual al cuadrado**:
$$v_t = r_t^2$$
donde $r_t = \ln(P_t / P_{t-1})$ (Ecuación 1). **No es RV** (no usa datos intradía) y **no es una desviación estándar móvil** (es un solo retorno al cuadrado). Además, es **inconsistente con la definición del propio paper**: la introducción define volatilidad como "the standard deviation of price returns over the sample time frame", pero la Ecuación 2 usa r².

**EL TARGET ES INADECUADO (tres problemas, en orden de gravedad):**

1. **Ruido extremo (problema principal).** r² es un estimador **extremadamente ruidoso** de σ². Para retornos con $\varepsilon_t \sim iid(0,1)$, $\text{Var}(r_t^2) \approx 2\sigma^4$, de modo que el error estándar de r² es $\text{SE}(r^2) \approx \sqrt{2}\,\sigma^2 \approx 1.41\,\sigma^2$. **El error estándar es mayor que el propio parámetro que se intenta estimar.** La relación señal/ruido es menor que 1. Es el estimador más crudo posible de volatilidad: una sola observación, sin ventana, sin suavizado. Esto no se corrige con más datos (el ruido es de muestreo, no de estimación); se corrige cambiando el target (ventana móvil o RV intradía).

2. **Inconsistencia con la definición del paper (problema de especificación).** El paper define volatilidad como **σ** ("standard deviation") en la introducción, pero la Ecuación 2 usa **r²**, que estima **σ²**, no σ. Para estimar σ a partir de un solo retorno habría que usar $|r_t|$, que por la desigualdad de Jensen es **sesgado a la baja** por un factor $\sqrt{2/\pi} \approx 0.798$ (≈ −20% para retornos normales). Es decir: **r² no estima el objeto que el paper dice estimar.** Es un problema de especificación, no solo de ruido.

3. **Sesgo pequeño hacia arriba para σ² (problema menor).** Bajo $r_t = \mu + \sigma_t \varepsilon_t$, $E[r_t^2] = \mu^2 + \sigma^2$. Por tanto, r² es **sesgado hacia arriba** como estimador de σ², con sesgo = μ². Para el WTI mensual 1994–2012 (μ ≈ 0.88%, σ ≈ 8.29%, según Tabla 1 del paper), el sesgo es $\mu^2/\sigma^2 \approx 1.1\%$. **Es pequeño y no es el problema principal**, pero conviene registrarlo.

**Conclusión:** el target es **inadecuado** por ruido extremo (SE > parámetro), por inconsistencia con la definición del paper (σ vs r²) y, en menor medida, por sesgo (μ²). Los resultados de pronóstico dependen de un target con relación señal/ruido < 1, lo que infla artificialmente el error de cualquier modelo y hace que las comparaciones entre modelos sean poco informativas (todos predicen ruido).

**Fuente de los datos:**
- **Precios:** WTI Crude Oil Futures (primer contrato mensual), US Energy Information Administration (EIA). Periodo: enero 1994 – abril 2012 (220 observaciones mensuales).
- **Variables exógenas (58 iniciales):** producción de crudo US, producción OPEP, inventarios US, consumo OCDE, GDP US, consumo de Rusia/India/China, capacidad ociosa OPEP, stocks mundiales, tipos de cambio, tipos de interés, etc. Frecuencia mensual.
- **Data availability:** los datos de la EIA son públicos. Las variables macro son públicas (FRED, World Bank, etc.). **No se reporta** la disponibilidad del código.

**Partición Temporal:**
- **Train:** 70% (enero 1994 – octubre 2006).
- **Validation:** 15% (noviembre 2006 – julio 2009).
- **Test:** 15% (agosto 2009 – abril 2012).
- **Split cronológico.** **Correcto.** Es una de las fortalezas del paper: respeta el orden temporal y separa validation de test.

**Normalización (fuga potencial, no confirmada):**
El paper indica que aplicó normalización media/desviación estándar. Sin embargo, en el flujo de trabajo (Figura 2), el paso "Normalization" aparece en la **Fase II (Data Mining/Preprocessing)**, **antes** del paso "Partitioning sets". Esto sugiere que normalizaron el dataset completo antes de dividirlo, lo cual constituiría un look-ahead bias. **No se puede confirmar** si usaron únicamente las estadísticas del train set. **Ambigüedad con riesgo potencial de fuga.**

**Selección de features (GA) (fuga potencial, no confirmada):**
El Algoritmo Genético (GA) se usa para seleccionar las 20 variables más importantes de las 58 iniciales. El paso "Feature Selection" aparece en la **Fase III**, **después** de la partición (Fase II). Sin embargo, **no se reporta** si el GA fue evaluado únicamente en el train set o si usó el validation/test set para seleccionar las 20 variables. **Ambigüedad con riesgo potencial de fuga.**

**Tuning de Hiperparámetros:**
- **Early stopping:** monitoreado por el validation set (15%). **Correcto.**
- **Selección de arquitectura:** se probaron 1, 2 y 3 capas ocultas (seleccionando 1), algoritmos GD/CG/Quasi-Newton (seleccionando Quasi-Newton), y funciones sigmoide/arctan/exponencial/identidad (seleccionando sigmoide). **No se reporta** sobre qué conjunto se evaluaron estas selecciones (¿validation? ¿test?). Si fue sobre test, hay fuga.
- **Learning rate:** **NO reportado.**
- **Número de épocas:** máximo 1,000, con early stopping (error residual 1e-6). **Reportado.*

# 2. Arquitectura Final (GANNATS: GA + MLP)

El modelo propuesto, denominado **GANNATS**, tiene la siguiente estructura:

1. **Selección de features (GA):** reduce 58 variables a 20 mediante un Algoritmo Genético. Las 20 variables finales incluyen: cambio en inventarios US, producción US, producción OPEP, inventarios OCDE, GDP US, consumo de Rusia/India/China, consumo OCDE, capacidad ociosa OPEP, entre otras.
2. **MLP (Multilayer Perceptron):** 3 capas (20 neuronas de entrada, 14 en la capa oculta, 1 de salida).
3. **Entrenamiento:** Quasi-Newton, sigmoide, early stopping (error residual 1e-6 o 1,000 épocas).

**Arquitectura interna:**
- **Capas:** 1 de entrada (20 nodos), 1 oculta (14 nodos), 1 de salida (1 nodo).
- **Función de activación:** sigmoide (reportada, pero no se especifica si se aplica solo en la capa oculta o también en la de salida).
- **Algoritmo de entrenamiento:** Quasi-Newton (seleccionado tras probar GD y CG).
- **Early stopping:** monitoreado por validation set.
- **Función de pérdida:** MSE (implícita en la descripción del entrenamiento).
- **Learning rate:** **NO reportado.**
- **Número de épocas:** máximo 1,000, con early stopping.

**Sobre el check permanente (ablación de la arquitectura):**
- **Sí ablate el número de capas ocultas:** 1 vs 2 vs 3. **Bien cubierto.**
- **Sí ablate el algoritmo de entrenamiento:** GD vs CG vs Quasi-Newton. **Bien cubierto.**
- **Sí ablate la función de activación:** sigmoide vs arctan vs exponencial vs identidad. **Bien cubierto.**
- **Sí ablate el método de selección de features:** GA vs forward stepwise vs backward stepwise vs PCA. **Bien cubierto.**
- **No ablate el número de neuronas en la capa oculta:** 14 se fija sin análisis de sensibilidad (¿qué pasa con 8, 20, 30?).
- **No ablate el tamaño del conjunto de features:** 20 se fija sin analizar 10, 30, 40, etc.
- **No ablate el target:** r² vs desviación estándar vs RV.
- **No ablate la frecuencia de datos:** mensual vs diario vs intradía.
- **No incluye baselines externos:** no compara contra GARCH, LSTM, etc.

**Conclusión:** el ablation es **parcial**. Cubre bien las dimensiones de arquitectura (profundidad, algoritmo, activación, selección de features), pero **no cubre el target, la frecuencia de datos, el tamaño del conjunto de features ni los baselines externos**.

# 3. Resultados y Comparaciones

**Modelos comparados:**
- **Variantes del propio GANNATS:** 1, 2, 3 capas ocultas; GD, CG, Quasi-Newton; sigmoide, arctan, exponencial, identidad; GA, forward stepwise, backward stepwise, PCA.
- **No hay comparación cuantitativa contra baselines externos** (GARCH, LSTM, etc.). El paper menciona la literatura pero no presenta una tabla comparativa.

**Métricas de Evaluación:**
- **Er:** mean relative percentage error.
- **Ea:** mean absolute percentage error.
- **Erms:** root mean squared error.
- **DPI:** Directional Predictability Indicator (mide la precisión de la dirección: subida/bajada).

**Ganador y Rendimiento (test set):**
- **Erms = 0.5247.**
- **DPI = 87.9%.**
- **Ea = 0.4105.**
- **Er = -0.0729.**
- Para el dataset completo: Erms = 0.6199, DPI = 83.6%.

**Sobre la interpretación de Erms:**
El target está normalizado (media 0, desviación 1). Un Erms de 0.5247 en unidades normalizadas significa que el error es aproximadamente 0.52 desviaciones estándar. Esto no es directamente comparable con RMSE de otros estudios que usan RV en unidades originales.

**Pruebas de Significancia:**
- **Jarque-Bera** para normalidad de residuos: JB = 21.02, p = 0.00003. Los residuos no se rechazan como normales al 1%.
- **No se reporta Diebold-Mariano, Model Confidence Set ni Clark-West.** No hay comparación estadística entre modelos.

**Sesgo de Reporte:**
Alto. Ocho problemas:
1. **Target inadecuado:** r² es un estimador **extremadamente ruidoso** de σ² (SE ≈ 1.41σ², mayor que el parámetro), con **sesgo pequeño** hacia arriba por μ² (~1% para WTI mensual), e **inconsistente con la definición del paper** (intro: "standard deviation"; fórmula: r²). No es RV ni desviación estándar móvil. El ruido extremo (relación señal/ruido < 1) es el problema principal; el sesgo es secundario.
2. **Sin comparación cuantitativa contra baselines:** no hay tabla contra GARCH, LSTM, etc.
3. **Normalización con timing ambiguo:** posible fuga (normalización antes del split).
4. **GA con timing ambiguo:** posible fuga (GA evaluado sobre train o dataset completo).
5. **Selección de arquitectura sin conjunto declarado:** no se sabe si se usó validation o test para elegir 1 capa, Quasi-Newton, sigmoide.
6. **Sin pruebas de significancia entre modelos:** solo Jarque-Bera de residuos.
7. **Learning rate no reportado.**
8. **Sin código.**

**Fortalezas del reporte:**
- **Partición cronológica correcta** (70/15/15) con validation set separado.
- **Early stopping** monitoreado por validation set.
- **Ablation de arquitectura** (profundidad, algoritmo, activación, selección de features).
- **Feature selection con GA** comparado con stepwise y PCA.
- **DPI** como métrica de dirección (relevante para trading).
- **Análisis de residuos** (histograma, normal probability plot, Jarque-Bera).
- **Reconocimiento de limitaciones** (datos mensuales, falta de comparación con modelos econométricos).
- **Data availability** (EIA público).

# 4. Complejidad Computacional
- **Parámetros:** no reporta el número de parámetros del MLP (pesos y sesgos).
- **FLOPs:** no reporta.
- **Complejidad Asintótica:** no reporta.
- **Tiempo de Entrenamiento:** no reporta.
- **Hardware:** no reporta.

# 5. Calidad y Reproducibilidad

## 5.1 Código y datos
- **Código disponible:** **No.**
- **Data availability:** EIA público; variables macro públicas. **El código no se menciona.**
- **Métricas de calibración:** no reporta ECE ni cobertura de intervalos (es point forecast).
- **Limitaciones reportadas:** datos mensuales (recomiendan frecuencias diarias o semanales); falta de comparación con modelos econométricos.

## 5.2 Hiperparámetros reportados (valores finales)
- **MLP:** 1 capa de entrada (20 nodos), 1 capa oculta (14 nodos), 1 capa de salida (1 nodo). Sigmoide. Quasi-Newton. Early stopping (error residual 1e-6 o 1,000 épocas).
- **GA:** **NO se reportan** los parámetros del GA (tamaño de población, número de generaciones, tasa de mutación, tasa de cruce).
- **Normalización:** media/desviación estándar. **NO se reporta el timing** (antes o después del split).
- **Semilla:** **NO reportada.**
- **Learning rate:** **NO reportado.**

## 5.3 Búsqueda de hiperparámetros

### 5.3.1 Algoritmo y parámetros del optimizador
| Aspecto | Valor reportado |
|---|---|
| Método | **Selección manual** (1 vs 2 vs 3 capas; GD vs CG vs Quasi-Newton; sigmoide vs arctan vs exponencial vs identidad). |
| Espacio de búsqueda | 3 arquitecturas × 3 algoritmos × 4 activaciones = 36 combinaciones (más variaciones de GA/stepwise/PCA). |
| Valores finales | 1 capa oculta (14 nodos), Quasi-Newton, sigmoide. |
| Optimizador | Quasi-Newton |
| Learning rate | **NO reportado.** |
| Batch size | **NO reportado** (se asume batch completo, dado que es MLP con 220 observaciones). |
| Épocas | Máximo 1,000, con early stopping. |
| Función de pérdida | MSE (implícita). |

### 5.3.2 Método de ajuste y validación
**¿Usaron validación cruzada?** **No se menciona validación cruzada.** Se usa un **validation set** (15%) para early stopping. **Correcto para early stopping**, pero **no se declara** si se usó para seleccionar la arquitectura (1 capa, Quasi-Newton, sigmoide).

**¿Sobre qué conjunto se evalúa el fitness?** **Early stopping:** validation set. **Selección de arquitectura:** **NO reportado** (¿validation o test?).

### 5.3.3 Variable objetivo a optimizar
**¿Qué se optimiza exactamente?** **MSE** (función de pérdida del MLP). **Reportada implícitamente** (no se declara explícitamente, pero es la estándar para MLP de regresión).

### 5.3.4 Modelos de referencia
- **No hay baselines externos** (GARCH, LSTM, etc.).
- **Variantes internas:** 1/2/3 capas, GD/CG/Quasi-Newton, sigmoide/arctan/exponencial/identidad, GA/stepwise/PCA.
- **No se declara** si todas las variantes reciben el mismo conjunto de 20 features (se asume que sí, porque el GA se aplica una vez y se usan las mismas 20 para todas).

# 6. Evaluación de suficiencia para reproducción

**¿Se puede reproducir el modelo final tal como se reporta?** **Parcialmente.** Faltan elementos críticos:

| Elemento | ¿Reportado? | Impacto en la reproducción |
|---|---|---|
| Arquitectura MLP (capas, nodos) | Sí (20-14-1) | Bajo: está reportada |
| Función de activación | Sí (sigmoide) | Bajo: está reportada |
| Algoritmo de entrenamiento | Sí (Quasi-Newton) | Bajo: está reportado |
| Early stopping | Sí (validation set, 1e-6 o 1,000 épocas) | Bajo: está reportado |
| Learning rate | No | Medio: afecta el entrenamiento |
| Parámetros del GA | No | Alto: no se puede replicar la selección de features |
| Normalización (timing) | No | Medio: potencial fuga |
| GA (timing) | No | Medio: potencial fuga |
| Selección de arquitectura (conjunto) | No | Alto: no se sabe si hubo fuga |
| Semilla | No | Medio: el MLP es estocástico |
| Código | No | Alto: no hay referencia ejecutable |

**¿Se puede reproducir una versión equivalente con supuestos razonables?** **Parcialmente, con un riesgo de divergencia medio.** Los supuestos que habría que tomar:
1. **GA:** población 50, 100 generaciones, mutación 0.01, cruce 0.8 (valores típicos).
2. **Normalización:** media/std ajustada solo con train.
3. **Selección de arquitectura:** asumir que se usó validation set.
4. **Learning rate:** 0.01 (valor típico para Quasi-Newton).
5. **Semilla:** fijar una.

**Riesgo de divergencia del desempeño reportado:** **medio.** El estudio tiene partición cronológica correcta y early stopping con validation, lo cual es bueno. Pero la ambigüedad en normalización y GA (posible fuga) y la falta de learning rate introducen incertidumbre. El desempeño cualitativo (DPI ~88%) probablemente se mantiene, pero la magnitud exacta puede variar.

**Riesgo de divergencia de la arquitectura:** **bajo.** La arquitectura está reportada (20-14-1, sigmoide, Quasi-Newton).

**Riesgo de divergencia del pipeline:** **medio.** Faltan los parámetros del GA, el learning rate y el timing de normalización/selección.

**Advertencia sobre comparabilidad de resultados.** Dadas las carencias identificadas —target r² (extremadamente ruidoso), normalización y GA con timing ambiguo, selección de arquitectura sin conjunto declarado, learning rate no reportado, parámetros del GA no reportados, sin semilla y ausencia de código—, **no se puede asegurar que los resultados de la reimplementación sean comparables a los reportados en el artículo**. La reimplementación sería una **aproximación**. El benchmark debe tratar la versión corregida (con normalización solo con train, GA solo con train) como una **variante** del modelo original y reportar la brecha entre ambos desempeños.

# 7. Verificación de criterios de inclusión/exclusión

- **C1. Relevancia + DL:** **Sí.** Pronóstico de volatilidad (proxy) del crudo con MLP (red neuronal) como componente central. **Nota:** MLP con 1 capa oculta es una red "shallow", no "deep" en el sentido estricto. Pero es una red neuronal y el benchmark incluye MLP.
- **C2. Reporte suficiente:** **Parcial.** Reporta arquitectura, activación, algoritmo, early stopping y ablation. Pero no reporta learning rate, parámetros del GA, timing de normalización/GA, conjunto de selección de arquitectura, semilla ni código. **La falta de baselines externos limita la verificación de la afirmación central.**
- **C3. Calidad editorial:** **Sí.** Q2 en JCR (SPE Reservoir Evaluation & Engineering, Engineering/Petroleum).
- **E1–E4:** No aplica.
- **E5. Protocolo no reconstruible:** **No aplica.** El protocolo es reconstruible con supuestos (partición cronológica documentada, arquitectura descrita, early stopping documentado).
- **E6. Solo precio, sin volatilidad:** **No aplica.** Predice volatilidad (proxy).
- **E7. GARCH como contribución principal:** **No aplica.** GARCH no se usa.

**Resultado:** **Incluido** (con reservas).

# 8. Notas para la evaluación de sesgo y reproducibilidad

**Indicadores de sesgo:**
- **Target:** r² (un solo retorno mensual al cuadrado). **INADECUADO** por tres razones, en orden de gravedad: (i) **ruido extremo** (SE ≈ 1.41σ², mayor que el propio parámetro; relación señal/ruido < 1); (ii) **inconsistencia con la definición del paper** (intro dice "standard deviation", fórmula usa r²; r² estima σ², no σ; para σ habría que usar |r|, que es sesgado a la baja por √(2/π) ≈ 0.798); (iii) **sesgo pequeño hacia arriba** para σ² por μ² (~1% para WTI mensual). No es RV ni desviación estándar móvil. El problema principal es el **ruido**, no el sesgo. → **alto**.
- **Partición:** cronológica 70/15/15 con validation set separado. → **bajo**.
- **Normalización:** media/std, **timing ambiguo** (posible antes del split). → **moderado-alto (fuga potencial)**.
- **GA:** selección de features, **timing ambiguo** (posible sobre dataset completo). → **moderado-alto (fuga potencial)**.
- **Selección de arquitectura:** 1 capa, Quasi-Newton, sigmoide. **Conjunto no declarado** (¿validation o test?). → **moderado**.
- **Predicción:** single-step (1 mes), sin rolling. → **bajo**.
- **Reporte selectivo:** reporta variantes internas, pero **sin baselines externos** y **sin varianza entre ejecuciones**. → **alto**.
- **Inferencia:** solo Jarque-Bera de residuos. **Sin Diebold-Mariano, MCS ni Clark-West.** → **alto**.
- **Comparación:** **sin baselines externos**. La comparación es solo entre variantes del propio modelo. → **alto**.
- **Ablation:** cubre profundidad, algoritmo, activación, selección de features. **No cubre target, frecuencia de datos, tamaño del conjunto de features ni baselines externos.** → **moderado**.
- **Reproducibilidad:** sin learning rate, sin parámetros del GA, sin timing de normalización/GA, sin semilla, sin código. → **parcial**.

**Indicadores de reproducibilidad operativa:**
- **Acceso a variables:** **sí.** WTI y variables macro son públicos (EIA, FRED, World Bank). → **Sí**.
- **Reconstructibilidad del pipeline:** **parcial.** Arquitectura y early stopping reportados, pero faltan learning rate, parámetros del GA, timing de normalización/GA y semilla. → **Parcial**.
- **Viabilidad computacional:** **sí.** MLP con 20-14-1, 220 observaciones, GA con 58 variables. Viable en CPU. → **Sí**.

**Clasificación: R-B (no reproducible por el equipo).** R1=Sí, R2=Tal vez, R3=Sí → no cumple R2=Sí. El estudio se conserva en la revisión como referencia de la familia `MLP (Multilayer Perceptron)` con selección de features por GA. Si el equipo decide reimplementarlo con **normalización solo con train** y **GA solo con train** (corrigiendo las posibles fugas) y los supuestos declarados, entra al benchmark como **variante corregida**.

**Decisión provisional:**
- **Entra al benchmark:** sí, con reservas (si se reimplementa con target corregido y normalización/GA solo con train).
- **Tipo de entrada:** **con corrección**.
- **Correcciones necesarias:**
    - **(PRIORITARIA) Corregir el target:** reemplazar r² por **volatilidad realizada (RV)** de alta frecuencia (5 min) o, en su defecto, por una **desviación estándar móvil** de ventana declarada (e.g., 5, 22 días). r² es un **target inadecuado** por tres razones: (i) **ruido extremo** (SE ≈ 1.41σ², relación señal/ruido < 1); (ii) **inconsistencia con la definición del paper** (intro dice "standard deviation", fórmula usa r²; r² estima σ², no σ); (iii) **sesgo pequeño** hacia arriba por μ² (~1%). El problema principal es el ruido y la inconsistencia, no el sesgo. Sin esta corrección, el modelo no es comparable con el resto del benchmark.
    - **(PRIORITARIA) Declarar y garantizar el timing de la normalización:** ajustar media/std **solo con el train set**. Sin esta corrección, hay fuga potencial.
    - **(PRIORITARIA) Declarar y garantizar el timing del GA:** evaluar el GA **solo con el train set** (o con train+validation, pero nunca con test). Sin esta corrección, hay fuga potencial.
    - **(PRIORITARIA) Incluir baselines externos:** comparar contra **GARCH(1,1), HAR-RV, LSTM** con los mismos 20 inputs o, al menos, contra GARCH univariado. Sin esta comparación, la afirmación "el modelo captura las dinámicas" no está identificada.
    - **(PRIORITARIA) Declarar el conjunto de selección de arquitectura:** ¿se usó validation o test para elegir 1 capa, Quasi-Newton, sigmoide? Si fue test, hay fuga.
    - **Reportar el learning rate** del Quasi-Newton.
    - **Reportar los parámetros del GA:** tamaño de población, número de generaciones, tasa de mutación, tasa de cruce.
    - **Reportar el número de neuronas en la capa oculta** y justificar 14 con un análisis de sensibilidad (8, 14, 20, 30).
    - **Reportar el tamaño del conjunto de features** y justificar 20 con un análisis de sensibilidad (10, 20, 30, 40).
    - **Reportar la semilla.**
    - **Añadir QLIKE** como métrica de evaluación, además de Erms, Ea, Er y DPI.
    - **Añadir pruebas de significancia entre modelos** (Diebold-Mariano, Model Confidence Set).
    - **Liberar el código** o, en su defecto, reportar todos los valores finales de hiperparámetros y la semilla.
- **Brecha principal:** El estudio aporta la familia `MLP (Multilayer Perceptron)` con selección de features por GA, una partición cronológica correcta con validation set, y un ablation de arquitectura (profundidad, algoritmo, activación, selección de features). **La brecha es de identificación y de validez:** el target es r² (un solo retorno mensual al cuadrado), un **target inadecuado** por ruido extremo (SE > parámetro), por inconsistencia con la definición del paper (σ vs r²) y, en menor medida, por sesgo (μ²); y **no hay comparación contra baselines externos** (GARCH, LSTM), por lo que la afirmación "el modelo captura las dinámicas de la volatilidad" no está identificada. Además, la normalización y el GA tienen timing ambiguo (posible fuga). El benchmark debe reimplementar el modelo con **target RV o desviación estándar móvil**, **normalización y GA solo con train**, e **incluir baselines externos** (GARCH, HAR-RV, LSTM) con los mismos inputs. Sin esas correcciones, la comparación debe tratarse como **no concluyente**. Las demás carencias (learning rate, parámetros del GA, selección de arquitectura, semilla, varianza entre ejecuciones) son deficiencias de reporte que afectan la reproducibilidad, pero no la identificación del efecto; se corrigen en la reimplementación y se documentan en la Sección 8.
