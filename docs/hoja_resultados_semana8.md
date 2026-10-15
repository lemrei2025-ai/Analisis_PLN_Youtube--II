# Semana 8, segunda parte: del dataset al modelo (estadísticas de YouTube)

**Curso Procesamiento de Lenguaje Natural · Hoja de resultados**

Para diligenciar, descargar la versión en Word: [Hoja_Resultados_Semana8.docx](Hoja_Resultados_Semana8.docx).


## 1. Propósito, entrega y forma de trabajo

Esta es la segunda parte del proyecto Estadísticas de YouTube. En la primera parte (semana 7) el equipo preparó y analizó los datos; en esta, entrena y compara modelos de Procesamiento de Lenguaje Natural que predicen el desempeño de un video a partir de sus comentarios. Con base en la información del dataset, el equipo responde con sus propios resultados siete preguntas: qué análisis estadísticos se pueden hacer, cuál fue el modelo seleccionado, cuál fue su desempeño y bajo qué métricas, cuántos modelos se probaron, qué herramientas se usaron para determinar el modelo, si es suficiente un solo modelo y qué variables se seleccionaron para predecir.

El trabajo se hace en el notebook 02_modelos.ipynb del repositorio del taller, que ya contiene el código. El equipo no programa: en cada sección toma decisiones en formularios (casillas, listas y deslizadores), ejecuta la celda e interpreta los resultados. Este documento es la **hoja de resultados**: en cada sección se registran las cifras obtenidas y su interpretación.

| Aspecto                 | Detalle                                                                                                                                                                 |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Curso                   | Procesamiento de Lenguaje Natural                                                                                                                                       |
| Entrega                 | Al finalizar la semana 8                                                                                                                                                |
| Duración de la práctica | 2 horas                                                                                                                                                                 |
| Notebook                | notebooks/02_modelos.ipynb (se abre en Colab con el enlace del README)                                                                                                  |
| Datos                   | Dataset YouTube Statistics de Kaggle (advaypatil/youtube-statistics) procesado con el pipeline de la primera parte; opcionalmente, la extracción de la YouTube Data API |
| Requisitos              | Los mismos de la primera parte: secretos de Kaggle en Colab y la carpeta del equipo en Google Drive                                                                     |

**Qué se predice.** Por defecto, la variable objetivo que el equipo eligió en la primera parte (engagement, log_views, log_likes o perf_class). Las variables predictoras salen del contexto del video y, sobre todo, del texto de sus comentarios.

**Distribución del tiempo**

| Sección del notebook                                                        | Tiempo | Sección de esta hoja |
|-----------------------------------------------------------------------------|--------|----------------------|
| Celdas 1 y 2 y sección 3: preparar el entorno y cargar la tabla de features | 15 min | 1                    |
| 4\. Análisis estadísticos                                                   | 20 min | 2                    |
| 5\. Selección de variables                                                  | 20 min | 3                    |
| 6\. Entrenar y comparar modelos                                             | 25 min | 4                    |
| 7\. Desempeño y comparación estadística                                     | 15 min | 5                    |
| 8\. Modelo seleccionado                                                     | 15 min | 6                    |
| 9\. ¿Es suficiente un solo modelo? y 10. Resumen                            | 10 min | 7 y 8                |

### Registro del equipo

| Campo                                                            | Respuesta del equipo |
|------------------------------------------------------------------|----------------------|
| Integrantes                                                      | (completar) |
| Fuente de datos usada (kaggle o api) y número de videos          | (completar) |
| Variable objetivo y tipo de problema (regresión o clasificación) | (completar) |

## 2. ¿Qué análisis estadísticos se pueden hacer?

### Conceptos clave

- **Análisis de la primera parte:** el pipeline ya calculó nueve análisis estadísticos (archivo stats_report), con su pregunta, prueba y conclusión; el notebook los reúne en una tabla.

- **Distribución de la variable objetivo:** forma y asimetría en regresión, o balance de clases en clasificación. Un objetivo muy desbalanceado hace engañosas métricas como la *accuracy*.

- **Correlación de Spearman con corrección de Benjamini-Hochberg:** mide la relación monótona entre cada feature de los comentarios y el objetivo. Al probar muchas variables a la vez, algunas parecen significativas por azar; la corrección ajusta los valores p para controlar ese riesgo.

- **Comparación estadística de modelos** (sección 7 del notebook): la prueba de Wilcoxon y el intervalo de confianza *bootstrap* indican si la diferencia entre modelos es sólida o podría deberse al azar.

### En el notebook

1.  Ejecutar las celdas 1 y 2 y la sección 3, eligiendo la fuente de datos (kaggle o api) y la variable objetivo.

2.  Ejecutar la sección 4 y revisar la tabla de análisis de la primera parte, la distribución del objetivo y las correlaciones.

### Resultados del equipo

| Análisis                                                                       | Resultado obtenido (cifras) | Interpretación |
|--------------------------------------------------------------------------------|-----------------------------|----------------|
| Dos análisis de la primera parte que el equipo considera más relevantes        | (completar) | (completar) |
| Distribución del objetivo (media, mediana, asimetría o proporción por clase)   | (completar) | (completar) |
| Features de comentarios con correlación significativa después de la corrección | (completar) | (completar) |
| Feature con la correlación más fuerte (rho y signo)                            | (completar) | (completar) |

### Interpretación del equipo

| Pregunta                                                                                                           | Respuesta del equipo |
|--------------------------------------------------------------------------------------------------------------------|----------------------|
| ¿Qué dicen estos análisis sobre la posibilidad de predecir el desempeño a partir de los comentarios?               | (completar) |
| ¿Qué otro análisis estadístico sobre el texto propondría el equipo (por ejemplo, sobre emojis, idioma o longitud)? | (completar) |

**Para profundizar:** ¿qué problema resuelve la corrección por comparaciones múltiples y en qué se diferencia el método de Benjamini-Hochberg del de Bonferroni?

## 3. ¿Qué variables se seleccionaron para predecir?

### Conceptos clave

- **Bloques de variables:** *contexto* (keyword, antigüedad, día y hora de publicación, características del título); *features de comentarios* (sentimiento medio y ponderado, polarización, porcentaje positivo y negativo, longitud, diversidad léxica, preguntas, exclamaciones y emojis); y *texto TF-IDF* (los n-gramas del texto de los comentarios).

- **TF-IDF:** representa cada video por las palabras de sus comentarios, ponderadas por su frecuencia en el video y su rareza en el conjunto. Los parámetros clave son los n-gramas (palabras sueltas o pares), el tamaño máximo del vocabulario y la frecuencia mínima de un término.

- **Fuga de información:** las vistas, los *likes* y el número de comentarios del video no se usan como predictores, porque con ellos se calcula la variable objetivo.

- **Aporte del texto:** comparar el modelo con y sin texto muestra si los comentarios realmente ayudan a predecir.

### En el notebook

1.  En la sección 5, marcar los bloques de variables y ajustar los parámetros del TF-IDF.

2.  Revisar el tamaño del vocabulario y los términos más asociados con el objetivo.

3.  Opcional: repetir las secciones 5 y 6 con otra configuración (por ejemplo, solo texto o sin texto) y comparar.

### Resultados del equipo

| Campo                                                                                            | Respuesta del equipo |
|--------------------------------------------------------------------------------------------------|----------------------|
| Bloques de variables seleccionados                                                               | (completar) |
| Parámetros del TF-IDF (n-gramas, máximo de términos, frecuencia mínima) y tamaño del vocabulario | (completar) |
| Términos más asociados con valores altos y bajos del objetivo                                    | (completar) |
| Si se hizo la prueba opcional: configuración alternativa y efecto en el error                    | (completar) |

### Interpretación del equipo

| Pregunta                                                                                                              | Respuesta del equipo |
|-----------------------------------------------------------------------------------------------------------------------|----------------------|
| ¿Los términos más asociados tienen sentido lingüístico o reflejan ruido (nombres propios, temas de una sola keyword)? | (completar) |
| ¿Qué otra representación del texto probaría el equipo (*embeddings*, modelos preentrenados, lematización) y por qué?  | (completar) |

**Para profundizar:** ¿qué diferencias hay entre la bolsa de palabras, TF-IDF y los *embeddings* de oraciones, y cuándo conviene cada uno?

## 4. ¿Cuántos modelos se probaron y con qué herramientas?

### Conceptos clave

- **Modelos disponibles:** base (promedio o clase más frecuente), lineal regularizado (Ridge o regresión logística), SVM lineal, bosque aleatorio y *gradient boosting*. Los modelos lineales trabajan directamente con el TF-IDF; para los de árboles, el texto se reduce primero con análisis semántico latente (TruncatedSVD).

- **Partición por video:** el 20 % de los videos se reserva para la prueba final; sobre el 80 % restante se hace una validación cruzada de 5 particiones para comparar modelos. Nunca se mezclan comentarios de un mismo video entre entrenamiento y prueba.

- **Herramientas:** Google Colab y Google Drive (entorno y almacenamiento), Python, pandas, SciPy, scikit-learn (TfidfVectorizer, ColumnTransformer, modelos, cross_validate y métricas), Matplotlib y el pipeline ytnlp del taller.

### En el notebook

1.  En la sección 6, marcar los modelos que se van a comparar (se recomienda incluir la base y al menos tres modelos más).

2.  Ejecutar la celda; la tabla muestra la métrica en validación cruzada, su desviación, las métricas en prueba y el tiempo de cada modelo.

### Resultados del equipo

| Campo                                                                                           | Respuesta del equipo |
|-------------------------------------------------------------------------------------------------|----------------------|
| Número de modelos probados y cuáles                                                             | (completar) |
| Configuraciones adicionales probadas (otros bloques de variables u otros parámetros del TF-IDF) | (completar) |
| Herramientas usadas para determinar el modelo                                                   | (completar) |
| Modelo más rápido y más lento (segundos)                                                        | (completar) |

### Interpretación del equipo

| Pregunta                                                                                                                       | Respuesta del equipo |
|--------------------------------------------------------------------------------------------------------------------------------|----------------------|
| ¿Por qué los modelos lineales suelen funcionar bien con representaciones TF-IDF?                                               | (completar) |
| ¿Qué modelo adicional probaría el equipo (por ejemplo, un transformador preentrenado ajustado al dataset) y qué costo tendría? | (completar) |

**Para profundizar:** ¿qué es el análisis semántico latente (LSA) y por qué reduce la dimensionalidad del texto?

## 5. ¿Cuál fue el desempeño y bajo qué métricas?

### Conceptos clave

- **Métricas de regresión** (engagement, log_views, log_likes): MAE (error absoluto medio en escala logarítmica; menor es mejor), RMSE (penaliza los errores grandes), R² (proporción de la variabilidad explicada; 0 equivale a predecir el promedio) y Spearman (si el modelo ordena bien los videos de menor a mayor desempeño).

- **Métricas de clasificación** (perf_class): *accuracy* (proporción de aciertos) y F1 macro (promedio del F1 de cada clase, que no se deja engañar por clases desbalanceadas).

- **Validación frente a prueba:** los modelos se comparan con la validación cruzada; la prueba confirma el desempeño esperado con videos nuevos.

- **Solidez de la diferencia:** la prueba de Wilcoxon compara dos modelos partición por partición, y el intervalo de confianza *bootstrap* del 95 % indica cuánto podría variar la métrica con otros videos de prueba.

### En el notebook

Copiar de la tabla de la sección 6 las métricas de cada modelo y ejecutar la sección 7 para obtener la comparación estadística y el intervalo de confianza.

### Resultados del equipo

| Modelo | Métrica en validación cruzada (MAE o F1 macro) | MAE o *accuracy* en prueba | RMSE o F1 macro en prueba | R² en prueba (regresión) |
|--------|------------------------------------------------|----------------------------|---------------------------|--------------------------|
| Base   | (completar) | (completar) | (completar) | (completar) |
|        |                                                |                            |                           |                          |
|        |                                                |                            |                           |                          |
|        |                                                |                            |                           |                          |
|        |                                                |                            |                           |                          |

| Campo                                                                 | Respuesta del equipo |
|-----------------------------------------------------------------------|----------------------|
| Prueba de Wilcoxon: modelos comparados, diferencia promedio y valor p | (completar) |
| Intervalo de confianza *bootstrap* del 95 % del mejor modelo          | (completar) |

### Interpretación del equipo

| Pregunta                                                                                                                         | Respuesta del equipo |
|----------------------------------------------------------------------------------------------------------------------------------|----------------------|
| ¿Qué modelos superaron a la base y la diferencia es estadísticamente sólida?                                                     | (completar) |
| ¿El desempeño es suficiente para usar el modelo en una decisión real (por ejemplo, priorizar qué videos promocionar)? Justificar | (completar) |

**Para profundizar:** ¿qué métrica conviene reportar cuando las clases están desbalanceadas y por qué la *accuracy* puede ser engañosa?

## 6. ¿Cuál fue el modelo seleccionado?

### Conceptos clave

- **Criterios de selección:** mejor métrica en validación cruzada, estabilidad entre particiones, diferencia estadísticamente sólida frente a los demás, tiempo de entrenamiento y facilidad de explicar las predicciones. Entre modelos con desempeño parecido se prefiere el más simple.

- **Interpretación del modelo:** en los modelos lineales, los coeficientes del TF-IDF muestran qué palabras empujan la predicción hacia arriba o hacia abajo; la importancia por permutación mide cuánto empeora el modelo al desordenar cada variable (incluido el texto completo).

- **Análisis de errores:** la matriz de confusión (clasificación), la gráfica de predicción frente a valor real (regresión) y el error por keyword muestran dónde falla el modelo.

### En el notebook

1.  En la sección 8, elegir en la lista el modelo que el equipo considera mejor.

2.  Ejecutar la celda y revisar los términos con más peso, la importancia de las variables, la matriz de confusión o la gráfica de predicción y el error por keyword.

### Resultados del equipo

| Campo                                      | Respuesta del equipo |
|--------------------------------------------|----------------------|
| Modelo seleccionado                        | (completar) |
| Métricas en prueba del modelo seleccionado | (completar) |
| Criterios usados para seleccionarlo        | (completar) |
| Variables o términos más importantes       | (completar) |
| Keywords con mejor y peor desempeño        | (completar) |

### Interpretación del equipo

| Pregunta                                                                                                            | Respuesta del equipo |
|---------------------------------------------------------------------------------------------------------------------|----------------------|
| ¿Las palabras con más peso tienen sentido para explicar el desempeño de un video? ¿Hay señales de sesgo o de ruido? | (completar) |
| ¿Cómo explicaría el equipo el modelo seleccionado a un creador de contenido, en un párrafo y sin términos técnicos? | (completar) |

**Para profundizar:** ¿qué técnicas de explicabilidad existen para modelos de PLN (coeficientes, importancia por permutación, LIME, valores SHAP) y qué aporta cada una?

## 7. ¿Es suficiente un solo modelo?

### Conceptos clave

- **Ensamble:** combina las predicciones de varios modelos (promedio en regresión, voto por mayoría en clasificación). Si mejora al mejor modelo individual, los modelos se complementan.

- **Aporte del texto:** el notebook entrena el mejor modelo sin el bloque TF-IDF; la diferencia muestra cuánto aporta el PLN frente a las variables de contexto y las features numéricas.

- **Error por segmentos:** si el desempeño cambia mucho entre keywords o idiomas, pueden necesitarse modelos especializados.

- **Un modelo por pregunta:** predecir el desempeño de un video, clasificar el sentimiento de cada comentario o detectar comentarios tóxicos son tareas distintas que suelen requerir modelos distintos, a veces encadenados (por ejemplo, un clasificador de sentimiento cuyas salidas alimentan el modelo de desempeño).

### En el notebook

Ejecutar la sección 9, que compara el mejor modelo individual con el ensamble y con el mismo modelo sin texto, y la sección 10, que genera el resumen.

### Resultados del equipo

| Campo                                      | Respuesta del equipo |
|--------------------------------------------|----------------------|
| Métricas del mejor modelo individual       | (completar) |
| Métricas del ensamble                      | (completar) |
| Métricas del mejor modelo sin texto TF-IDF | (completar) |

### Interpretación del equipo

| Pregunta                                                                       | Respuesta del equipo |
|--------------------------------------------------------------------------------|----------------------|
| ¿Es suficiente un solo modelo? Tomar posición y sustentarla con los resultados | (completar) |
| ¿Aportan los comentarios a la predicción? ¿Cuánto?                             | (completar) |
| ¿Qué modelo complementario propondría el equipo y para qué tarea?              | (completar) |

**Para profundizar:** ¿qué es la deriva de datos (*data drift*) en modelos de PLN y por qué el lenguaje de los comentarios de YouTube cambia con el tiempo?

## 8. Síntesis para la entrega de la semana 8

Con apoyo del resumen que genera la sección 10 del notebook, el equipo responde en pocas líneas cada pregunta, basado en la información del dataset.

| Pregunta                                               | Respuesta del equipo |
|--------------------------------------------------------|----------------------|
| ¿Qué análisis estadísticos podemos hacer?              | (completar) |
| ¿Cuál fue el modelo seleccionado?                      | (completar) |
| ¿Cuál fue su desempeño y bajo qué métricas?            | (completar) |
| ¿Cuántos modelos se probaron?                          | (completar) |
| ¿Qué herramientas se usaron para determinar el modelo? | (completar) |
| ¿Es suficiente un solo modelo?                         | (completar) |
| ¿Qué variables se seleccionaron para predecir?         | (completar) |

**Qué se entrega al finalizar la semana 8:** esta hoja de resultados diligenciada y la copia del notebook 02_modelos ejecutada con datos reales, compartida con el docente como *Lector* o *Comentador*.

## 9. Fuentes de consulta

| Fuente                                                                                                                                                                | Acceso                                                                                                                                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Patil, A. *YouTube Statistics* (dataset). Kaggle                                                                                                                      | https://www.kaggle.com/datasets/advaypatil/youtube-statistics                                                                                                                                                           |
| Jurafsky, D. y Martin, J. H. *Speech and Language Processing*, 3.ª ed. (borrador en línea)                                                                            | https://web.stanford.edu/~jurafsky/slp3/                                                                                                                                                                                |
| Manning, C., Raghavan, P. y Schütze, H. (2008). *Introduction to Information Retrieval* (capítulos sobre TF-IDF y clasificación de texto). Cambridge University Press | https://nlp.stanford.edu/IR-book/                                                                                                                                                                                       |
| scikit-learn: trabajo con datos de texto, validación cruzada y métricas                                                                                               | https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction · https://scikit-learn.org/stable/modules/cross_validation.html · https://scikit-learn.org/stable/modules/model_evaluation.html |
| SciPy: pruebas estadísticas                                                                                                                                           | https://docs.scipy.org/doc/scipy/reference/stats.html                                                                                                                                                                   |
| Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7                                          | https://jmlr.org/papers/v7/demsar06a.html                                                                                                                                                                               |
| Molnar, C. *Interpretable Machine Learning*                                                                                                                           | https://christophm.github.io/interpretable-ml-book/                                                                                                                                                                     |