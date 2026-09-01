# Predicción del Tipo de Cambio USD/PEN mediante Análisis de Sentimiento en Twitter/X

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-cardiffnlp%2Fxlm--twitter--politics--sentiment-yellow)
![Status](https://img.shields.io/badge/Estado-Tesis%20de%20pregrado-blue)

Código reproducible de la tesis **"Predicción de la direccionalidad del tipo de cambio USD/PEN mediante análisis de sentimiento en Twitter/X durante elecciones en Perú"** (elecciones presidenciales 2021), presentada para optar el título de Ingeniero Informático.

| | |
|---|---|
| **Autores** | Ruben Anderson Rojas Ramos&nbsp;&nbsp;·&nbsp;&nbsp;Renzo Antonio Vilca Loayza |
| **Asesores** | Nelson Enrique Castro Zárate&nbsp;&nbsp;·&nbsp;&nbsp;Roy Marco Yali Samaniego |
| **Lugar y año** | Lima, Perú — 2025 |

## ¿Qué hace este proyecto?

Combina un modelo Transformer preentrenado (`cardiffnlp/xlm-twitter-politics-sentiment`) con un **motor de reglas contextuales** diseñado para el discurso electoral peruano (apodos de candidatos, jerga política, sarcasmo, referencias históricas), arbitrados mediante un índice de densidad política (`political_strength`). El resultado es un **Modelo Híbrido** que clasifica el sentimiento político de cada tweet y construye un **Índice Diario de Sentimiento Electoral (IDSE)**, evaluado luego como predictor de la dirección del tipo de cambio USD/PEN del día siguiente.

```
Tweets (API Twitter/X) ──► Limpieza y homologación ──► Modelo Híbrido (Transformer + Reglas)
                                                                │
                                                                ▼
                                              Índice Diario de Sentimiento Electoral
                                                                │
                                                                ▼
                          Correlación (Pearson/Spearman/Kendall) + Clasificación binaria
                                          vs. Tipo de cambio USD/PEN (BCRP, t+1)
```

## Resultados clave

**Correlación más fuerte** entre sentimiento y tipo de cambio (media móvil 7 días, *lead* 1 día):

| Predictor | r (Pearson) | p-valor |
|---|---|---|
| `ratio_pos_neg_ma_7` | **−0.627** | < 0.001 |

> ⚠️ El notebook que agrega el sentimiento diario y calcula esta correlación no forma parte todavía de este repositorio.

**Clasificación de sentimiento político** (gold standard de 200 tweets etiquetados manualmente):

| Modelo | Exactitud | F1-Score |
|---|---|---|
| **Modelo Híbrido** | **0.555** | **0.553** |
| Random Forest | 0.510 | 0.474 |
| SVM | 0.485 | 0.480 |
| Transformer solo (sin reglas) | 0.390 | 0.288 |

**Código:** [`notebooks/03_comparacion_modelos.ipynb`](notebooks/03_comparacion_modelos.ipynb) — celda "PARTE 1". Carga el gold standard (200 tweets con etiqueta humana) y representa cada tweet combinando TF-IDF (5000 dim) con embeddings FastText en español (300 dim); con esa representación entrena Random Forest y SVM usando solo esos 200 tweets (validación cruzada de 5 particiones), corre el Transformer solo por inferencia directa, y toma la salida ya calculada del Modelo Híbrido. Compara los cuatro contra la etiqueta humana con exactitud, precisión, recall y F1.

**Predicción direccional del tipo de cambio** (validación fuera de muestra, `TimeSeriesSplit`, línea base = 69%):

| Modelo | Exactitud | F1-Score | AUC-ROC |
|---|---|---|---|
| SVM | 0.78 | 0.77 | 0.72 |
| **Modelo Híbrido** | **0.78** | 0.76 | 0.71 |
| Random Forest | 0.69 | 0.56 | 0.53 |
| Transformer solo | 0.69 | 0.56 | 0.55 |

**Código:** [`notebooks/03_comparacion_modelos.ipynb`](notebooks/03_comparacion_modelos.ipynb) — celda "PARTE 2 (CORREGIDA)". Aplica por inferencia (sin reentrenar) los modelos RF y SVM guardados en la Parte 1, y el Transformer solo, sobre los 97,118 tweets del corpus; agrega el sentimiento a un índice diario con media móvil de 7 días y, con `TimeSeriesSplit`, busca el umbral de decisión (sube/baja) usando solo los datos de entrenamiento de cada partición temporal para evitar fuga de información, evaluando después en los días de prueba que esa búsqueda nunca vio.

El sistema de reglas contextuales **casi duplica el F1-Score del Transformer usado solo** (0.288 → 0.553), evidenciando el aporte del conocimiento del discurso político peruano.

## Estructura del repositorio

```
├── notebooks/
│   ├── 01_generacion_gold_standard.ipynb    # Muestreo estratificado y armado del set de validación (200 tweets)
│   ├── 02_modelo_hibrido_clasificacion.ipynb  # Clasificador híbrido (Transformer + reglas) y construcción del IDSE
│   └── 03_comparacion_modelos.ipynb         # Evaluación comparativa: Híbrido vs. RF, SVM y Transformer solo (Tablas 8 y 9)
├── data/
│   ├── corpus/
│   │   └── corpus_tweets_clasificado.xlsx   # 97,118 tweets (abril–mayo 2021) con salida del Modelo Híbrido
│   └── gold_standard/
│       └── gold_standard_200_tweets_etiquetado.xlsx  # 200 tweets etiquetados manualmente (validación)
└── requirements.txt
```

> **Nota sobre los datos:** el corpus fue recolectado vía la API de Twitter/X con Tweepy entre el 1 de abril y el 31 de mayo de 2021, y se comparte aquí con fines exclusivamente académicos y de reproducibilidad de esta investigación.

## Cómo reproducir

```bash
git clone https://github.com/anttox/TESIS-MODELO-HIBRIDO.git
cd TESIS-MODELO-HIBRIDO
pip install -r requirements.txt
jupyter notebook notebooks/
```

- `01_generacion_gold_standard.ipynb`: parte del corpus ya clasificado y selecciona una muestra estratificada (clase × confianza × bloque temporal) para etiquetado humano.
- `02_modelo_hibrido_clasificacion.ipynb`: implementa `PeruvianSentimentClassifier`, la clase que combina el Transformer con el motor de reglas y produce las columnas de sentimiento (`sentimiento_economico`, `confianza_sentimiento`, `metodo_clasificacion`, etc.) sobre el corpus completo.
- `03_comparacion_modelos.ipynb`: entrena y evalúa Random Forest y SVM (TF-IDF + embeddings FastText) sobre el gold standard, y compara los cuatro modelos tanto en clasificación de sentimiento como en predicción direccional del tipo de cambio.

## Metodología

Diseño no experimental, longitudinal de series temporales, con alcance correlacional, siguiendo **CRISP-DM**. Corpus de 97,118 tweets válidos (61 días de observación, 1 abril –31 mayo 2021) cruzado con el tipo de cambio de compra USD/PEN oficial del BCRP. Validación mediante correlación de Pearson/Spearman/Kendall y clasificación binaria con `TimeSeriesSplit` para evitar fuga de información.
