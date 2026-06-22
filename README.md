<<<<<<< HEAD
# 🌍 Predicción del Gasto Turístico Internacional mediante Machine Learning

## 📌 Descripción del Proyecto

Este proyecto ha sido desarrollado por un equipo de **4 analistas de datos** con el objetivo de analizar patrones de comportamiento de viajeros internacionales y construir un modelo predictivo capaz de estimar el **gasto total de un viaje** a partir de diferentes características relacionadas con el perfil del viajero, el destino y las condiciones del viaje.

A través de técnicas de análisis de datos y Machine Learning, se ha realizado un proceso completo que incluye:

- Limpieza y transformación de datos.
- Análisis exploratorio (EDA).
- Ingeniería de características.
- Construcción y evaluación de modelos predictivos.
- Optimización mediante validación cruzada.
- Extracción de insights de negocio.

---

## 🎯 Objetivos

### Objetivo General

Desarrollar un modelo de regresión capaz de predecir el gasto total de un viaje turístico utilizando información disponible antes de que el viaje se realice.

### Objetivos Específicos

✅ Preparar y transformar los datos para su análisis.

✅ Analizar patrones de comportamiento turístico.

✅ Identificar las variables con mayor impacto sobre el gasto.

✅ Construir modelos predictivos robustos.

✅ Evaluar el rendimiento mediante métricas de regresión.

✅ Obtener conclusiones que ayuden a la toma de decisiones en el sector turístico.

---

## 👥 Equipo de Trabajo

Proyecto realizado por un equipo de **4 integrantes**, participando de forma colaborativa en las fases de:

- Limpieza de datos
- Análisis exploratorio
- Visualización
- Modelado predictivo
- Evaluación de resultados
- Documentación del proyecto

---

# 📂 Dataset

El proyecto utiliza dos conjuntos de datos principales.

## 1️⃣ Dataset Original

**Archivo:** `global_tourism_travel_trends.csv`

Contiene información de aproximadamente **10.000 viajes internacionales**.

### Variables incluidas

#### 📅 Información temporal

- Año
- Mes
- Estación del año

#### 🌎 Información geográfica

- País de origen
- País de destino

#### 👤 Perfil del viajero

- Tipo de viajero
- Número de viajeros
- Motivo del viaje

#### 💰 Información económica

- Presupuesto por persona
- Gasto total del viaje
- Gasto diario en restaurantes

#### ✈️ Información logística

- Transporte
- Alojamiento
- Tipo de visado
- Método de reserva
- Antelación de reserva

#### ⭐ Experiencia del viaje

- Nivel de satisfacción
- Seguridad percibida
- Calidad del WiFi
- Barrera idiomática
- Recomendación del destino
- Uso de redes sociales

#### 🌱 Sostenibilidad

- Huella de carbono
- Decisiones ecológicas
- Cumplimiento de medidas sanitarias

---

## 2️⃣ Dataset Procesado

**Archivo:** `dataset_viajes_procesado.csv`

Dataset preparado para Machine Learning tras realizar:

- Limpieza de datos
- Tratamiento de valores nulos
- Codificación de variables categóricas
- Eliminación de variables irrelevantes
- Preparación de variables predictoras

### Características finales

| Métrica | Valor |
|----------|---------|
| Registros | 10.000 |
| Variables | 23 |
| Tipo de problema | Regresión |

---

# 🔧 Preparación de Datos

## Limpieza de Datos

Durante esta fase se realizaron:

- Eliminación de registros duplicados.
- Corrección de formatos.
- Estandarización de variables categóricas.
- Gestión de valores faltantes.

### Variables con valores nulos tratados

- `language_barrier`
- `social_media_shared`
- `health_safety_compliance`

---

## Tratamiento de Outliers

Se aplicó el método **IQR (Interquartile Range)** para detectar y analizar valores atípicos presentes en las variables numéricas.

---

# 📊 Análisis Exploratorio (EDA)

Durante la fase exploratoria se realizaron diferentes análisis para comprender mejor los datos.

## Análisis realizados

- Matriz de correlaciones.
- Distribución de variables numéricas.
- Análisis de asimetría y curtosis.
- Destinos más visitados.
- Métodos de reserva por tipo de viajero.
- Relación entre presupuesto y gasto final.

---

# 🎯 Variable Objetivo

La variable utilizada para entrenar el modelo fue:

```python
total_trip_spend_usd
```

Representa el gasto total realizado por cada viajero durante su estancia.

---

# 🧠 Selección de Características

Las variables con mayor influencia sobre el gasto total fueron:

- Budget per person
- Trip duration
- Daily restaurant spending
- Traveler type
- Travel purpose
- Destination country

---

# 🤖 Modelado Predictivo

El flujo de Machine Learning seguido fue:

```text
Datos Limpios
      ↓
Train / Test Split
      ↓
Entrenamiento
      ↓
Evaluación
      ↓
Optimización
      ↓
Modelo Final
```

---

# 📈 Métricas de Evaluación

Para evaluar el rendimiento de los modelos se utilizaron:

| Métrica | Descripción |
|----------|-------------|
| MAE | Error Absoluto Medio |
| MSE | Error Cuadrático Medio |
| RMSE | Raíz del Error Cuadrático Medio |
| R² | Capacidad explicativa del modelo |

---

# ⚙️ Optimización

Se utilizaron técnicas avanzadas para mejorar el rendimiento:

### 🔄 K-Fold Cross Validation

Permite validar la estabilidad del modelo utilizando múltiples particiones de los datos.

### 🚀 Optuna

Framework utilizado para la optimización automática de hiperparámetros.

---

# 💾 Exportación del Modelo

Una vez finalizado el entrenamiento, el modelo fue almacenado para su reutilización en futuros análisis y entornos productivos.

---

# 📊 Principales Insights

Tras el análisis realizado se concluye que:

- El presupuesto inicial es uno de los factores más influyentes.
- La duración del viaje impacta directamente sobre el gasto final.
- El gasto diario en restauración presenta una fuerte relación con el gasto total.
- Existen diferencias significativas según el tipo de viajero.
- El destino elegido condiciona notablemente el comportamiento de gasto.

---

# 🛠️ Tecnologías Utilizadas

<p align="left">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">

<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">

<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">

<img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge">

<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">

<img src="https://img.shields.io/badge/Optuna-5C4EE5?style=for-the-badge">

<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">

</p>

---

# 📁 Estructura del Proyecto

```text
📦 Tourism-Spending-Prediction
│
├── git
│── .venv
│── venv
│── DA_Project_Regression_Grupo_4_final.ipynb
├── dataset_viajes_procesado
│── global_tourism_travel_trends
│── modelo_metadata
│── modelo_turismo
└── README
├── requirements
```

---

# 🚀 Resultados

El proyecto demuestra que es posible predecir el gasto turístico con un nivel de precisión adecuado utilizando variables disponibles antes del viaje.

Los resultados obtenidos permiten identificar patrones de comportamiento y ofrecen información valiosa para:

- Agencias de viaje.
- Empresas turísticas.
- Equipos de marketing.
- Plataformas de reservas.
- Organismos de promoción turística.

---

# 📚 Conclusiones

Este proyecto integra todas las fases fundamentales de un flujo de trabajo de Data Analytics y Machine Learning: desde la preparación de datos hasta la construcción y optimización de modelos predictivos.

Más allá de las métricas obtenidas, el principal valor reside en transformar datos turísticos en información útil para apoyar la toma de decisiones y comprender mejor los factores que influyen en el comportamiento de los viajeros.
=======
# Tourism_spending_forecasting
# Tourism Spending Forecasting  End-to-end machine learning project focused on predicting tourist expenditure patterns using regression models and deploying interactive predictions through Streamlit.
>>>>>>> c498d3dcdcb6645b3c32022e077fc96cc4164b31
