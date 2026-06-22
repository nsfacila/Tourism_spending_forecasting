# Tourism Spending Forecasting

End-to-end Machine Learning project focused on forecasting tourism expenditure through regression modeling, business analytics and Streamlit deployment.

---

## Equipo de Trabajo

Proyecto realizado por un equipo de **4 integrantes**, participando de forma colaborativa en las fases de:

- Limpieza de datos
- Análisis exploratorio
- Visualización
- Modelado predictivo
- Evaluación de resultados
- Documentación del proyecto

---

# Dataset

El proyecto utiliza dos conjuntos de datos principales.

## Dataset Original

**Archivo:** `global_tourism_travel_trends.csv`

Contiene información de aproximadamente **10.000 viajes internacionales**.

---

## Dataset Procesado

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


# Modelado Predictivo

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

# Métricas de Evaluación

Para evaluar el rendimiento de los modelos se utilizaron:

| Métrica | Descripción |
|----------|-------------|
| MAE | Error Absoluto Medio |
| MSE | Error Cuadrático Medio |
| RMSE | Raíz del Error Cuadrático Medio |
| R² | Capacidad explicativa del modelo |


---

# Principales Insights

Tras el análisis realizado se concluye que:

- El presupuesto inicial es uno de los factores más influyentes.
- La duración del viaje impacta directamente sobre el gasto final.
- El gasto diario en restauración presenta una fuerte relación con el gasto total.
- Existen diferencias significativas según el tipo de viajero.
- El destino elegido condiciona notablemente el comportamiento de gasto.

---

# Tecnologías Utilizadas

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

# Estructura del Proyecto

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

# Resultados

El proyecto demuestra que es posible predecir el gasto turístico con un nivel de precisión adecuado utilizando variables disponibles antes del viaje.

Los resultados obtenidos permiten identificar patrones de comportamiento y ofrecen información valiosa para:

- Agencias de viaje.
- Empresas turísticas.
- Equipos de marketing.
- Plataformas de reservas.
- Organismos de promoción turística.

---

# Conclusiones

Este proyecto integra todas las fases fundamentales de un flujo de trabajo de Data Analytics y Machine Learning: desde la preparación de datos hasta la construcción y optimización de modelos predictivos.

Más allá de las métricas obtenidas, el principal valor reside en transformar datos turísticos en información útil para apoyar la toma de decisiones y comprender mejor los factores que influyen en el comportamiento de los viajeros.

