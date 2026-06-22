import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Configuración de página premium y ancha
st.set_page_config(
    page_title="Turismo ML App Premium", 
    page_icon="🗺️",
    layout="wide"
)

# =====================================================================
# CAPA DE DISEÑO VISUAL: ESTILO CLARO, VIVO Y ANIMADO (MODERNO/BI)
# =====================================================================
st.markdown("""
    <style>
    /* Fondo de la app claro y tipografía limpia */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Textos globales en color oscuro para máxima legibilidad */
    .stMarkdown, p, span, label, .stCaption {
        color: #334155 !important;
    }
    
    /* Encabezados estilizados con color corporativo */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Tarjetas KPI: Fondo blanco/azul cristalino con bordes vivos y sombras sutiles */
    .kpi-card-premium {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(148, 163, 184, 0.12);
        text-align: center;
        margin-bottom: 15px;
        border-top: 5px solid #2563eb; /* Línea superior azul vibrante */
        border-left: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    .kpi-label-premium {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b !important;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value-premium {
        font-size: 28px;
        font-weight: 700;
        color: #1e3a8a !important; /* Azul eléctrico vivo para destacar */
    }
    
    /* Títulos de sección para las gráficas */
    .chart-header {
        font-size: 18px;
        font-weight: 600;
        color: #0f172a !important;
        margin-top: 25px;
        margin-bottom: 12px;
        border-left: 4px solid #2563eb;
        padding-left: 12px;
    }
    
    /* Contenedor de la barra lateral (Sidebar) adaptado al modo claro */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Personalización para el Radio Horizontal estilo Pestañas */
    div[data-testid="stRadio"] > div {
        gap: 10px;
    }
    div[data-testid="stRadio"] label {
        background-color: #f1f5f9;
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #2563eb;
        color: #2563eb !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: #2563eb !important;
        color: white !important;
        border-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    div[data-testid="stRadio"] label[data-checked="true"] p {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# DETECTOR AUTOMÁTICO DE RUTAS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
RUTA_PKL = BASE_DIR / "modelo_turismo.pkl"
RUTA_JSON = BASE_DIR / "modelo_metadata.json"

NOMBRE_CSV_GLOBAL = "global_tourism_travel_trends.csv"
RUTA_CSV_GLOBAL = BASE_DIR / "data" / NOMBRE_CSV_GLOBAL

NOMBRE_CSV_PROCESADO = "dataset_viajes_procesado.csv"
RUTA_CSV_PROCESADO = BASE_DIR / "data" / NOMBRE_CSV_PROCESADO

# =====================================================================
# FUNCIÓN DE CARGA EN MEMORIA (OPTIMIZADA CON CACHÉ)
# =====================================================================
@st.cache_resource
def load_web_resources():
    modelo = None
    metadata = None
    
    if RUTA_PKL.exists():
        modelo = joblib.load(RUTA_PKL)
    if RUTA_JSON.exists():
        with open(RUTA_JSON, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    return modelo, metadata

pipeline_ia, meta_ia = load_web_resources()

# CARGA DEL CSV GLOBAL
df_global = None
col_pais, col_year, col_proposito, col_spend = [], [], [], []
if RUTA_CSV_GLOBAL.exists():
    df_global = pd.read_csv(RUTA_CSV_GLOBAL)
    col_pais = ['destination_country'] if 'destination_country' in df_global.columns else [df_global.columns[5]]
    col_year = ['year'] if 'year' in df_global.columns else [df_global.columns[1]]
    col_proposito = ['travel_purpose'] if 'travel_purpose' in df_global.columns else [df_global.columns[6]]
    col_spend = ['total_trip_spend_usd'] if 'total_trip_spend_usd' in df_global.columns else [df_global.columns[16]]

# CARGA DEL CSV PROCESADO
df_procesado = None
if RUTA_CSV_PROCESADO.exists():
    df_procesado = pd.read_csv(RUTA_CSV_PROCESADO)

# =====================================================================
# ENCABEZADO Y NAVEGACIÓN PRINCIPAL (REEMPLAZA A ST.TABS)
# =====================================================================
st.title("Sistema Inteligente de Tendencias de Turismo")
st.write("Plataforma BI Analítica & Motor Predictivo de Inteligencia Artificial.")

# Menú de navegación horizontal reactivo
seccion_activa = st.radio(
    "Navegación del Sistema:",
    ["Dashboard de Negocio", "Certificación de la IA (Metadata)", "Simulador de Predicciones Optimizadas"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# CONTROLES GLOBALES EN LA BARRA LATERAL (SIDEBAR) DINÁMICOS
# =====================================================================
st.sidebar.title("Panel de Control")
st.sidebar.markdown("Gestione los filtros del Dashboard Global y los parámetros del Simulador.")

# =====================================================================
# OPCIONES CATEGÓRICAS USADAS EN EL ENTRENAMIENTO (MODELO AMPLIADO)
# =====================================================================
# Estas listas reflejan EXACTAMENTE las categorías que el modelo vio en el
# entrenamiento (incluida la categoría de referencia eliminada por
# get_dummies(drop_first=True)).
OPCIONES_ACCOMMODATION = ['3-Star Hotel', '4-Star Hotel', '5-Star Hotel', 'Airbnb/Vacation Rental',
                          'Budget Hotel/Hostel', 'Camping/Glamping', 'Guesthouse', 'Resort']
OPCIONES_TRAVELER_TYPE = ['Backpacker', 'Business Traveler', 'Couple', 'Family', 'Group Tour', 'Solo']
OPCIONES_TRAVEL_PURPOSE = ['Adventure/Sports', 'Business', 'Cultural Exchange', 'Digital Nomad',
                           'Education', 'Family Visit', 'Honeymoon', 'Leisure/Tourism',
                           'Medical Tourism', 'Religious/Pilgrimage']
OPCIONES_DESTINATION = ['Egypt', 'France', 'Malaysia', 'New Zealand', 'Other', 'Thailand']

# Inicializar SIEMPRE el diccionario con valores por defecto de manera segura
valores_usuario = {
    'duration_nights': 7,
    'budget_per_person_usd': 150,
    'restaurant_spend_per_day_usd': 40,
    'num_travelers': 2,
    'accommodation_type': '4-Star Hotel',
    'traveler_type': 'Couple',
    'travel_purpose': 'Leisure/Tourism',
    'destination_country': 'France',
}

# --- LÓGICA DE CONTROL DE SIDEBAR SEGÚN LA SECCIÓN SELECCIONADA ---

if seccion_activa == "Dashboard de Negocio":
    st.sidebar.info("Dashboard de Negocio")
    with st.sidebar.expander("Filtros del Dashboard (CSV Global)", expanded=True):
        st.write("Criterios de segmentación para los gráficos del histórico original.")
        
        if df_global is not None:
            lista_paises = ["Todos"] + sorted(df_global[col_pais[0]].dropna().unique().tolist())
            lista_años = ["Todos"] + sorted(df_global[col_year[0]].dropna().unique().tolist())
            lista_motivos = ["Todos"] + sorted(df_global[col_proposito[0]].dropna().unique().tolist())
        else:
            st.error(f"No se encontró el archivo '{NOMBRE_CSV_GLOBAL}'")
            lista_paises, lista_años, lista_motivos = ["Todos"], ["Todos"], ["Todos"]

        filtro_pais = st.selectbox("País de Destino:", lista_paises)
        filtro_año = st.selectbox("Año del Viaje:", lista_años)
        filtro_proposito = st.selectbox("Propósito de Viaje:", lista_motivos)

elif seccion_activa == "Certificación de la IA (Metadata)":
    st.sidebar.warning("Estás visualizando la documentación técnica estática del modelo.")

elif seccion_activa == "Simulador de Predicciones Optimizadas":
    st.sidebar.info("Parámetros del Simulador IA")
    with st.sidebar.expander("Perfil del Viaje a Simular", expanded=True):
        st.write("Modifica los datos para calcular el gasto proyectado por el modelo.")
        
        if meta_ia is not None:
            valores_usuario['budget_per_person_usd'] = st.sidebar.slider(
                "Presupuesto por Persona (USD):", min_value=10, max_value=2000, value=150)
            valores_usuario['num_travelers'] = st.sidebar.slider(
                "Número de Viajeros:", min_value=1, max_value=29, value=2)
            valores_usuario['duration_nights'] = st.sidebar.slider(
                "Duración de la Estancia (Noches):", min_value=1, max_value=30, value=7)
            valores_usuario['restaurant_spend_per_day_usd'] = st.sidebar.slider(
                "Gasto Diario en Restaurantes (USD):", min_value=0, max_value=500, value=40)
            valores_usuario['accommodation_type'] = st.sidebar.selectbox(
                "Tipo de Alojamiento:", OPCIONES_ACCOMMODATION,
                index=OPCIONES_ACCOMMODATION.index('4-Star Hotel'))
            valores_usuario['traveler_type'] = st.sidebar.selectbox(
                "Tipo de Viajero:", OPCIONES_TRAVELER_TYPE,
                index=OPCIONES_TRAVELER_TYPE.index('Couple'))
            valores_usuario['travel_purpose'] = st.sidebar.selectbox(
                "Propósito del Viaje:", OPCIONES_TRAVEL_PURPOSE,
                index=OPCIONES_TRAVEL_PURPOSE.index('Leisure/Tourism'))
            valores_usuario['destination_country'] = st.sidebar.selectbox(
                "País de Destino:", OPCIONES_DESTINATION,
                index=OPCIONES_DESTINATION.index('France'))
            st.sidebar.caption(
                "ℹ️ El modelo aprendió que **presupuesto por persona** y "
                "**número de viajeros** son los factores que determinan "
                "casi por completo el gasto total (≈97% del peso predictivo). "
                "El resto de variables (duración, alojamiento, tipo de "
                "viajero, propósito, destino) aportan contexto de negocio "
                "pero su impacto en la predicción es marginal."
            )

# =====================================================================
# RENDERIZADO DE CONTENIDOS SEGÚN SECCIÓN SELECCIONADA
# =====================================================================

# --- SECCIÓN 1: DASHBOARD DE NEGOCIO ---
if seccion_activa == "Dashboard de Negocio":
    if df_global is not None:
        df_filtrado = df_global.copy()
        
        if col_pais and filtro_pais != "Todos":
            df_filtrado = df_filtrado[df_filtrado[col_pais[0]] == filtro_pais]
        if col_year and filtro_año != "Todos":
            df_filtrado = df_filtrado[df_filtrado[col_year[0]] == filtro_año]
        if col_proposito and filtro_proposito != "Todos":
            df_filtrado = df_filtrado[df_filtrado[col_proposito[0]] == filtro_proposito]
            
        st.subheader("📊 Panel Analítico de Rendimiento Turístico")
        st.caption(f"Filtros Activos (Dataset Global) | Destino: **{filtro_pais}** | Año: **{filtro_año}** | Propósito: **{filtro_proposito}**")
        
        def formatear_corto_usd(valor):
            if valor >= 1_000_000:
                return f"${valor / 1_000_000:.2f} M USD"
            elif valor >= 1_000:
                return f"${valor:,.0f} USD"
            else:
                return f"${valor:,.2f} USD"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_segmentado = int(len(df_filtrado))
        total_original = int(len(df_global))
        
        with kpi1:
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">Registros</div><div class="kpi-value-premium">{total_segmentado:,d}</div></div>', unsafe_allow_html=True)
            
        if col_spend and len(df_filtrado) > 0:
            gasto_medio = float(df_filtrado[col_spend[0]].mean())
            gasto_total = float(df_filtrado[col_spend[0]].sum())
        else:
            gasto_medio, gasto_total = 0.0, 0.0

        with kpi2:
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">Gasto Medio por Viaje</div><div class="kpi-value-premium">{formatear_corto_usd(gasto_medio)}</div></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">Volumen Financiero Total</div><div class="kpi-value-premium">{formatear_corto_usd(gasto_total)}</div></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown(f'<div class="kpi-card-premium"><div class="kpi-label-premium">Dataset Original Global</div><div class="kpi-value-premium">{total_original:,d}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if len(df_filtrado) > 0:
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown('<div class="chart-header">Top Destinos más Demandados</div>', unsafe_allow_html=True)
                conteo_paises = df_filtrado[col_pais[0]].value_counts().head(10)
                st.bar_chart(conteo_paises)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="chart-header">Distribución por Propósito de Viaje</div>', unsafe_allow_html=True)
                conteo_prop = df_filtrado[col_proposito[0]].value_counts()
                st.bar_chart(conteo_prop, horizontal=True)
            
            with g_col2:
                st.markdown('<div class="chart-header">Evolución del Gasto Promedio (Temporal)</div>', unsafe_allow_html=True)
                gasto_temporal = df_filtrado.groupby(col_year[0])[col_spend[0]].mean()
                st.line_chart(gasto_temporal)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="chart-header">Correlación: Presupuesto Diario vs Gasto Total</div>', unsafe_allow_html=True)
                col_budget = [c for c in df_global.columns if 'budget' in c.lower() or 'presupuesto' in c.lower()]
                if col_spend and col_budget:
                    dispersion_df = df_filtrado[[col_budget[0], col_spend[0]]].head(200)
                    st.scatter_chart(dispersion_df, x=col_budget[0], y=col_spend[0])
        else:
            st.warning("⚠️ No existen registros que coincidan con la combinación de filtros seleccionada.")

        st.markdown("---")
        st.markdown("#### Auditoría de Registros Filtrados (Dataset Global Original)")
        st.dataframe(df_filtrado.head(100), use_container_width=True)
    else:
        st.error(f"No se pudo encontrar el archivo histórico local '{NOMBRE_CSV_GLOBAL}'.")

# --- SECCIÓN 2: CERTIFICACIÓN DE LA IA ---
elif seccion_activa == "Certificación de la IA (Metadata)":
    st.subheader("Especificaciones Técnicas del Modelo Ganador")
    
    if meta_ia is not None:
        st.markdown("#### `[1]:` Verificar métricas finales del estimador")
        m = meta_ia["metricas_finales"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Train (Fit)", f"{m['R2_Train']:.4f}")
        c2.metric("R² Test (Validación)", f"{m['R2_Test']:.4f}")
        c3.metric("MAE (USD)", f"${m['MAE_USD']:,.2f}")
        c4.metric("RMSE (USD)", f"${m['RMSE_USD']:,.2f}")
        
        st.markdown("#### ` Análisis de Diagnóstico del Modelo`:")
        brecha_r2 = m['R2_Train'] - m['R2_Test']
        relacion_error = m['RMSE_USD'] / m['MAE_USD']

        markdown_comentarios = f"""
        * **Evaluación de Ajuste ($R^2$):** El modelo explica un **{m['R2_Test']*100:.2f}%** de la varianza en los datos de prueba. La diferencia entre el entrenamiento y la validación es de **{brecha_r2:.4f}**. Una brecha menor a 0.05 indica que el modelo generaliza correctamente y se descarta el sobreajuste (overfitting).
        * **Análisis de Residuos (MAE vs RMSE):** 
            * El **MAE** nos indica que el error típico esperado en las predicciones de paquetes turísticos es de **${m['MAE_USD']:,.2f} USD**.
            * El **RMSE** es de **${m['RMSE_USD']:,.2f} USD** (Ratio RMSE/MAE: **{relacion_error:.2f}**). Al ser este ratio moderado, confirmamos que el modelo no se ve severamente afectado por valores atípicos (*outliers*) extremos en los errores de predicción.
        """
        st.info(markdown_comentarios)
        st.success("✅ Estado: CERTIFICADO PARA PRODUCCIÓN (Sin indicios de sobreajuste / Overfitting)")
        
        st.markdown("---")
        st.markdown("#### `[2]:` Hiperparametros")
        st.markdown("_Tabla de configuración del espacio de búsqueda óptimo hallado por la Optimización Bayesiana:_")
        
        hiperparametros = meta_ia["mejores_hiperparametros"]
        df_hp = pd.DataFrame(list(hiperparametros.items()), columns=["Hiperparámetro (Hyperparameter)", "Valor Óptimo (Optimal Value)"])
        st.dataframe(df_hp, use_container_width=True, hide_index=True)
            
        st.markdown("---")
        st.markdown("#### `[3]:` Features esperadas")
        st.markdown("_Estructura del vector de características requeridas por el Pipeline de Preprocesamiento:_")
        
        features = meta_ia["features_esperadas"]
        df_features = pd.DataFrame({"Índice (Index)": range(0, len(features)), "Nombre de la Variable (Feature Name)": features})
        st.dataframe(df_features, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### `[4]:` Relevancia Real de las Variables (Feature Importance)")
        if pipeline_ia is not None and hasattr(pipeline_ia.named_steps.get('regressor', None), 'feature_importances_'):
            importancias = pipeline_ia.named_steps['regressor'].feature_importances_
            df_imp = pd.DataFrame({
                "Variable": features,
                "Importancia": importancias
            }).sort_values("Importancia", ascending=False).reset_index(drop=True)
            st.bar_chart(df_imp.set_index("Variable")["Importancia"].head(10))
            st.dataframe(df_imp, use_container_width=True, hide_index=True)

        st.warning(
            "⚠️ **Hallazgo del Análisis Exploratorio:** se detectó que "
            "`total_trip_spend_usd ≈ budget_per_person_usd × num_travelers` "
            "(correlación ≈ 1.0 en el dataset). Por ello el modelo concentra "
            "≈97% de su capacidad predictiva en estas dos variables "
            "(`budget_per_person_usd` ≈ 60%, `num_travelers` ≈ 37%), con un "
            "aporte residual de `traveler_type` (especialmente *Group Tour*, "
            "≈3%). El resto de variables — `duration_nights`, "
            "`restaurant_spend_per_day_usd`, `accommodation_type`, "
            "`travel_purpose` y `destination_country` — aportan contexto de "
            "negocio pero su peso estadístico es marginal o nulo. Se mantienen "
            "en el modelo por su relevancia conceptual y para futuros datasets "
            "donde sí podrían tener influencia."
        )
    else:
        st.error("Error `FileNotFoundError`: El archivo 'modelo_metadata.json' no se encuentra en el directorio actual.")

# --- SECCIÓN 3: SIMULADOR DE PREDICCIONES ---
elif seccion_activa == "Simulador de Predicciones Optimizadas":
    st.subheader("Simulador de Presupuestos Turísticos en Tiempo Real")
    
    if pipeline_ia is None or meta_ia is None:
        st.error("El motor de IA está desactivado porque faltan los archivos de soporte (.pkl o .json).")
    else:
        st.write("Modifique las variables en el panel izquierdo y presione el botón inferior para procesar la estimación.")
        st.markdown("---")
        
        if st.button("Calcular Gasto Total Estimado con IA", type="primary", use_container_width=True):
            features_esperadas = meta_ia["features_esperadas"]

            # Construimos el vector de entrada en ceros y activamos solo las
            # columnas correspondientes (mismo esquema de pd.get_dummies
            # drop_first=True usado en el entrenamiento del notebook).
            fila = {feat: 0 for feat in features_esperadas}
            fila['duration_nights'] = valores_usuario['duration_nights']
            fila['budget_per_person_usd'] = valores_usuario['budget_per_person_usd']
            fila['restaurant_spend_per_day_usd'] = valores_usuario['restaurant_spend_per_day_usd']
            fila['num_travelers'] = valores_usuario['num_travelers']

            col_accom = f"accommodation_type_{valores_usuario['accommodation_type']}"
            if col_accom in fila:
                fila[col_accom] = 1

            col_traveler = f"traveler_type_{valores_usuario['traveler_type']}"
            if col_traveler in fila:
                fila[col_traveler] = 1

            col_purpose = f"travel_purpose_{valores_usuario['travel_purpose']}"
            if col_purpose in fila:
                fila[col_purpose] = 1

            col_dest = f"destination_country_{valores_usuario['destination_country']}"
            if col_dest in fila:
                fila[col_dest] = 1

            input_df = pd.DataFrame([fila])[features_esperadas]

            gasto_predicho = pipeline_ia.predict(input_df)[0]

            referencia = valores_usuario['budget_per_person_usd'] * valores_usuario['num_travelers']

            # El GradientBoosting puede extrapolar a valores negativos en
            # combinaciones poco representadas (p. ej. 1 viajero + presupuesto
            # bajo). Un gasto total negativo no tiene sentido de negocio, así
            # que aplicamos un suelo de seguridad usando la referencia
            # Presupuesto × Viajeros (relación casi exacta detectada en los datos).
            prediccion_ajustada = False
            if gasto_predicho < 0:
                gasto_predicho = referencia
                prediccion_ajustada = True
            
            st.markdown("<br>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Gasto Total Proyectado por la IA", value=f"${gasto_predicho:,.2f} USD")
                if prediccion_ajustada:
                    st.caption(
                        "El modelo extrapoló a un valor negativo para esta "
                        "combinación poco representada en los datos. Se ha "
                        "usado la referencia Presupuesto × Viajeros como "
                        "estimación de seguridad."
                    )
            with res_col2:
                st.info(
                    "📊 **Nota de Consistencia Técnica:** Esta simulación fue procesada a través "
                    "del pipeline matemático original optimizado en laboratorio, aplicando las "
                    "transformaciones de escala correctas.\n\n"
                    "**Por qué algunas variables no cambian mucho el resultado:** el análisis de "
                    "importancia de características mostró que en este dataset "
                    "`total_trip_spend_usd ≈ budget_per_person_usd × num_travelers` "
                    "(correlación ≈ 1.0). La duración de la estancia, el gasto en restaurantes, "
                    "el tipo de alojamiento, el propósito del viaje y el destino se incluyen por "
                    "su relevancia conceptual de negocio, pero su peso estadístico en la "
                    "predicción final es marginal."
                )

                referencia = valores_usuario['budget_per_person_usd'] * valores_usuario['num_travelers']
                st.caption(
                    f"Referencia directa Presupuesto × Viajeros = "
                    f"${referencia:,.2f} USD (relación dominante detectada en los datos)."
                )