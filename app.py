import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Configuración de página premium y ancha
st.set_page_config(
    page_title="Turismo ML App Premium", 
    page_icon="🧳",
    layout="wide"
)

# =====================================================================
# 🎨 CAPA DE DISEÑO VISUAL: ESTILO NATIVO CLARO Y VIVO
# =====================================================================
st.markdown("""
    <style>
    /* Fondo de la app claro y limpio */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Forzar a los números de st.metric a verse en azul vivo/eléctrico y no cortarse */
    [data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }
    
    /* Estilo para las etiquetas superiores de las métricas */
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
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
    
    /* Pestañas (Tabs) estilo minimalista con fondo suave */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b !important;
        font-weight: 600;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 2px 8px rgba(148, 163, 184, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 🧭 DETECTOR AUTOMÁTICO DE RUTAS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
RUTA_PKL = BASE_DIR / "modelo_turismo.pkl"
RUTA_JSON = BASE_DIR / "modelo_metadata.json"
NOMBRE_CSV = "dataset_viajes_procesado.csv"
RUTA_CSV = BASE_DIR / NOMBRE_CSV

# =====================================================================
# 🧠 FUNCIÓN DE CARGA EN MEMORIA (OPTIMIZADA CON CACHÉ)
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

# Pre-cargamos el CSV de forma segura para extraer los filtros reales de tus columnas
df_valores_sidebar = None
col_pais, col_proposito, col_spend = [], [], []

if RUTA_CSV.exists():
    df_valores_sidebar = pd.read_csv(RUTA_CSV)
    col_pais = [c for c in df_valores_sidebar.columns if 'country' in c.lower() or 'destino' in c.lower() or 'pais' in c.lower()]
    col_proposito = [c for c in df_valores_sidebar.columns if 'purpose' in c.lower() or 'proposito' in c.lower() or 'motivo' in c.lower()]
    col_spend = [c for c in df_valores_sidebar.columns if 'spend' in c.lower() or 'gasto' in c.lower() or 'total' in c.lower()]

# =====================================================================
# 🎛️ CONTROLES GLOBALES EN LA BARRA LATERAL (SIDEBAR)
# =====================================================================
st.sidebar.title("🎮 Panel de Control")

modo_app = st.sidebar.radio(
    "Selecciona qué deseas configurar:",
    ["📊 Filtros del Dashboard", "🔮 Parámetros del Simulador"]
)

filtro_pais = "Todos"
filtro_proposito = "Todos"
valores_usuario = {}

if modo_app == "📊 Filtros del Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.header("🕹️ Filtros del Dashboard Histórico")
    
    if df_valores_sidebar is not None:
        lista_paises = ["Todos"] + sorted(df_valores_sidebar[col_pais[0]].dropna().unique().tolist()) if col_pais else ["Todos"]
        lista_motivos = ["Todos"] + sorted(df_valores_sidebar[col_proposito[0]].dropna().unique().tolist()) if col_proposito else ["Todos"]
    else:
        lista_paises = ["Todos"]
        lista_motivos = ["Todos"]

    filtro_pais = st.sidebar.selectbox("🗺️ País de Destino:", lista_paises)
    filtro_proposito = st.sidebar.selectbox("💼 Propósito de Viaje:", lista_motivos)

elif modo_app == "🔮 Parámetros del Simulador" and meta_ia is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Perfil del Viaje a Simular")
    
    for feature in meta_ia["features_esperadas"]:
        if feature == 'duration_nights':
            valores_usuario[feature] = st.sidebar.slider("🔢 Duración de la Estancia (Noches):", min_value=1, max_value=30, value=7)
        elif feature == 'budget_per_person_usd':
            valores_usuario[feature] = st.sidebar.slider("💰 Presupuesto Diario Base (USD):", min_value=10, max_value=1000, value=150)
        else:
            nombre_limpio = feature.replace("traveler_type_", "").replace("_", " ").title()
            seleccion_texto = st.sidebar.selectbox(f"📍 ¿Aplica: {nombre_limpio}?", ["No", "Sí"], index=0)
            valores_usuario[feature] = 1 if seleccion_texto == "Sí" else 0

# =====================================================================
# 🏛️ INTERFAZ PRINCIPAL DE PESTAÑAS
# =====================================================================
st.title("🧳 Sistema Inteligente de Tendencias de Turismo")
st.write("Plataforma BI Analítica & Motor Predictivo de Inteligencia Artificial.")

tab1, tab2, tab3 = st.tabs([
    "Dashboard de Negocio",
    "Certificación de la IA (Metadata)",
    "Simulador de Predicciones Optimizadas"
])

# =====================================================================
# PESTAÑA 1: DASHBOARD DE NEGOCIO REAL
# =====================================================================
with tab1:
    if RUTA_CSV.exists():
        df = df_valores_sidebar.copy()
        df_filtrado = df.copy()
        
        if col_pais and filtro_pais != "Todos":
            df_filtrado = df_filtrado[df_filtrado[col_pais[0]] == filtro_pais]
            
        if col_proposito and filtro_proposito != "Todos":
            df_filtrado = df_filtrado[df_filtrado[col_proposito[0]] == filtro_proposito]
            
        st.subheader("📊 Panel Analítico de Rendimiento Turístico")
        st.caption(f"Filtros Activos | 📍 Destino: **{filtro_pais}** | 💼 Propósito: **{filtro_proposito}**")
        
        # --- CÁLCULO SEGURO ---
        total_segmentado = int(len(df_filtrado))
        total_original = int(len(df))
        
        if col_spend and total_segmentado > 0:
            gasto_medio_num = float(df_filtrado[col_spend[0]].mean())
            gasto_total_num = float(df_filtrado[col_spend[0]].sum())
        else:
            gasto_medio_num = 0.00
            gasto_total_num = 0.00

        # --- 🛠️ FUNCIÓN DE FORMATEO COMPACTO INTELIGENTE ---
        def formatear_corto_usd(valor):
            if valor >= 1_000_000:
                # Si pasa de un millón, muestra ej: $115.50 M USD (Corto y limpio)
                return f"${valor / 1_000_000:.2f} M USD"
            elif valor >= 1_000:
                # Si es un número intermedio largo, quita centavos ej: $11,550 USD
                return f"${valor:,.0f} USD"
            else:
                return f"${valor:,.2f} USD"

        # --- CONTENEDORES KPI NATIVOS, DINÁMICOS Y COMPACTOS ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric(
                label="Muestra Segmentada", 
                value=f"{total_segmentado:,d}"
            )

        with kpi2:
            st.metric(
                label="Gasto Medio por Viaje", 
                value=formatear_corto_usd(gasto_medio_num)
            )

        with kpi3:
            st.metric(
                label="Volumen Financiero Total", 
                value=formatear_corto_usd(gasto_total_num)
            )
            
        with kpi4:
            st.metric(
                label="Dataset Original", 
                value=f"{total_original:,d}"
            )
        
        st.markdown("---")
        
        # --- FILA DE GRÁFICOS INTERACTIVOS ---
        if len(df_filtrado) > 0:
            g_col1, g_col2 = st.columns(2)
            
            with g_col1:
                st.markdown('<div class="chart-header">🗺️ Top Destinos más Demandados</div>', unsafe_allow_html=True)
                col_destino_real = [c for c in df.columns if 'destination_country' in c or 'country' in c.lower() or 'destino' in c.lower()][0]
                conteo_paises = df_filtrado[col_destino_real].value_counts().head(6)
                st.bar_chart(conteo_paises)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="chart-header">💼 Distribución por Propósito de Viaje</div>', unsafe_allow_html=True)
                col_purpose_real = [c for c in df.columns if 'travel_purpose' in c or 'purpose' in c.lower() or 'proposito' in c.lower()][0]
                conteo_prop = df_filtrado[col_purpose_real].value_counts()
                st.bar_chart(conteo_prop, horizontal=True)
            
            with g_col2:
                st.markdown('<div class="chart-header">🧑‍🤝‍🧑 Análisis por Tipo de Viajero</div>', unsafe_allow_html=True)
                col_traveler_real = [c for c in df.columns if 'traveler_type' in c or 'traveler' in c.lower()][0]
                conteo_traveler = df_filtrado[col_traveler_real].value_counts()
                st.bar_chart(conteo_traveler)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="chart-header">💰 Correlación: Presupuesto Diario vs Gasto Total</div>', unsafe_allow_html=True)
                col_budget = [c for c in df.columns if 'budget' in c.lower() or 'presupuesto' in c.lower()][0]
                if col_spend and col_budget:
                    dispersion_df = df_filtrado[[col_budget, col_spend[0] if isinstance(col_spend, list) else col_spend]].head(200)
                    st.scatter_chart(dispersion_df, x=col_budget, y=col_spend[0] if isinstance(col_spend, list) else col_spend)
        else:
            st.warning("⚠️ No existen registros que coincidan con la combinación de filtros seleccionada.")

        st.markdown("---")
        st.markdown("#### 📋 Auditoría de Registros Filtrados (Dataset Transaccional)")
        st.dataframe(df_filtrado.head(100), use_container_width=True)
        
    else:
        st.error(f"❌ No se pudo encontrar el archivo histórico local '{NOMBRE_CSV}'.")

# =====================================================================
# PESTAÑA 2: CERTIFICACIÓN DE LA IA
# =====================================================================
with tab2:
    st.subheader("🧠 Especificaciones Técnicas del Modelo Ganador")
    if meta_ia is not None:
        m = meta_ia["metricas_finales"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Entrenamiento (Fit)", f"{m['R2_Train']:.4f}")
        c2.metric("R² Validación (Test)", f"{m['R2_Test']:.4f}")
        c3.metric("Error Medio Absoluto (MAE)", f"${m['MAE_USD']:,.2f} USD")
        c4.metric("Desviación Estándar (RMSE)", f"${m['RMSE_USD']:,.2f} USD")
        
        st.success("✅ Estado del modelo: CERTIFICADO PARA PRODUCCIÓN")
        
        with st.expander("⚙️ Ver hiperparámetros óptimos"):
            st.json(meta_ia["mejores_hiperparametros"])
        with st.expander("📋 Ver vector de características"):
            st.write(meta_ia["features_esperadas"])
    else:
        st.error("❌ Los metadatos técnicos no se pudieron cargar.")

# =====================================================================
# PESTAÑA 3: PREDECIR
# =====================================================================
with tab3:
    st.subheader("🔮 Simulador de Presupuestos Turísticos en Tiempo Real")
    
    if pipeline_ia is None or meta_ia is None:
        st.error("❌ El motor de IA está desactivado.")
    else:
        if modo_app != "🔮 Parámetros del Simulador":
            st.warning("💡 **Atención:** Para modificar el perfil, cambia la opción en el Panel de Control izquierdo a **'Parámetros del Simulador'**.")
        
        st.markdown("---")
        if st.button("🔮 Calcular Gasto Total Estimado con IA", type="primary", use_container_width=True):
            input_df = pd.DataFrame([valores_usuario])
            input_df = input_df[meta_ia["features_esperadas"]]
            gasto_predicho = pipeline_ia.predict(input_df)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(
                    label="💸 Gasto Total Proyectado por la IA", 
                    value=formatear_corto_usd(gasto_predicho)
                )
            with res_col2:
                st.info("💡 **Nota de Consistencia Técnica:** Esta simulación fue procesada a través del pipeline matemático original.")