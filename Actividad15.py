"""
app.py - Dashboard de Productividad y Calidad en Líneas de Producción
Asignatura: Herramientas de Visualización para la Inteligencia de Negocios
Autor: [Tu Nombre]
Descripción: 
Aplicación Streamlit que permite analizar el desempeño de líneas de producción
mediante indicadores de productividad (unidades/hora) y tasa de defectos.
Responde la pregunta: ¿Qué líneas presentan mayor productividad y menor defectos?
"""

# ------------------------------------------------------------
# 1. Importación de librerías
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime, date

# ------------------------------------------------------------
# 2. Configuración de la página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Industrial",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos neutros profesionales
st.markdown("""
<style>
    :root {
        --primary: #333333;
        --secondary: #555555;
        --bg-light: #F5F5F5;
    }
    .stMetric { text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0px 0px; padding: 8px 16px; }
    h1, h2, h3 { color: #333333; }
    .chart-description {
        font-size: 0.9rem; color: #555555; font-style: italic;
        margin: 0.5rem 0 1rem 0; padding: 0.5rem;
        background-color: #F5F5F5; border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 3. Funciones de datos (con caché)
# ------------------------------------------------------------

@st.cache_data
def generar_datos_ejemplo():
    """
    Genera un dataset sintético de producción para 4 líneas durante 6 meses.
    """
    np.random.seed(42)
    lineas = ["Línea A", "Línea B", "Línea C", "Línea D"]
    fechas = pd.date_range(start="2025-01-02", end="2025-06-30", freq="D")
    datos = []
    params = {
        "Línea A": {"base_unidades": 800, "base_defectos_pct": 0.03, "horas_base": 16, "inactividad_base": 15},
        "Línea B": {"base_unidades": 750, "base_defectos_pct": 0.05, "horas_base": 16, "inactividad_base": 20},
        "Línea C": {"base_unidades": 600, "base_defectos_pct": 0.04, "horas_base": 12, "inactividad_base": 10},
        "Línea D": {"base_unidades": 900, "base_defectos_pct": 0.02, "horas_base": 16, "inactividad_base": 25},
    }
    for linea in lineas:
        p = params[linea]
        for fecha in fechas:
            dia_semana = fecha.weekday()
            if dia_semana >= 5:
                factor = np.random.uniform(0.3, 0.5)
            else:
                factor = np.random.uniform(0.8, 1.2)
            mes = fecha.month
            estacionalidad = 1.0 + 0.05 * np.sin(2 * np.pi * mes / 12)
            unidades = int(p["base_unidades"] * factor * estacionalidad)
            horas = p["horas_base"] * (0.9 + 0.2 * np.random.random())
            defectos = np.random.binomial(unidades, p["base_defectos_pct"] * np.random.uniform(0.8, 1.2))
            tiempo_inactivo = max(0, int(np.random.normal(p["inactividad_base"], scale=5)))
            datos.append({
                "fecha": fecha.strftime("%Y-%m-%d"),
                "linea_produccion": linea,
                "unidades_producidas": unidades,
                "unidades_defectuosas": defectos,
                "horas_trabajadas": round(horas, 1),
                "tiempo_inactivo_min": tiempo_inactivo
            })
    return pd.DataFrame(datos)

@st.cache_data
def cargar_csv(archivo):
    """Carga un DataFrame desde CSV. Retorna (df, None) o (None, mensaje_error)."""
    if archivo is None:
        return None, None
    try:
        df = pd.read_csv(archivo)
        return df, None
    except Exception as e:
        return None, f"Error al leer el archivo: {e}"

def validar_columnas(df):
    """Verifica las columnas obligatorias y tipos numéricos."""
    requeridas = {"fecha", "linea_produccion", "unidades_producidas",
                  "unidades_defectuosas", "horas_trabajadas", "tiempo_inactivo_min"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        return False, f"Faltan las columnas: {', '.join(faltantes)}"
    for col in ["unidades_producidas", "unidades_defectuosas", "horas_trabajadas", "tiempo_inactivo_min"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                return False, f"La columna '{col}' debe ser numérica."
    return True, None

# ------------------------------------------------------------
# 4. Preprocesamiento y KPIs
# ------------------------------------------------------------

def preprocesar(df):
    """Convierte fechas, ordena y calcula métricas: productividad, tasa de defectos, eficiencia."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if df["fecha"].isna().any():
        st.warning("Algunas fechas no pudieron ser interpretadas; se mantendrán como texto.")
    else:
        df["fecha"] = df["fecha"].dt.date
    df["productividad"] = np.where(df["horas_trabajadas"] > 0,
                                   df["unidades_producidas"] / df["horas_trabajadas"], 0)
    df["tasa_defectos"] = np.where(df["unidades_producidas"] > 0,
                                   df["unidades_defectuosas"] / df["unidades_producidas"], 0)
    df["eficiencia"] = np.where(df["horas_trabajadas"] + df["tiempo_inactivo_min"]/60 > 0,
                                 df["horas_trabajadas"] / (df["horas_trabajadas"] + df["tiempo_inactivo_min"]/60), 0)
    return df.sort_values(["linea_produccion", "fecha"])

# ------------------------------------------------------------
# 5. Formatos
# ------------------------------------------------------------

def formato_porcentaje(val):
    return f"{val:.2%}"

def formato_decimal(val, dec=2):
    return f"{val:,.{dec}f}"

# ------------------------------------------------------------
# 6. Visualizaciones (colores neutros, intensidades para múltiples)
# ------------------------------------------------------------

def grafico_evolucion_productividad(df, titulo):
    """
    Plotly: líneas de productividad diaria por línea.
    Utiliza una paleta monocromática gris para diferenciar las líneas.
    """
    # Generar tantos tonos grises como líneas de producción haya
    num_lineas = df["linea_produccion"].nunique()
    # Escala de grises desde oscuro (#333333) hasta más claro (#B0B0B0)
    grises = [f"#{int(51 + i*(160-51)/max(1,num_lineas-1)):02x}{int(51 + i*(160-51)/max(1,num_lineas-1)):02x}{int(51 + i*(160-51)/max(1,num_lineas-1)):02x}" 
              for i in range(num_lineas)]
    # Alternativamente, usar px.colors.sequential.Greys (más simple)
    # Pero para control exacto, usamos la lista generada.
    fig = px.line(df, x="fecha", y="productividad", color="linea_produccion",
                  title=titulo, markers=False,
                  color_discrete_sequence=grises)
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Productividad (unidades/hora)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#333333")
    )
    return fig

def grafico_barras_defectos(df, titulo):
    """
    Altair: barras de tasa de defectos promedio por línea.
    Color neutro único para todas las barras.
    """
    resumen = df.groupby("linea_produccion", as_index=False).agg(
        tasa_defectos=("tasa_defectos", "mean"),
        unidades_totales=("unidades_producidas", "sum")
    ).round(3)
    chart = alt.Chart(resumen).mark_bar(color="#555555").encode(
        x=alt.X("linea_produccion:N", title="Línea de Producción", sort="-y"),
        y=alt.Y("tasa_defectos:Q", title="Tasa de Defectos Promedio", axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip("linea_produccion:N", title="Línea"),
            alt.Tooltip("tasa_defectos:Q", format=".2%", title="Tasa de Defectos"),
            alt.Tooltip("unidades_totales:Q", format=",")
        ]
    ).properties(
        title=titulo,
        width="container"
    ).configure_title(fontSize=16, anchor="start")
    return chart

# ------------------------------------------------------------
# 7. Interpretación analítica automática
# ------------------------------------------------------------

def insight_productividad(df):
    prod_media = df.groupby("linea_produccion")["productividad"].mean().sort_values(ascending=False)
    mejor = prod_media.index[0]
    valor = prod_media.iloc[0]
    peor = prod_media.index[-1]
    return (f"La **{mejor}** muestra la productividad promedio más alta ({formato_decimal(valor)} und/h), "
            f"mientras que **{peor}** es la de menor rendimiento. "
            "Optimizar los tiempos de inactividad en las líneas menos productivas podría mejorar el output global.")

def insight_defectos(df):
    calidad = df.groupby("linea_produccion")["tasa_defectos"].mean().sort_values()
    mejor = calidad.index[0]
    peor = calidad.index[-1]
    return (f"La línea **{mejor}** presenta la menor tasa de defectos ({formato_porcentaje(calidad.iloc[0])}), "
            f"convirtiéndola en referente de calidad. Por el contrario, **{peor}** necesita atención inmediata "
            f"con una tasa del {formato_porcentaje(calidad.iloc[-1])}. "
            "Se recomienda revisar los procedimientos de control de calidad en esa línea.")

# ------------------------------------------------------------
# 8. Interfaz principal
# ------------------------------------------------------------

def main():
    st.title("🏭 Dashboard de Inteligencia de Producción")
    st.markdown("""
    **Caso de negocio:** monitoreo de líneas de producción para identificar 
    cuellos de botella en productividad y calidad.  
    **Pregunta analítica:** *¿Qué líneas de producción presentan mayor productividad 
    y menor tasa de defectos?*
    """)
    st.divider()

    # ============ BARRA LATERAL ============
    with st.sidebar:
        st.header("⚙️ Configuración")
        archivo_csv = st.file_uploader("Cargar archivo CSV", type=["csv"],
                                       help="Columnas: fecha, linea_produccion, unidades_producidas, unidades_defectuosas, horas_trabajadas, tiempo_inactivo_min")
        if st.button("🔄 Restablecer datos de ejemplo"):
            st.cache_data.clear()
            st.rerun()

        # Carga/validación
        if archivo_csv is not None:
            df_crudo, error = cargar_csv(archivo_csv)
            if error:
                st.error(error)
                st.info("Se utilizará el dataset de ejemplo.")
                df_crudo = None
        else:
            df_crudo = None

        if df_crudo is None:
            st.info("Dataset de ejemplo: 4 líneas de producción, 6 meses de datos diarios.")
            df = generar_datos_ejemplo()
        else:
            valido, mensaje = validar_columnas(df_crudo)
            if not valido:
                st.error(mensaje)
                st.warning("El archivo no es válido. Se cargan datos de ejemplo.")
                df = generar_datos_ejemplo()
            else:
                df = df_crudo.copy()

        df = preprocesar(df)

        # Filtros
        st.subheader("🔍 Filtros")
        lineas_disponibles = sorted(df["linea_produccion"].unique())
        lineas_seleccionadas = st.multiselect(
            "Líneas de producción",
            options=lineas_disponibles,
            default=lineas_disponibles
        )

        inicio = None
        fin = None
        fechas_seleccionadas = None
        es_datetime = pd.api.types.is_datetime64_any_dtype(df["fecha"]) or isinstance(df["fecha"].iloc[0], (datetime, date))

        if es_datetime:
            min_fecha = df["fecha"].min()
            max_fecha = df["fecha"].max()
            if isinstance(min_fecha, pd.Timestamp):
                min_fecha = min_fecha.date()
                max_fecha = max_fecha.date()
            rango = st.date_input("Rango de fechas", value=(min_fecha, max_fecha))
            if len(rango) == 2:
                inicio, fin = rango
            else:
                inicio, fin = min_fecha, max_fecha
        else:
            fechas_unicas = sorted(df["fecha"].unique())
            fechas_seleccionadas = st.multiselect("Fechas", fechas_unicas, default=fechas_unicas)

        st.divider()
        st.caption("Dashboard desarrollado para Herramientas de Visualización para la Inteligencia de Negocios.")

    # ============ APLICAR FILTROS ============
    df_filtrado = df[df["linea_produccion"].isin(lineas_seleccionadas)]

    if inicio is not None and fin is not None:
        if es_datetime:
            df_filtrado = df_filtrado[(df_filtrado["fecha"] >= inicio) & (df_filtrado["fecha"] <= fin)]
        else:
            df_filtrado["fecha_temp"] = pd.to_datetime(df_filtrado["fecha"])
            df_filtrado = df_filtrado[(df_filtrado["fecha_temp"] >= pd.Timestamp(inicio)) & 
                                      (df_filtrado["fecha_temp"] <= pd.Timestamp(fin))]
            df_filtrado.drop(columns="fecha_temp", inplace=True)
    elif fechas_seleccionadas is not None:
        df_filtrado = df_filtrado[df_filtrado["fecha"].isin(fechas_seleccionadas)]

    if df_filtrado.empty:
        st.warning("No hay datos para los filtros actuales. Ajuste la selección.")
        return

    # ============ PESTAÑAS ============
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen Ejecutivo", "📈 Productividad", "🔍 Calidad", "📋 Datos"])

    with tab1:
        st.subheader("Indicadores Clave de Desempeño (KPIs)")
        total_unidades = df_filtrado["unidades_producidas"].sum()
        total_defectos = df_filtrado["unidades_defectuosas"].sum()
        tasa_defectos_global = total_defectos / total_unidades if total_unidades > 0 else 0
        productividad_media = df_filtrado["productividad"].mean()
        eficiencia_media = df_filtrado["eficiencia"].mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Unidades Producidas", f"{total_unidades:,}")
        col2.metric("⚙️ Productividad Media", f"{productividad_media:.1f} und/h")
        col3.metric("⚠️ Tasa de Defectos", formato_porcentaje(tasa_defectos_global))
        col4.metric("⏱️ Eficiencia Operativa", f"{eficiencia_media:.1f}%")

        st.divider()
        st.subheader("Visión General")
        resumen = df_filtrado.groupby("linea_produccion").agg(
            Unidades=("unidades_producidas", "sum"),
            Productividad=("productividad", "mean"),
            Defectos=("tasa_defectos", "mean")
        ).reset_index()
        resumen["Defectos"] = resumen["Defectos"].apply(formato_porcentaje)
        st.dataframe(resumen, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Evolución Diaria de la Productividad")
        fig_prod = grafico_evolucion_productividad(df_filtrado, "Productividad por Línea (unidades/hora)")
        st.plotly_chart(fig_prod, use_container_width=True)
        st.markdown(insight_productividad(df_filtrado))

    with tab3:
        st.subheader("Comparación de Tasa de Defectos entre Líneas")
        fig_def = grafico_barras_defectos(df_filtrado, "Tasa de Defectos Promedio por Línea de Producción")
        st.altair_chart(fig_def, use_container_width=True)
        st.markdown(insight_defectos(df_filtrado))

    with tab4:
        st.subheader("Datos Filtrados")
        columnas_mostrar = ["fecha", "linea_produccion", "unidades_producidas",
                            "unidades_defectuosas", "productividad", "tasa_defectos", "eficiencia"]
        df_tabla = df_filtrado[columnas_mostrar].copy()
        df_tabla["tasa_defectos"] = df_tabla["tasa_defectos"].apply(formato_porcentaje)
        df_tabla["eficiencia"] = df_tabla["eficiencia"].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)

        csv = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar datos filtrados (CSV)",
            data=csv,
            file_name="produccion_filtrada.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()