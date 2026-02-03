import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Configuración de página
st.set_page_config(layout="wide", page_title="Planeador GeoKAM - Febrero 2026")

# --- FUNCIONES DE APOYO ---
def agregar_poligono_zona(mapa_obj, coords, color):
    if len(coords) < 3: return
    try:
        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices].tolist()
        hull_coords.append(hull_coords[0])
        folium.Polygon(locations=hull_coords, color=None, fill=True, fill_color=color, fill_opacity=0.1).add_to(mapa_obj)
    except: pass

def ordenar_ruta_optima(points_df):
    if len(points_df) <= 2: return list(points_df.index)
    coords = points_df[['num_latitud', 'num_longitud']].values
    dist_matrix = (cdist(coords, coords, metric='euclidean') * 100000).astype(int)
    manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index, ordered_indices = routing.Start(0), []
        while not routing.IsEnd(index):
            ordered_indices.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return points_df.iloc[ordered_indices].index.tolist()
    return list(points_df.index)

def generar_html_popup(row, color):
    """Genera el globo de información con barras de GPV (como en tu Jupyter)."""
    cols_gpv = ['gpv_l6m', 'gpv_l5m', 'gpv_l4m', 'gpv_l3m', 'gpv_l2m', 'gpv_lm']
    valores = [float(row.get(c, 0)) if pd.notnull(row.get(c, 0)) else 0 for c in cols_gpv]
    max_val = max(valores) if max(valores) > 0 else 1
    
    bars_html = ""
    for m, v in zip(['L6M', 'L5M', 'L4M', 'L3M', 'L2M', 'LM'], valores):
        h = (v / max_val) * 60  # Altura proporcional a 60px
        bars_html += f"""
        <div style="display:flex; flex-direction:column; align-items:center; margin:0 2px;">
            <div style="background-color:{color}; width:12px; height:{h}px; border-radius:2px 2px 0 0;"></div>
            <span style="font-size:7px; color:#666;">{m}</span>
        </div>"""
    
    return f"""
    <div style="width:220px; font-family:Arial; font-size:11px; line-height:1.4;">
        <b style="color:{color}; font-size:13px;">{row['nbr_comercio']}</b><br>
        <b>RUC:</b> {row['num_documento']} | <b>Tel:</b> {row.get('telefono','-')}<br>
        <b>Etiqueta:</b> <span style="color:#E67E22;">{row.get('etiqueta','-')}</span>
        <hr style="margin:5px 0; border:0; border-top:1px solid #eee;">
        <div style="display:flex; align-items:flex-end; height:75px; background:#f9f9f9; padding:5px; border-radius:4px;">
            {bars_html}
        </div>
        <p style="font-size:9px; color:#888; margin-top:5px; text-align:center;">Tendencia GPV (Últimos 6 meses)</p>
    </div>
    """

# --- CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    return pd.read_csv("data.csv") 

df = cargar_datos()

# --- SIDEBAR (FILTROS) ---
st.sidebar.header(" Control de Estrategia")
kam_sel = st.sidebar.selectbox("Seleccionar KAM", sorted(df['kam_final_reasignado'].unique()))
n_clusters = st.sidebar.slider("Número de Zonas (Días)", 1, 40, 20)
filtro_etiqueta = st.sidebar.selectbox("Filtrar Etiqueta", ['TODOS'] + sorted(df['etiqueta'].unique().tolist()))
buscar_txt = st.sidebar.text_input("🔍 Buscar por Nombre o RUC")
solo_lima = st.sidebar.checkbox("Solo Lima Metropolitana", value=True)

# --- PROCESAMIENTO ---
df_f = df[df['kam_final_reasignado'] == kam_sel].copy()
if solo_lima: df_f = df_f[df_f['Departamento'].str.upper() == 'LIMA']
if filtro_etiqueta != 'TODOS': df_f = df_f[df_f['etiqueta'] == filtro_etiqueta]
if buscar_txt:
    df_f = df_f[df_f['nbr_comercio'].str.contains(buscar_txt, case=False, na=False) | 
                df_f['num_documento'].astype(str).str.contains(buscar_txt, na=False)]

# --- DASHBOARD PRINCIPAL ---
st.title(f" Rutas KAM: {kam_sel}")

col1, col2 = st.columns([3, 1])

with col1:
    # Lógica de Metas (Andrea)
    es_esp = kam_sel.upper() in ['WALTHER FAJARDO', 'WALTER YARLEQUE', 'LUIS DE LUCIO', 'JUAN BUSTAMANTE']
    meta_txt = "🔴 Mínimo 40 visitas" if es_esp else "🟢 Máximo 25 visitas"
    st.info(f"**Meta:** {meta_txt} | **Cartera actual:** {len(df_f)} comercios")

    # Mapa
    if not df_f.empty:
        m = folium.Map(location=[df_f['num_latitud'].mean(), df_f['num_longitud'].mean()], zoom_start=12, tiles='cartodbpositron')
        colores = ['#E6194B', '#4363D8', '#3CB44B', '#F58231', '#911EB4', '#42E6F2', '#F032E6', '#BFEF45', '#008080']
        
        # Clustering
        cls_r = max(1, min(n_clusters, len(df_f)))
        kmeans = KMeans(n_clusters=cls_r, random_state=42, n_init='auto')
        df_f['cluster'] = kmeans.fit_predict(df_f[["num_latitud", "num_longitud"]])

        for i, cid in enumerate(sorted(df_f['cluster'].unique())):
            color = colores[i % 9]
            df_c = df_f[df_f['cluster'] == cid].copy()
            idx_ruta = ordenar_ruta_optima(df_c)
            df_c = df_c.loc[idx_ruta]
            
            # Geometría
            agregar_poligono_zona(m, df_c[['num_latitud', 'num_longitud']].values, color)
            folium.PolyLine(df_c[['num_latitud', 'num_longitud']].values, color=color, weight=2, opacity=0.4).add_to(m)
            
            # Marcadores con Popup Enriquecido
            for _, r in df_c.iterrows():
                folium.CircleMarker(
                    location=[r['num_latitud'], r['num_longitud']],
                    radius=7, color=color, fill=True, fill_opacity=0.7,
                    popup=folium.Popup(generar_html_popup(r, color), max_width=250),
                    tooltip=f"{r['nbr_comercio']} | GPV: S/ {r.get('gpv',0):,.0f}"
                ).add_to(m)
        
        st_folium(m, use_container_width=True, height=600)
    else:
        st.error("No se encontraron comercios con los filtros aplicados.")

with col2:
    st.subheader(" Distribución")
    if not df_f.empty:
        conteo = df_f['etiqueta'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(conteo, labels=conteo.index, autopct='%1.1f%%', startangle=140, colors=colores)
        ax.axis('equal')
        st.pyplot(fig)
        
        st.write("---")
        st.write("**Top Comercios (GPV Feb):**")
        st.dataframe(df_f.sort_values('gpv', ascending=False)[['nbr_comercio', 'gpv']].head(10), hide_index=True)
