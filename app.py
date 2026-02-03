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
st.set_page_config(layout="wide", page_title="Planeador KAM - Febrero 2026")

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

# --- CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    # Aquí asumo que df_final ya existe o lo cargas de un CSV
    return pd.read_csv("data.csv") 

df = cargar_datos()

# --- SIDEBAR (FILTROS) ---
st.sidebar.header("🎯 Filtros de Control")
kam_sel = st.sidebar.selectbox("Seleccionar KAM", sorted(df['kam_final_reasignado'].unique()))
n_clusters = st.sidebar.slider("Número de Zonas", 1, 40, 20)
filtro_etiqueta = st.sidebar.selectbox("Etiqueta", ['TODOS'] + sorted(df['etiqueta'].unique().tolist()))
buscar_txt = st.sidebar.text_input("🔍 Buscar Comercio o RUC")
solo_lima = st.sidebar.checkbox("Solo Lima Metrop.", value=True)

# --- PROCESAMIENTO ---
df_f = df[df['kam_final_reasignado'] == kam_sel].copy()
if solo_lima: df_f = df_f[df_f['Departamento'].str.upper() == 'LIMA']
if filtro_etiqueta != 'TODOS': df_f = df_f[df_f['etiqueta'] == filtro_etiqueta]
if buscar_txt:
    df_f = df_f[df_f['nbr_comercio'].str.contains(buscar_txt, case=False) | df_f['num_documento'].astype(str).str.contains(buscar_txt)]

# --- DASHBOARD PRINCIPAL ---
st.title(f"📍 Dashboard de Rutas: {kam_sel}")

col1, col2 = st.columns([3, 1])

with col1:
    # Lógica de Metas (Andrea)
    es_esp = kam_sel.upper() in ['WALTHER FAJARDO', 'WALTER YARLEQUE', 'LUIS DE LUCIO', 'JUAN BUSTAMANTE']
    meta_txt = "🔴 Mínimo 40 visitas" if es_esp else "🟢 Máximo 25 visitas"
    st.info(f"**Meta Mensual:** {meta_txt} | **Cartera:** {len(df_f)} comercios")

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
            agregar_poligono_zona(m, df_c[['num_latitud', 'num_longitud']].values, color)
            folium.PolyLine(df_c[['num_latitud', 'num_longitud']].values, color=color, weight=2, opacity=0.4).add_to(m)
            for _, r in df_c.iterrows():
                folium.CircleMarker([r['num_latitud'], r['num_longitud']], radius=6, color=color, fill=True).add_to(m)
        
        st_folium(m, width=900, height=500)
    else:
        st.error("No hay datos para esta selección.")

with col2:
    st.subheader(" Mix de Cartera")
    if not df_f.empty:
        conteo = df_f['etiqueta'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(conteo, labels=conteo.index, autopct='%1.1f%%', startangle=140)
        st.pyplot(fig)