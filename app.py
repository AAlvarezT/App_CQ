import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide", page_title="GeoKAM — Rutas Optimizada (Marzo 2026)")

DEFAULT_SPEED_KMH = 15.0  # "Factor Lima"
MAX_POINTS_PER_CLUSTER_SOFT = 85  # para no matar OR-Tools / performance
MAP_TILE = "CartoDB positron"


# ---------------------------
# HELPERS
# ---------------------------
def _safe_upper(x):
    try:
        return str(x).strip().upper()
    except Exception:
        return ""


def _to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia geodésica (km)."""
    R = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def route_distance_km(df_route):
    coords = df_route[["num_latitud", "num_longitud"]].to_numpy()
    if len(coords) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine_km(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    return float(total)


def compute_time_minutes(distance_km, speed_kmh=DEFAULT_SPEED_KMH):
    if speed_kmh <= 0:
        return None
    return float(distance_km / speed_kmh * 60.0)


def coalesce_series(a: pd.Series, b: pd.Series):
    """Prioriza a; si a es nulo/vacío, usa b."""
    a2 = a.copy()
    a2 = a2.replace("", np.nan)
    out = a2.combine_first(b.replace("", np.nan))
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el CSV a un esquema estable.
    Maneja duplicados tipo 'etiqueta', 'etiqueta.1' o 'kam_asignado.1'.
    """
    df = df.copy()

    # Pandas renombra columnas duplicadas como ".1", ".2" al leer CSV.
    # Queremos quedarnos con una sola 'etiqueta' y un solo 'kam_asignado'.
    def pick_dupe(base):
        cols = [c for c in df.columns if c == base or c.startswith(base + ".")]
        return cols

    # etiqueta final: prioriza 'etiqueta' base, luego 'etiqueta.1' si existe
    etiqueta_cols = pick_dupe("etiqueta")
    if len(etiqueta_cols) >= 2:
        df["etiqueta_final"] = coalesce_series(df[etiqueta_cols[0]].astype(str), df[etiqueta_cols[1]].astype(str))
    elif len(etiqueta_cols) == 1:
        df["etiqueta_final"] = df[etiqueta_cols[0]].astype(str)
    else:
        df["etiqueta_final"] = ""

    # kam_asignado final: igual
    kam_cols = pick_dupe("kam_asignado")
    if len(kam_cols) >= 2:
        df["kam_asignado_final"] = coalesce_series(df[kam_cols[0]].astype(str), df[kam_cols[1]].astype(str))
    elif len(kam_cols) == 1:
        df["kam_asignado_final"] = df[kam_cols[0]].astype(str)
    else:
        df["kam_asignado_final"] = ""

    # Limpieza de llaves / strings principales
    for c in ["num_documento", "ruc", "grp_economico", "nbr_comercio", "nbr_direccion_comercio",
              "Departamento", "nbr_provincia_hexagono", "nbr_distrito_hexagono", "tipo_kam_asignado",
              "correo_kam", "prioridad", "prioridad_final", "movimiento", "periodo_cartera", "tipo_lead"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Numéricos principales
    for c in ["num_latitud", "num_longitud", "gpvabo", "ntrx", "meta_tendencial", "meta_sow", "meta_total",
              "gpv", "gpv_lm", "gpv_l2m", "gpv_l3m", "gpv_l4m", "gpv_l5m", "gpv_l6m", "gpv_l7m", "gpv_l8m",
              "maduracion"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_float)

    # Normaliza KAM
    df["kam_asignado_final"] = df["kam_asignado_final"].apply(_safe_upper)

    # Normaliza etiqueta
    df["etiqueta_final"] = df["etiqueta_final"].fillna("").astype(str).str.strip()

    return df


def clean_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina puntos sin geo y filtra rangos razonables (Perú aprox)."""
    d = df.copy()
    d = d.dropna(subset=["num_latitud", "num_longitud"])
    # rango amplio Perú (ajustable)
    d = d[d["num_latitud"].between(-19.5, -0.5)]
    d = d[d["num_longitud"].between(-82.5, -66.0)]
    return d


def solve_tsp_ortools(coords: np.ndarray):
    """
    coords: array Nx2 [[lat, lon],...]
    Retorna: order (lista de índices) que recorre todos los puntos una vez.
    Nota: No cierra el ciclo (no vuelve al inicio) por defecto; lo hacemos si quieres en map.
    """
    n = len(coords)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Matriz de distancias (km) convertida a int para OR-Tools
    dist = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0
            else:
                d = haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                dist[i, j] = int(d * 1000)  # metros (int)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 vehículo, start=0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return dist[a, b]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Parametría básica (rápida y estable)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(2)  # sube a 5 si tienes pocos puntos/cluster

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        # fallback simple: orden por nearest-neighbor greedy
        return greedy_nn_order(coords)

    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        order.append(node)
        index = solution.Value(routing.NextVar(index))

    return order


def greedy_nn_order(coords: np.ndarray):
    """Fallback: nearest neighbor greedy."""
    n = len(coords)
    if n <= 1:
        return list(range(n))
    remaining = set(range(n))
    order = [0]
    remaining.remove(0)
    while remaining:
        last = order[-1]
        best = None
        best_d = float("inf")
        for j in remaining:
            d = haversine_km(coords[last, 0], coords[last, 1], coords[j, 0], coords[j, 1])
            if d < best_d:
                best_d = d
                best = j
        order.append(best)
        remaining.remove(best)
    return order


def approx_baseline_distance_km(df_cluster: pd.DataFrame):
    """
    Baseline simple para demostrar valor:
    - Ordena por lat+lon (tipo barrido) y calcula distancia
    """
    if df_cluster.empty:
        return 0.0
    tmp = df_cluster.sort_values(["num_latitud", "num_longitud"]).reset_index(drop=True)
    return route_distance_km(tmp)


def make_popup(row: pd.Series) -> str:
    parts = [
        f"<b>{row.get('nbr_comercio','S/N')}</b>",
        f"RUC/Doc: {row.get('num_documento','')}",
        f"Dirección: {row.get('nbr_direccion_comercio','')}",
        f"Distrito: {row.get('nbr_distrito_hexagono','')}",
        f"Provincia: {row.get('nbr_provincia_hexagono','')}",
        f"Departamento: {row.get('Departamento','')}",
        f"Etiqueta: {row.get('etiqueta_final','')}",
        f"KAM: {row.get('kam_asignado_final','')}",
        f"GPV (abo): {row.get('gpvabo','')}",
        f"NTRX: {row.get('ntrx','')}",
        f"Grupo Económico: {row.get('grp_economico','')}",
    ]
    return "<br>".join(parts)


def add_route_polyline(m: folium.Map, df_route: pd.DataFrame, name: str):
    coords = df_route[["num_latitud", "num_longitud"]].to_numpy().tolist()
    if len(coords) >= 2:
        folium.PolyLine(coords, weight=4, opacity=0.9, tooltip=name).add_to(m)


# ---------------------------
# AUTH
# ---------------------------
def auth_block():
    """
    Autenticación simple:
    - Si st.secrets["auth"]["enabled"] = true:
        login por user/pass y amarra el KAM permitido.
    - Caso contrario: modo libre (para pruebas internas).
    """
    auth_enabled = False
    users_cfg = {}
    demo_pwd = None

    try:
        auth_enabled = bool(st.secrets.get("auth", {}).get("enabled", False))
        users_cfg = dict(st.secrets.get("auth", {}).get("users", {}))
        demo_pwd = st.secrets.get("auth", {}).get("demo_password", None)
    except Exception:
        auth_enabled = False

    st.sidebar.markdown("### 🔐 Acceso")

    if not auth_enabled:
        st.sidebar.info("Auth desactivado (modo interno).")
        return {"ok": True, "kam_locked": None}

    user = st.sidebar.text_input("Usuario")
    pwd = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Ingresar"):
        pass

    if not user or not pwd:
        st.stop()

    # Demo gate (opcional): si coincide con demo_password, no bloquea KAM
    if demo_pwd and pwd == demo_pwd:
        st.sidebar.success("Acceso demo habilitado.")
        return {"ok": True, "kam_locked": None}

    u = users_cfg.get(user)
    if not u:
        st.sidebar.error("Usuario inválido.")
        st.stop()

    if pwd != u.get("password"):
        st.sidebar.error("Password incorrecto.")
        st.stop()

    kam_locked = _safe_upper(u.get("kam_asignado", ""))
    if not kam_locked:
        st.sidebar.error("Este usuario no tiene KAM asignado en secrets.")
        st.stop()

    st.sidebar.success(f"Acceso OK — KAM: {kam_locked}")
    return {"ok": True, "kam_locked": kam_locked}


# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data(show_spinner=True)
def load_data(path="data.csv"):
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)

    # Asegura columnas mínimas
    needed = [
        "num_documento", "nbr_comercio", "nbr_direccion_comercio", "Departamento",
        "nbr_provincia_hexagono", "nbr_distrito_hexagono", "num_latitud", "num_longitud",
        "etiqueta_final", "kam_asignado_final"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en data.csv: {missing}")

    # Limpieza geo
    df = clean_geo(df)
    return df


# ---------------------------
# UI
# ---------------------------
st.title("GeoKAM — Mapa y Rutas Optimizada (Marzo 2026)")
st.caption(
    "Filtra por KAM / etiqueta / ubicación. Genera rutas optimizadas por clusters y muestra métricas de distancia/tiempo."
)

auth = auth_block()

# Carga data.csv (se asume en el mismo folder que app.py)
DATA_PATH = "data.csv"
if not os.path.exists(DATA_PATH):
    st.error("No encuentro data.csv junto a app.py. Sube data.csv a la misma carpeta del deploy.")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error cargando data.csv: {e}")
    st.stop()

# Si auth bloquea KAM, forzamos ese KAM.
all_kams = sorted([k for k in df["kam_asignado_final"].dropna().unique() if str(k).strip() != ""])
if auth["kam_locked"]:
    if auth["kam_locked"] not in all_kams:
        st.error(f"Tu KAM '{auth['kam_locked']}' no aparece en el data.csv.")
        st.stop()
    kam_sel = auth["kam_locked"]
    st.sidebar.write(f"✅ KAM fijo: {kam_sel}")
else:
    kam_sel = st.sidebar.selectbox("Seleccionar KAM", all_kams)

df_f = df[df["kam_asignado_final"] == kam_sel].copy()

# Filtros extra
st.sidebar.markdown("### 🎯 Filtros")
etiquetas = sorted([x for x in df_f["etiqueta_final"].dropna().unique() if str(x).strip() != ""])
et_sel = st.sidebar.multiselect("Etiqueta(s)", etiquetas, default=[])
if et_sel:
    df_f = df_f[df_f["etiqueta_final"].isin(et_sel)].copy()

dept = sorted([x for x in df_f["Departamento"].dropna().unique() if str(x).strip() != ""])
dept_sel = st.sidebar.multiselect("Departamento", dept, default=[])
if dept_sel:
    df_f = df_f[df_f["Departamento"].isin(dept_sel)].copy()

dist = sorted([x for x in df_f["nbr_distrito_hexagono"].dropna().unique() if str(x).strip() != ""])
dist_sel = st.sidebar.multiselect("Distrito", dist, default=[])
if dist_sel:
    df_f = df_f[df_f["nbr_distrito_hexagono"].isin(dist_sel)].copy()

buscar_txt = st.sidebar.text_input("Buscar (comercio o documento)", "")
if buscar_txt.strip():
    q = buscar_txt.strip()
    m1 = df_f["nbr_comercio"].fillna("").astype(str).str.contains(q, case=False, regex=False)
    m2 = df_f["num_documento"].fillna("").astype(str).str.contains(q, case=False, regex=False)
    df_f = df_f[m1 | m2].copy()

# Controles rutas
st.sidebar.markdown("### 🧠 Rutas")
n_clusters = st.sidebar.slider("Número de clusters", min_value=1, max_value=12, value=4, step=1)
speed_kmh = st.sidebar.slider("Velocidad promedio (km/h)", min_value=5, max_value=40, value=int(DEFAULT_SPEED_KMH), step=1)

# performance control
max_points = st.sidebar.slider("Máx. puntos (para performance)", min_value=50, max_value=400, value=250, step=25)

generate = st.sidebar.button("🚀 Generar rutas", use_container_width=True)

# Resumen base
colA, colB, colC, colD = st.columns(4)
colA.metric("KAM", kam_sel)
colB.metric("Puntos (filtrados)", len(df_f))
colC.metric("Etiquetas distintas", df_f["etiqueta_final"].nunique())
colD.metric("Distritos distintos", df_f["nbr_distrito_hexagono"].nunique())

# Tabla preview
with st.expander("Ver tabla (preview)"):
    show_cols = [
        "num_documento", "nbr_comercio", "nbr_direccion_comercio",
        "Departamento", "nbr_provincia_hexagono", "nbr_distrito_hexagono",
        "etiqueta_final", "gpvabo", "ntrx", "grp_economico"
    ]
    show_cols = [c for c in show_cols if c in df_f.columns]
    st.dataframe(df_f[show_cols].head(200), use_container_width=True)

# Recorte por performance
if len(df_f) > max_points:
    st.warning(f"Hay {len(df_f)} puntos. Para rendimiento, se usarán {max_points} (muestra aleatoria).")
    df_work = df_f.sample(max_points, random_state=42).copy()
else:
    df_work = df_f.copy()

if df_work.empty:
    st.warning("No hay puntos con estos filtros.")
    st.stop()

# Centro mapa
center_lat = float(df_work["num_latitud"].mean())
center_lon = float(df_work["num_longitud"].mean())

m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=MAP_TILE)

# Markers
cluster = MarkerCluster(name="Comercios").add_to(m)
for _, r in df_work.iterrows():
    folium.Marker(
        location=[r["num_latitud"], r["num_longitud"]],
        popup=folium.Popup(make_popup(r), max_width=450),
        tooltip=f"{r.get('nbr_comercio','')}"
    ).add_to(cluster)

route_metrics = []

if generate:
    # KMeans clustering
    coords = df_work[["num_latitud", "num_longitud"]].to_numpy()
    if len(coords) < n_clusters:
        n_clusters_eff = max(1, len(coords))
    else:
        n_clusters_eff = n_clusters

    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=42, n_init="auto")
    df_work["cluster_id"] = kmeans.fit_predict(coords)

    # Para cada cluster: TSP
    for cid in sorted(df_work["cluster_id"].unique()):
        df_c = df_work[df_work["cluster_id"] == cid].copy()

        # si cluster demasiado grande, reducimos (soft)
        if len(df_c) > MAX_POINTS_PER_CLUSTER_SOFT:
            df_c = df_c.sample(MAX_POINTS_PER_CLUSTER_SOFT, random_state=cid).copy()

        coords_c = df_c[["num_latitud", "num_longitud"]].to_numpy()

        order = solve_tsp_ortools(coords_c)
        df_route = df_c.iloc[order].reset_index(drop=True)

        # métricas
        dist_opt = route_distance_km(df_route)
        time_opt = compute_time_minutes(dist_opt, speed_kmh)

        dist_base = approx_baseline_distance_km(df_c)
        time_base = compute_time_minutes(dist_base, speed_kmh)

        savings_km = dist_base - dist_opt
        savings_min = (time_base - time_opt) if (time_base is not None and time_opt is not None) else None

        route_metrics.append({
            "cluster": int(cid),
            "puntos": int(len(df_route)),
            "dist_opt_km": round(dist_opt, 2),
            "tiempo_opt_min": round(time_opt, 1) if time_opt is not None else None,
            "dist_base_km": round(dist_base, 2),
            "tiempo_base_min": round(time_base, 1) if time_base is not None else None,
            "ahorro_km": round(savings_km, 2),
            "ahorro_min": round(savings_min, 1) if savings_min is not None else None,
        })

        add_route_polyline(m, df_route, name=f"Ruta Cluster {cid}")

    folium.LayerControl().add_to(m)

# Render map
st.subheader("Mapa")
st_folium(m, width=None, height=650)

# Métricas
if generate and route_metrics:
    st.subheader("Métricas de valor")
    met = pd.DataFrame(route_metrics).sort_values("cluster")

    total_opt = met["dist_opt_km"].sum()
    total_base = met["dist_base_km"].sum()
    total_save = met["ahorro_km"].sum()

    total_opt_min = met["tiempo_opt_min"].sum() if met["tiempo_opt_min"].notna().all() else None
    total_base_min = met["tiempo_base_min"].sum() if met["tiempo_base_min"].notna().all() else None
    total_save_min = met["ahorro_min"].sum() if met["ahorro_min"].notna().all() else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Distancia optimizada (km)", f"{total_opt:.2f}")
    c2.metric("Distancia baseline (km)", f"{total_base:.2f}")
    c3.metric("Ahorro estimado (km)", f"{total_save:.2f}")

    c4, c5, c6 = st.columns(3)
    if total_opt_min is not None and total_base_min is not None and total_save_min is not None:
        c4.metric("Tiempo optimizado (min)", f"{total_opt_min:.1f}")
        c5.metric("Tiempo baseline (min)", f"{total_base_min:.1f}")
        c6.metric("Ahorro estimado (min)", f"{total_save_min:.1f}")
    else:
        c4.info("Tiempo no disponible (revisa velocidad o nulos).")
        c5.info("")
        c6.info("")

    st.dataframe(met, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("GeoKAM • Optimización TSP por clusters • Marzo 2026")
