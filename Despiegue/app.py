import streamlit as st
import pandas as pd
import pickle
import math
import random
import glob

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Predicción de entrega", page_icon="📦", layout="wide")

# ---------------------------
# Estilos: fondo, glassmorphism, alineación
# ---------------------------
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 30%, #1E3A9E 0%, #0A0F2C 55%, #05081A 100%);
    }

    h1, h2, h3, label, p, span, .stMarkdown {
        color: #F5F7FF !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(76, 201, 240, 0.30) !important;
        border-radius: 16px !important;
        box-shadow: 0 0 16px rgba(47, 99, 246, 0.18);
    }

    [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    [data-testid="stImage"] img {
        margin: 0 auto !important;
        display: block !important;
        border-radius: 12px;
        background: #FFFFFF;
        padding: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    }

    [data-testid="stCaptionContainer"] {
        text-align: center;
        width: 100%;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #1E4FD6, #2F63F6);
        color: white;
        border-radius: 999px;
        padding: 0.55rem 1.5rem;
        border: 1px solid rgba(76, 201, 240, 0.55);
        font-weight: 600;
        width: 100%;
        box-shadow: 0 0 12px rgba(76, 201, 240, 0.4);
        transition: 0.2s ease-in-out;
        display: block;
        margin: 0 auto;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 20px rgba(76, 201, 240, 0.7);
    }

    div[data-baseweb="select"] > div, input, textarea {
        background-color: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(76, 201, 240, 0.25) !important;
        color: #FFFFFF !important;
    }

    [data-testid="stMetricValue"] {
        color: #4CC9F0 !important;
        font-size: 3rem;
    }

    .header-panel-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 110px;
    }

    .header-panel {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(76, 201, 240, 0.30);
        border-radius: 16px;
        box-shadow: 0 0 16px rgba(47, 99, 246, 0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        height: 110px;
    }
    .header-panel-logo img {
        height: 90px;
        width: 90px;
        object-fit: cover;
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(76, 201, 240, 0.4);
    }
    .header-panel h1 {
        margin: 0 !important;
        font-size: 1.8rem !important;
    }

    .glow-corner-1 {
        position: fixed;
        top: -120px; left: -120px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(76,201,240,0.12) 0%, transparent 70%);
        z-index: -1;
    }
    .glow-corner-2 {
        position: fixed;
        bottom: -120px; right: -120px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(255, 105, 180, 0.08) 0%, transparent 70%);
        z-index: -1;
    }

    /* Centrado real de las tarjetas de categoría: se fuerza a que cada
       elemento (imagen, texto, botón) se encoja a su tamaño y se centre
       con márgenes automáticos, en vez de ocupar el 100% del ancho */


    .st-key-categorias div[data-testid="stElementContainer"]:has(div[data-testid="stImage"]) {
        width: fit-content !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    .st-key-categorias div[data-testid="stElementContainer"][width="fit-content"] {
        margin-left: auto !important;
        margin-right: auto !important;
    }   
    </style>

    <div class="glow-corner-1"></div>
    <div class="glow-corner-2"></div>
""", unsafe_allow_html=True)

# ---------------------------
# Cargar modelo de regresión
# ---------------------------
@st.cache_resource
def cargar_modelo():
    with open(BASE_DIR / 'modelo_xgb.pkl', 'rb') as archivo:
        modelo, columnas = pickle.load(archivo)
    return modelo, columnas

modelo, columnas = cargar_modelo()

# ---------------------------
# Cargar modelo de clustering
# ---------------------------
@st.cache_resource
def cargar_cluster():
    with open(BASE_DIR / 'cluster_pipe.pkl', 'rb') as archivo:
        return pickle.load(archivo)   # dict: pipeline, columnas, nombres_clusters

art_cluster = cargar_cluster()

DESCRIPCIONES_CLUSTER = {
    'Local y ligero': 'Paquete pequeño y cercano, típicamente dentro del mismo estado',
    'Voluminoso y pesado': 'Paquete grande o pesado, con distancia intermedia.',
    'Larga distancia': 'Envío a otro estado, con el mayor costo de flete.',
}

# ---------------------------
# Cargar muestra
# ---------------------------
@st.cache_data
def cargar_muestra():
    return pd.read_csv(BASE_DIR / 'data' / 'df_muestra.csv')

df_muestra = cargar_muestra()

CATEGORIAS = ["b'brinquedos'", "b'moveis_decoracao'", "b'cama_mesa_banho'", "b'beleza_saude'",
              "b'relogios_presentes'", "b'esporte_lazer'", "b'informatica_acessorios'", "b'utilidades_domesticas'"]

NOMBRES_LEGIBLES = {
    "b'brinquedos'": "Juguetes",
    "b'moveis_decoracao'": "Muebles y Decoración",
    "b'cama_mesa_banho'": "Cama, Mesa y Baño",
    "b'beleza_saude'": "Belleza y Salud",
    "b'relogios_presentes'": "Relojes y Regalos",
    "b'esporte_lazer'": "Deporte y Ocio",
    "b'informatica_acessorios'": "Informática",
    "b'utilidades_domesticas'": "Utilidades del Hogar"
}

def encontrar_imagen(numero):
    coincidencias = glob.glob(str(BASE_DIR / f"{numero}.*.png")) or glob.glob(str(BASE_DIR / f"{numero} *.png"))
    return coincidencias[0] if coincidencias else None

IMAGENES_CATEGORIA = {
    "b'brinquedos'": encontrar_imagen(1),
    "b'moveis_decoracao'": encontrar_imagen(2),
    "b'cama_mesa_banho'": encontrar_imagen(3),
    "b'beleza_saude'": encontrar_imagen(4),
    "b'relogios_presentes'": encontrar_imagen(5),
    "b'esporte_lazer'": encontrar_imagen(6),
    "b'informatica_acessorios'": encontrar_imagen(7),
    "b'utilidades_domesticas'": encontrar_imagen(8)
}

# ---------------------------
# Inicialización de valores aleatorios (solo una vez por sesión)
# ---------------------------
if 'inicializado' not in st.session_state:
    st.session_state['indice_referencia'] = random.choice(df_muestra.index.tolist())
    st.session_state['categoria_seleccionada'] = random.choice(CATEGORIAS)
    st.session_state['es_post_domingo_festivo'] = random.choice([True, False])
    st.session_state['es_diciembre'] = random.choice([True, False])
    st.session_state['es_fin_de_semana_extendido'] = random.choice([True, False])
    st.session_state['inicializado'] = True

caso_referencia = df_muestra.loc[st.session_state['indice_referencia']]

# ---------------------------
# Encabezado: dos paneles de vidrio integrados
# ---------------------------
import base64

def imagen_a_base64(ruta):
    with open(ruta, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = imagen_a_base64(BASE_DIR / "logo.png")

col_logo, col_titulo = st.columns([1.3, 4])
with col_logo:
    st.markdown(f"""
        <div class="header-panel-logo">
            <img src="data:image/png;base64,{logo_b64}">
        </div>
    """, unsafe_allow_html=True)
with col_titulo:
    st.markdown("""
        <div class="header-panel">
            <h1>Predicción de días de entrega</h1>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------
# Caso de referencia (minimizado)
# ---------------------------
with st.expander("📋 Ver contexto histórico usado (avanzado)"):
    columnas_historicas = [
        'facturacion_promedio_t1', 'facturacion_mediana_t1', 'facturacion_std_t1',
        'facturacion_total_7d_avg', 'dias_entrega_promedio_7d_avg',
        'dias_entrega_promedio_t2', 'dias_entrega_promedio_prev_month',
        'distancia_total_t1', 'distancia_total_prev_month'
    ]
    st.write(caso_referencia[columnas_historicas])
    if st.button("🔀 Elegir otro caso de referencia aleatorio"):
        st.session_state['indice_referencia'] = random.choice(df_muestra.index.tolist())
        st.rerun()

st.write("")

# ---------------------------
# Grid de categorías (alineación centrada)
# ---------------------------
st.subheader("Categoría del producto")

st.success(f"🛒 Producto seleccionado: **{NOMBRES_LEGIBLES[st.session_state['categoria_seleccionada']]}**")

with st.container(key="categorias"):
    cols_categorias = st.columns(4)
    for i, cat in enumerate(CATEGORIAS):
        with cols_categorias[i % 4]:
            seleccionada = (st.session_state['categoria_seleccionada'] == cat)
            with st.container(border=True):
                if IMAGENES_CATEGORIA[cat]:
                    st.image(IMAGENES_CATEGORIA[cat], width=90)
                st.caption(NOMBRES_LEGIBLES[cat])
                etiqueta_boton = "✅ Seleccionado" if seleccionada else "Comprar"
                if st.button(etiqueta_boton, key=f"cat_{i}"):
                    st.session_state['categoria_seleccionada'] = cat
                    st.rerun()

principal_category_by_price = st.session_state['categoria_seleccionada']

st.write("")

# ---------------------------
# Ubicación / Pedido / Paquete
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("Ubicación")
        customer_state = st.selectbox("Estado del cliente", ['RJ', 'SP'])
        principal_seller_state_by_price = st.selectbox("Estado del vendedor", ['SP', 'RJ', 'PR', 'MG'])

        subset_par = df_muestra[
            (df_muestra['customer_state'] == customer_state) &
            (df_muestra['principal_seller_state_by_price'] == principal_seller_state_by_price)
        ]

        if len(subset_par) > 0:
            dist_min = float(subset_par['distancia_km'].min())
            dist_max = float(subset_par['distancia_km'].max())
        else:
            dist_min = float(df_muestra['distancia_km'].min())
            dist_max = float(df_muestra['distancia_km'].max())
            st.caption("⚠️ Sin historial exacto para esta combinación; se usa el rango general.")

        distancia_km = st.number_input(
            "Distancia (km)",
            min_value=dist_min,
            max_value=dist_max,
            value=dist_min
        )

columnas_numericas = ['total_payment_value', 'total_freight', 'total_weight_g', 'total_volume_cm3']
rangos_por_estado = df_muestra.groupby('customer_state')[columnas_numericas].agg(['min', 'max'])
limites = rangos_por_estado.loc[customer_state]

with col2:
    with st.container(border=True):
        st.subheader("Pedido")
        total_payment_value = st.number_input(
            "Valor total del pago",
            min_value=float(limites[('total_payment_value', 'min')]),
            max_value=float(limites[('total_payment_value', 'max')]),
            value=float(limites[('total_payment_value', 'min')])
        )
        total_freight = st.number_input(
            "Flete total",
            min_value=float(limites[('total_freight', 'min')]),
            max_value=float(limites[('total_freight', 'max')]),
            value=float(limites[('total_freight', 'min')])
        )
        purchase_year = st.selectbox("Año de compra", [2017, 2018])

with col3:
    with st.container(border=True):
        st.subheader("Paquete")
        total_weight_g = st.number_input(
            "Peso total (g)",
            min_value=float(limites[('total_weight_g', 'min')]),
            max_value=float(limites[('total_weight_g', 'max')]),
            value=float(limites[('total_weight_g', 'min')])
        )
        total_volume_cm3 = st.number_input(
            "Volumen total (cm3)",
            min_value=float(limites[('total_volume_cm3', 'min')]),
            max_value=float(limites[('total_volume_cm3', 'max')]),
            value=float(limites[('total_volume_cm3', 'min')])
        )

# ---------------------------
# Contexto de fecha
# ---------------------------
with st.container(border=True):
    st.subheader("Contexto de fecha")
    es_post_domingo_festivo = st.checkbox(
        "¿Es post domingo/festivo?",
        value=st.session_state['es_post_domingo_festivo']
    )
    es_diciembre = st.checkbox(
        "¿Es diciembre?",
        value=st.session_state['es_diciembre']
    )
    es_fin_de_semana_extendido = st.checkbox(
        "¿Es fin de semana extendido?",
        value=st.session_state['es_fin_de_semana_extendido']
    )

st.write("")

# ---------------------------
# Predicción
# ---------------------------
if st.button("Predecir"):
    entrada = {
        'customer_state': customer_state,
        'total_payment_value': total_payment_value,
        'total_freight': total_freight,
        'total_weight_g': total_weight_g,
        'total_volume_cm3': total_volume_cm3,
        'principal_category_by_price': principal_category_by_price,
        'principal_seller_state_by_price': principal_seller_state_by_price,
        'purchase_year': purchase_year,
        'facturacion_promedio_t1': caso_referencia['facturacion_promedio_t1'],
        'facturacion_mediana_t1': caso_referencia['facturacion_mediana_t1'],
        'facturacion_std_t1': caso_referencia['facturacion_std_t1'],
        'facturacion_total_7d_avg': caso_referencia['facturacion_total_7d_avg'],
        'dias_entrega_promedio_7d_avg': caso_referencia['dias_entrega_promedio_7d_avg'],
        'dias_entrega_promedio_t2': caso_referencia['dias_entrega_promedio_t2'],
        'dias_entrega_promedio_prev_month': caso_referencia['dias_entrega_promedio_prev_month'],
        'es_post_domingo_festivo': int(es_post_domingo_festivo),
        'es_diciembre': int(es_diciembre),
        'es_fin_de_semana_extendido': int(es_fin_de_semana_extendido),
        'distancia_km': distancia_km,
        'distancia_total_t1': caso_referencia['distancia_total_t1'],
        'distancia_total_prev_month': caso_referencia['distancia_total_prev_month'],
    }

    df_entrada = pd.DataFrame([entrada])
    df_entrada_encoded = pd.get_dummies(
        df_entrada,
        columns=['customer_state', 'principal_category_by_price', 'principal_seller_state_by_price']
    )
    X_input = df_entrada_encoded.reindex(columns=columnas, fill_value=0)

    prediccion = modelo.predict(X_input)[0]
    dias_estimados = math.ceil(prediccion) + 3

    # ---------------------------
    # Clustering
    # ---------------------------
    fila_cluster = pd.DataFrame([{
        'distancia_km': distancia_km,
        'customer_state_SP': int(customer_state == 'SP'),
        'total_freight': total_freight,
        'total_volume_cm3': total_volume_cm3,
        'total_weight_g': total_weight_g,
    }])[art_cluster['columnas']]

    cluster_id = int(art_cluster['pipeline'].predict(fila_cluster)[0])
    nombre_cluster = art_cluster['nombres_clusters'][cluster_id]

    # ---------------------------
    # Resultados
    # ---------------------------
    st.divider()
    col_dias, col_cluster = st.columns(2)

    with col_dias:
        st.metric("Tu paquete llegará en menos de:", f"{dias_estimados} días")

    with col_cluster:
        st.markdown(f"""
            <div style="font-size:0.875rem; color:#F5F7FF;">Tipo de envío:</div>
            <div style="color:#4CC9F0; font-size:2rem; font-weight:600; line-height:1.3;">
                {nombre_cluster}
            </div>
        """, unsafe_allow_html=True)
        st.caption(DESCRIPCIONES_CLUSTER.get(nombre_cluster, ''))

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(X_input)
        st.write("Datos enviados al clustering:")
        st.dataframe(fila_cluster)