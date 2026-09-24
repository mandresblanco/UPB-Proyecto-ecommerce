import streamlit as st
import pandas as pd
import pickle
import math

st.set_page_config(page_title="Predicción de entrega", page_icon="📦", layout="wide")

# ---------------------------
# Estilos (paleta inspirada en el sitio real de Olist)
# ---------------------------
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #2F63F6;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #1E4FD6;
        color: white;
    }
    [data-testid="stMetricValue"] {
        color: #2F63F6;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Cargar modelo y columnas
# ---------------------------
@st.cache_resource
def cargar_modelo():
    with open('modelo_xgb.pkl', 'rb') as archivo:
        modelo, columnas = pickle.load(archivo)
    return modelo, columnas

modelo, columnas = cargar_modelo()

@st.cache_data
def cargar_muestra():
    return pd.read_csv('data/df_muestra.csv')

df_muestra = cargar_muestra()

# ---------------------------
# Encabezado
# ---------------------------
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    st.image("logo.png", width=120)
with col_titulo:
    st.title("Predicción de días de entrega")

st.divider()

# ---------------------------
# Categoría del producto
# ---------------------------
principal_category_by_price = st.selectbox(
    "Categoría del producto",
    ["b'brinquedos'", "b'moveis_decoracao'", "b'cama_mesa_banho'", "b'beleza_saude'",
     "b'relogios_presentes'", "b'esporte_lazer'", "b'informatica_acessorios'", "b'utilidades_domesticas'"]
)

st.write("")

# ---------------------------
# Ubicación (define primero el par de estados)
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

# ---------------------------
# Rangos por estado del cliente
# ---------------------------
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
# Contexto de fecha + caso de referencia
# ---------------------------
col4, col5 = st.columns(2)

with col4:
    with st.container(border=True):
        st.subheader("Contexto de fecha")
        es_post_domingo_festivo = st.checkbox("¿Es post domingo/festivo?")
        es_diciembre = st.checkbox("¿Es diciembre?")
        es_fin_de_semana_extendido = st.checkbox("¿Es fin de semana extendido?")

with col5:
    with st.container(border=True):
        st.subheader("Caso de referencia histórico")
        indice_referencia = st.selectbox("Registro de referencia", df_muestra.index)
        caso_referencia = df_muestra.loc[indice_referencia]

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
    dias_pred = math.ceil(prediccion)
    dias_pred_margen = dias_pred + 3

    st.divider()
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Predicción de días de entrega", f"{dias_pred} días")
    with col_r2:
        st.metric("Predicción + margen (3 días)", f"{dias_pred_margen} días")

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(X_input)