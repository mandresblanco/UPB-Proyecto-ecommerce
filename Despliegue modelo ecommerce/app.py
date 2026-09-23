import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Cargar modelo y columnas
# ---------------------------
@st.cache_resource
def cargar_modelo():
    with open('modelo_xgb.pkl', 'rb') as archivo:
        modelo, columnas = pickle.load(archivo)
    return modelo, columnas

modelo, columnas = cargar_modelo()

# ---------------------------
# Cargar muestra de referencia
# ---------------------------
@st.cache_data
def cargar_muestra():
    return pd.read_csv('data/df_muestra.csv')

df_muestra = cargar_muestra()

st.title("Predicción de días de entrega")

# ---------------------------
# 1. Elegir caso de referencia (para variables históricas)
# ---------------------------
st.subheader("1. Caso de referencia (contexto histórico)")
indice_referencia = st.selectbox("Selecciona un registro de referencia", df_muestra.index)
caso_referencia = df_muestra.loc[indice_referencia]

columnas_historicas = [
    'facturacion_promedio_t1', 'facturacion_mediana_t1', 'facturacion_std_t1',
    'facturacion_total_7d_avg', 'dias_entrega_promedio_7d_avg',
    'dias_entrega_promedio_t2', 'dias_entrega_promedio_prev_month',
    'distancia_total_t1', 'distancia_total_prev_month'
]

with st.expander("Ver valores históricos usados"):
    st.write(caso_referencia[columnas_historicas])

# ---------------------------
# 2. Variables del pedido (editables)
# ---------------------------
st.subheader("2. Datos del pedido")

# Calculamos rangos por estado (a partir de la muestra histórica)
columnas_numericas = ['total_payment_value', 'total_freight', 'total_weight_g',
                       'total_volume_cm3', 'distancia_km']

rangos_por_estado = df_muestra.groupby('customer_state')[columnas_numericas].agg(['min', 'max'])

customer_state = st.selectbox("Estado del cliente", ['RJ', 'SP'])

# Obtenemos los límites correspondientes a ese estado
limites = rangos_por_estado.loc[customer_state]

principal_category_by_price = st.selectbox(
    "Categoría principal",
    ["b'brinquedos'", "b'moveis_decoracao'", "b'cama_mesa_banho'", "b'beleza_saude'",
     "b'relogios_presentes'", "b'esporte_lazer'", "b'informatica_acessorios'", "b'utilidades_domesticas'"]
)

principal_seller_state_by_price = st.selectbox("Estado principal del vendedor", ['SP', 'RJ', 'PR', 'MG'])

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

distancia_km = st.number_input(
    "Distancia (km)",
    min_value=float(limites[('distancia_km', 'min')]),
    max_value=float(limites[('distancia_km', 'max')]),
    value=float(limites[('distancia_km', 'min')])
)

purchase_year = st.selectbox("Año de compra", [2017, 2018])

es_post_domingo_festivo = st.checkbox("¿Es post domingo/festivo?")
es_diciembre = st.checkbox("¿Es diciembre?")
es_fin_de_semana_extendido = st.checkbox("¿Es fin de semana extendido?")

# ---------------------------
# 3. Armar el registro completo
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

    # Aplicamos el mismo encoding que en el entrenamiento
    df_entrada_encoded = pd.get_dummies(
        df_entrada,
        columns=['customer_state', 'principal_category_by_price', 'principal_seller_state_by_price']
    )

    # Alineamos con las columnas del modelo
    X_input = df_entrada_encoded.reindex(columns=columnas, fill_value=0)

    prediccion = modelo.predict(X_input)[0]
    prediccion_mas_3 = prediccion + 3

    st.success(f"Predicción de días de entrega: {prediccion:.2f}")
    st.info(f"Predicción + margen (3 días): {prediccion_mas_3:.2f}")

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(X_input)
        