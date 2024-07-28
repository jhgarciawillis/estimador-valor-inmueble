import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn import __version__ as sklearn_version
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import folium
from streamlit_folium import folium_static
import re
import logging
from datetime import datetime
from cryptography.fernet import Fernet
import json
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Configurar registro
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MongoDB connection
@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(st.secrets["mongo"]["connection_string"], server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        return client
    except Exception as e:
        logger.error(f"Error de configuración de MongoDB: {str(e)}")
        return None

# Encryption setup
def get_or_create_key():
    try:
        return Fernet(st.secrets["encryption"]["key"].encode())
    except KeyError:
        logger.warning("Encryption key not found in secrets. Generating a new one.")
        key = Fernet.generate_key()
        return Fernet(key)

fernet = get_or_create_key()

def encrypt_data(data):
    return fernet.encrypt(json.dumps(data).encode()).decode()

def decrypt_data(encrypted_data):
    return json.loads(fernet.decrypt(encrypted_data.encode()).decode())

def store_data(data):
    encrypted_data = encrypt_data(data)
    client = get_mongo_client()
    if client:
        try:
            db = client.property_database
            collection = db.property_data
            collection.insert_one({"timestamp": datetime.now().isoformat(), "data": encrypted_data})
            logger.debug("Data stored successfully")
        except Exception as e:
            logger.error(f"Error al almacenar datos: {str(e)}")
        finally:
            client.close()
    else:
        logger.error("No se pudo conectar a MongoDB")

def retrieve_data():
    client = get_mongo_client()
    if client:
        try:
            db = client.property_database
            collection = db.property_data
            for doc in collection.find():
                yield doc['timestamp'], decrypt_data(doc['data'])
        except Exception as e:
            logger.error(f"Error al recuperar datos: {str(e)}")
        finally:
            client.close()
    else:
        logger.error("No se pudo conectar a MongoDB")

# Configuración de la página
st.set_page_config(page_title="Estimador de Valor de Propiedades", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    body {
        color: #FFFFFF;
        background-color: #1E1E1E;
        font-family: 'Roboto', sans-serif;
    }
    .widget-container {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        color: #FFFFFF;
        background-color: #1E1E1E;
        border: 1px solid #4A4A4A;
        border-radius: 4px;
        padding: 8px 10px;
        width: 100%;
    }
    .stButton > button {
        width: 100%;
        background-color: #4A90E2;
        color: white;
    }
    h1 {
        color: #4A90E2;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .etiqueta-entrada {
        font-size: 14px;
        color: #B0B0B0;
        margin-bottom: 5px;
    }
    .folium-map {
        width: 100%;
        height: 300px;
        max-width: 600px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelos y herramientas
@st.cache_resource
def cargar_modelos(tipo_propiedad):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    prefijo = "renta_" if tipo_propiedad == "Departamento" else ""
    logger.debug(f"Cargando modelos para {tipo_propiedad} con prefijo: {prefijo}")
    modelos = {}
    modelos_requeridos = {
        'modelo': 'bosque_aleatorio.joblib',
        'escalador': 'escalador.joblib',
        'imputador': 'imputador.joblib',
        'agrupamiento': 'agrupamiento.joblib'
    }
    try:
        for nombre_modelo, nombre_archivo in modelos_requeridos.items():
            ruta_archivo = os.path.join(directorio_actual, f"{prefijo}{nombre_archivo}")
            if os.path.exists(ruta_archivo):
                modelos[nombre_modelo] = joblib.load(ruta_archivo)
            else:
                logger.error(f"Archivo de modelo no encontrado: {ruta_archivo}")
                raise FileNotFoundError(f"Archivo de modelo no encontrado: {ruta_archivo}")
        logger.debug("Modelos cargados exitosamente")
    except Exception as e:
        logger.error(f"Error al cargar los modelos: {str(e)}")
        st.error(f"Error al cargar los modelos: {str(e)}. Por favor contacte al soporte.")
    return modelos

# Inicializar geocodificador
geolocalizador = Nominatim(user_agent="aplicacion_propiedades")

def geocodificar_direccion(direccion):
    logger.debug(f"Intentando geocodificar dirección: {direccion}")
    try:
        ubicacion = geolocalizador.geocode(direccion)
        if ubicacion:
            logger.debug(f"Geocodificación exitosa: {ubicacion.latitude}, {ubicacion.longitude}")
            return ubicacion.latitude, ubicacion.longitude, ubicacion
    except (GeocoderTimedOut, GeocoderUnavailable):
        logger.warning("Servicio de geocodificación no disponible")
    return None, None, None

def obtener_sugerencias_direccion(consulta):
    logger.debug(f"Obteniendo sugerencias para: {consulta}")
    try:
        ubicaciones = geolocalizador.geocode(consulta + ", México", exactly_one=False, limit=5)
        if ubicaciones:
            logger.debug(f"Sugerencias obtenidas: {len(ubicaciones)}")
            return [ubicacion.address for ubicacion in ubicaciones]
    except (GeocoderTimedOut, GeocoderUnavailable):
        logger.warning("Servicio de geocodificación no disponible")
    return []

def agregar_caracteristica_grupo(latitud, longitud, modelos):
    logger.debug(f"Agregando característica de grupo para: {latitud}, {longitud}")
    try:
        grupo = modelos['agrupamiento'].predict(pd.DataFrame({'Latitud': [latitud], 'Longitud': [longitud]}))[0]
        logger.debug(f"Grupo obtenido: {grupo}")
        return grupo
    except Exception as e:
        logger.error(f"Error al agregar característica de grupo: {str(e)}")
        return None

def preprocesar_datos(latitud, longitud, terreno, construccion, habitaciones, banos, modelos):
    logger.debug("Preprocesando datos")
    try:
        grupo_ubicacion = agregar_caracteristica_grupo(latitud, longitud, modelos)
        
        datos_entrada = pd.DataFrame({
            'Terreno': [terreno],
            'Construccion': [construccion],
            'Habitaciones': [habitaciones],
            'Banos': [banos],
            'GrupoUbicacion': [grupo_ubicacion],
        })
        
        logger.debug(f"Datos de entrada: {datos_entrada}")
        
        datos_imputados = modelos['imputador'].transform(datos_entrada)
        datos_escalados = modelos['escalador'].transform(datos_imputados)
        logger.debug("Datos preprocesados exitosamente")
        return pd.DataFrame(datos_escalados, columns=datos_entrada.columns)
    except Exception as e:
        logger.error(f"Error al preprocesar datos: {str(e)}")
        return None

def predecir_precio(datos_procesados, modelos):
    logger.debug("Prediciendo precio")
    try:
        precio_bruto = modelos['modelo'].predict(datos_procesados)[0]
        precio_ajustado = precio_bruto
        precio_redondeado = math.floor((precio_ajustado * .77) / 1000) * 1000

        factor_escala_bajo = math.exp(-0.05)
        factor_escala_alto = math.exp(0.01 * math.log(precio_redondeado / 1000 + 1))

        rango_precio_min = max(0, math.floor((precio_redondeado * factor_escala_bajo) / 1000) * 1000)
        rango_precio_max = math.ceil((precio_redondeado * factor_escala_alto) / 1000) * 1000

        rango_precio_min = min(rango_precio_min, precio_redondeado)
        rango_precio_max = max(rango_precio_max, precio_redondeado)

        logger.debug(f"Precio predicho: {precio_redondeado}, Rango: {rango_precio_min} - {rango_precio_max}")
        return precio_redondeado, rango_precio_min, rango_precio_max
    except Exception as e:
        logger.error(f"Error al predecir el precio: {str(e)}")
        return None, None, None

def validar_correo(correo):
    patron = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(patron, correo) is not None

def validar_telefono(telefono):
    patron = r'^\+?[1-9]\d{1,14}$'
    return re.match(patron, telefono) is not None

# Interfaz de usuario
st.title("Estimador de Valor de Propiedades")

# Contenedor principal
with st.container():
    st.markdown('<div class="widget-container">', unsafe_allow_html=True)

    # Tipo de propiedad
    st.markdown('<div class="etiqueta-entrada">Tipo de Propiedad</div>', unsafe_allow_html=True)
    tipo_propiedad = st.selectbox("", ["Casa", "Departamento"], key="tipo_propiedad", help="Seleccione el tipo de propiedad")

    # Cargar modelos basados en el tipo de propiedad
    modelos = cargar_modelos(tipo_propiedad)

    # Dirección de la propiedad
    st.markdown('<div class="etiqueta-entrada">Dirección de la Propiedad</div>', unsafe_allow_html=True)
    entrada_direccion = st.text_input("", key="entrada_direccion", help="Ingrese la dirección completa de la propiedad")

    latitud, longitud = None, None

    if entrada_direccion:
        logger.debug(f"Dirección ingresada: {entrada_direccion}")
        sugerencias = obtener_sugerencias_direccion(entrada_direccion)
        if sugerencias:
            st.markdown('<div class="etiqueta-entrada">Dirección Sugerida</div>', unsafe_allow_html=True)
            direccion_seleccionada = st.selectbox("", sugerencias, index=0, key="direccion_sugerida", help="Seleccione la dirección correcta de las sugerencias")
            if direccion_seleccionada:
                latitud, longitud, ubicacion = geocodificar_direccion(direccion_seleccionada)
                if latitud and longitud:
                    st.success(f"Ubicación encontrada: {direccion_seleccionada}")
                    logger.debug(f"Ubicación encontrada: Lat {latitud}, Lon {longitud}")
                    
                    # Crear y mostrar el mapa responsivo
                    m = folium.Map(location=[latitud, longitud], zoom_start=15, tiles="CartoDB dark_matter")
                    folium.Marker([latitud, longitud], popup=direccion_seleccionada).add_to(m)
                    folium_static(m, width=300, height=200)
                else:
                    logger.warning("No se pudo geocodificar la dirección seleccionada")
                    st.error("No se pudo geocodificar la dirección seleccionada.")

    # Entradas para detalles de la propiedad
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="etiqueta-entrada">Terreno (m²)</div>', unsafe_allow_html=True)
        terreno = st.number_input("", min_value=0, step=1, format="%d", key="terreno", help="Área total del terreno en metros cuadrados")

        st.markdown('<div class="etiqueta-entrada">Construcción (m²)</div>', unsafe_allow_html=True)
        construccion = st.number_input("", min_value=0, step=1, format="%d", key="construccion", help="Área construida en metros cuadrados")

    with col2:
        st.markdown('<div class="etiqueta-entrada">Habitaciones</div>', unsafe_allow_html=True)
        habitaciones = st.number_input("", min_value=0, step=1, format="%d", key="habitaciones", help="Número de habitaciones")

        st.markdown('<div class="etiqueta-entrada">Baños</div>', unsafe_allow_html=True)
        banos = st.number_input("", min_value=0.0, step=0.5, format="%.1f", key="banos", help="Número de baños (use decimales para medios baños)")

    # Campos de correo electrónico y teléfono
    st.markdown('<div class="etiqueta-entrada">Correo Electrónico</div>', unsafe_allow_html=True)
    correo = st.text_input("", key="correo", help="Ingrese su dirección de correo electrónico")

    st.markdown('<div class="etiqueta-entrada">Teléfono</div>', unsafe_allow_html=True)
    telefono = st.text_input("", key="telefono", help="Ingrese su número de teléfono")

    # Botón de cálculo
    texto_boton = "Estimar Valor" if tipo_propiedad == "Casa" else "Estimar Renta"
    if st.button(texto_boton, key="boton_calcular"):
        logger.debug(f"Botón presionado: {texto_boton}")
        if not validar_correo(correo):
            logger.warning(f"Correo electrónico inválido: {correo}")
            st.error("Por favor, ingrese una dirección de correo electrónico válida.")
        elif not validar_telefono(telefono):
            logger.warning(f"Teléfono inválido: {telefono}")
            st.error("Por favor, ingrese un número de teléfono válido.")
        elif latitud and longitud and terreno and construccion and habitaciones and banos:
            logger.debug("Todos los campos requeridos están completos")
            
            datos_procesados = preprocesar_datos(latitud, longitud, terreno, construccion, habitaciones, banos, modelos)
            if datos_procesados is not None:
                precio, precio_min, precio_max = predecir_precio(datos_procesados, modelos)
                if precio is not None:
                    if tipo_propiedad == "Casa":
                        st.markdown(f"<h3 style='color: #50E3C2;'>Valor Estimado: ${precio:,}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #B0B0B0;'>Rango de Precio Estimado: ${precio_min:,} - ${precio_max:,}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color: #50E3C2;'>Renta Mensual Estimada: ${precio:,}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #B0B0B0;'>Rango de Renta Estimado: ${precio_min:,} - ${precio_max:,}</p>", unsafe_allow_html=True)

                    # Prepare data to store
                    data_to_store = {
                        "tipo_propiedad": tipo_propiedad,
                        "direccion": direccion_seleccionada,
                        "latitud": latitud,
                        "longitud": longitud,
                        "terreno": terreno,
                        "construccion": construccion,
                        "habitaciones": habitaciones,
                        "banos": banos,
                        "correo": correo,
                        "telefono": telefono,
                        "precio_estimado": precio,
                        "precio_min": precio_min,
                        "precio_max": precio_max
                    }

                    # Store data
                    try:
                        store_data(data_to_store)
                        logger.info("Datos almacenados exitosamente")
                    except Exception as e:
                        logger.error(f"Error al almacenar datos: {str(e)}")
                        # No mostramos el error al usuario para mantener la experiencia sin problemas

                    # Gráfico de barras para visualizar el rango de precios
                    fig = go.Figure(go.Bar(
                        x=['Precio Mínimo', 'Precio Estimado', 'Precio Máximo'],
                        y=[precio_min, precio, precio_max],
                        text=[f'${precio_min:,}', f'${precio:,}', f'${precio_max:,}'],
                        textposition='auto',
                        marker_color=['#4A90E2', '#50E3C2', '#4A90E2']
                    ))
                    fig.update_layout(
                        title_text='Rango de Precio Estimado',
                        font=dict(family="Roboto", color="#FFFFFF"),
                        paper_bgcolor="#2D2D2D",
                        plot_bgcolor="#3D3D3D",
                        xaxis=dict(tickfont=dict(color="#FFFFFF")),
                        yaxis=dict(tickfont=dict(color="#FFFFFF"))
                    )
                    st.plotly_chart(fig)
                else:
                    logger.error("Error predicting the price")
                    st.error("Hubo un error al calcular el precio. Por favor, inténtelo de nuevo.")
            else:
                logger.error("Error preprocessing the data")
                st.error("Hubo un error al procesar los datos. Por favor, inténtelo de nuevo.")
        else:
            logger.warning("Incomplete fields")
            st.error("Por favor, asegúrese de ingresar una dirección válida y completar todos los campos.")

    st.markdown('</div>', unsafe_allow_html=True)

# Instrucciones de uso
with st.expander("Instrucciones de Uso"):
    st.markdown("""
    1. Seleccione el tipo de propiedad: Casa (en venta) o Departamento (en alquiler).
    2. Ingrese la dirección completa de la propiedad y seleccione la sugerencia correcta.
    3. Verifique la ubicación en el mapa mostrado.
    4. Proporcione el área del terreno y el área construida en metros cuadrados.
    5. Indique el número de habitaciones y baños (puede usar decimales para baños, por ejemplo, 2.5 para dos baños completos y un medio baño).
    6. Ingrese su correo electrónico y número de teléfono en los campos correspondientes.
    7. Haga clic en "Estimar Valor" o "Estimar Alquiler" para obtener la estimación.

    Nota: Asegúrese de que todos los campos estén completos para obtener una estimación precisa.
    """)

# Pie de página
st.markdown("---")
st.markdown("© 2024 Estimador de Valor de Propiedad. Todos los derechos reservados.")
