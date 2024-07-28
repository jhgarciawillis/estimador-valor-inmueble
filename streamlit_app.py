import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import folium
from streamlit_folium import folium_static
import re
import logging

# Configurar registro
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(page_title="Estimador de Valor de Propiedades", layout="wide")

# Color palette for real estate agency
PRIMARY_COLOR = "#003366"  # Deep blue
SECONDARY_COLOR = "#FFD700"  # Gold
BACKGROUND_COLOR = "#1E1E1E"  # Dark background
TEXT_COLOR = "#FFFFFF"  # White text
ACCENT_COLOR = "#4CAF50"  # Green

# CSS personalizado
st.markdown(f"""
<style>
    body {{
        color: {TEXT_COLOR};
        background-color: {BACKGROUND_COLOR};
        font-family: 'Arial', sans-serif;
    }}
    .stApp {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }}

    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {{
        color: {TEXT_COLOR};
        background-color: #2D2D2D;
        border: 1px solid #4A4A4A;
        border-radius: 4px;
        padding: 8px 10px;
    }}
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {SECONDARY_COLOR};
        box-shadow: 0 0 0 1px {SECONDARY_COLOR};
    }}
    .stButton > button {{
        width: 100%;
        background-color: {PRIMARY_COLOR};
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY_COLOR};
        color: {PRIMARY_COLOR};
    }}
    h1 {{
        color: {PRIMARY_COLOR};
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }}
    .etiqueta-entrada {{
        font-size: 14px;
        color: {PRIMARYY_COLOR};
        margin-bottom: 5px;
        font-weight: bold;
    }}
    .stExpander {{
        border: 1px solid {SECONDARY_COLOR};
        border-radius: 5px;
    }}
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
        precio_redondeado = math.floor((precio_ajustado * .63) / 1000) * 1000

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

    col1, col2 = st.columns(2)

    with col1:
        # Tipo de propiedad
        st.markdown('<div class="etiqueta-entrada">Tipo de Propiedad</div>', unsafe_allow_html=True)
        tipo_propiedad = st.selectbox("", ["Casa", "Departamento"], key="tipo_propiedad")

        # Cargar modelos basados en el tipo de propiedad
        modelos = cargar_modelos(tipo_propiedad)

        # Dirección de la propiedad
        st.markdown('<div class="etiqueta-entrada">Dirección de la Propiedad</div>', unsafe_allow_html=True)
        entrada_direccion = st.text_input("", key="entrada_direccion", placeholder="Ej., Calle Principal 123, Ciudad de México")

    with col2:
        # Placeholder para mantener el alineamiento
        st.markdown('<div style="height: 76px;"></div>', unsafe_allow_html=True)

        if entrada_direccion:
            logger.debug(f"Dirección ingresada: {entrada_direccion}")
            sugerencias = obtener_sugerencias_direccion(entrada_direccion)
            if sugerencias:
                st.markdown('<div class="etiqueta-entrada">Dirección Sugerida</div>', unsafe_allow_html=True)
                direccion_seleccionada = st.selectbox("", sugerencias, index=0, key="direccion_sugerida")

    # Detalles de la propiedad
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="etiqueta-entrada">Terreno (m²)</div>', unsafe_allow_html=True)
        terreno = st.number_input("", min_value=0, step=1, format="%d", key="terreno")

    with col2:
        st.markdown('<div class="etiqueta-entrada">Construcción (m²)</div>', unsafe_allow_html=True)
        construccion = st.number_input("", min_value=0, step=1, format="%d", key="construccion")

    with col3:
        st.markdown('<div class="etiqueta-entrada">Habitaciones</div>', unsafe_allow_html=True)
        habitaciones = st.number_input("", min_value=0, step=1, format="%d", key="habitaciones")

    with col4:
        st.markdown('<div class="etiqueta-entrada">Baños</div>', unsafe_allow_html=True)
        banos = st.number_input("", min_value=0.0, step=0.5, format="%.1f", key="banos")

    # Información personal
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="etiqueta-entrada">Nombre</div>', unsafe_allow_html=True)
        nombre = st.text_input("", key="nombre", placeholder="Ingrese su nombre")

        st.markdown('<div class="etiqueta-entrada">Correo Electrónico</div>', unsafe_allow_html=True)
        correo = st.text_input("", key="correo", placeholder="Ej., usuario@ejemplo.com")

    with col2:
        st.markdown('<div class="etiqueta-entrada">Apellido</div>', unsafe_allow_html=True)
        apellido = st.text_input("", key="apellido", placeholder="Ingrese su apellido")

        st.markdown('<div class="etiqueta-entrada">Teléfono</div>', unsafe_allow_html=True)
        telefono = st.text_input("", key="telefono", placeholder="Ej., 1234567890")

    # Geocodificación y mapa
    latitud, longitud = None, None
    if 'direccion_seleccionada' in locals():
        latitud, longitud, ubicacion = geocodificar_direccion(direccion_seleccionada)
        if latitud and longitud:
            st.success(f"Ubicación encontrada: {direccion_seleccionada}")
            logger.debug(f"Ubicación encontrada: Lat {latitud}, Lon {longitud}")
            
            # Crear y mostrar el mapa responsivo
            m = folium.Map(location=[latitud, longitud], zoom_start=15, tiles="CartoDB dark_matter")
            folium.Marker([latitud, longitud], popup=direccion_seleccionada).add_to(m)
            folium_static(m)
        else:
            logger.warning("No se pudo geocodificar la dirección seleccionada")
            st.error("No se pudo geocodificar la dirección seleccionada.")
        
    # Botón de cálculo
    texto_boton = "Estimar Valor" if tipo_propiedad == "Casa" else "Estimar Renta"
    if st.button(texto_boton, key="boton_calcular"):
        logger.debug(f"Botón presionado: {texto_boton}")
        if not nombre or not apellido:
            st.error("Por favor, ingrese su nombre y apellido.")
        elif not validar_correo(correo):
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
                        st.markdown(f"<h3 style='color: {SECONDARY_COLOR};'>Valor Estimado: ${precio:,}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: {TEXT_COLOR};'>Rango de Precio Estimado: ${precio_min:,} - ${precio_max:,}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color: {SECONDARY_COLOR};'>Renta Mensual Estimada: ${precio:,}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: {TEXT_COLOR};'>Rango de Renta Estimado: ${precio_min:,} - ${precio_max:,}</p>", unsafe_allow_html=True)

                    # Gráfico de barras para visualizar el rango de precios
                    fig = go.Figure(go.Bar(
                        x=['Precio Mínimo', 'Precio Estimado', 'Precio Máximo'],
                        y=[precio_min, precio, precio_max],
                        text=[f'${precio_min:,}', f'${precio:,}', f'${precio_max:,}'],
                        textposition='auto',
                        marker_color=[SECONDARY_COLOR, PRIMARY_COLOR, SECONDARY_COLOR]
                    ))
                    fig.update_layout(
                        title_text='Rango de Precio Estimado',
                        font=dict(family="Arial", color=TEXT_COLOR),
                        paper_bgcolor=BACKGROUND_COLOR,
                        plot_bgcolor=BACKGROUND_COLOR,
                        xaxis=dict(tickfont=dict(color=TEXT_COLOR)),
                        yaxis=dict(tickfont=dict(color=TEXT_COLOR))
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
    6. Ingrese su nombre, apellido, correo electrónico y número de teléfono en los campos correspondientes.
    7. Haga clic en "Estimar Valor" o "Estimar Alquiler" para obtener la estimación.

    Nota: Asegúrese de que todos los campos estén completos para obtener una estimación precisa.
    """)

# Pie de página
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: {TEXT_COLOR};'>© 2024 Estimador de Valor de Propiedad. Todos los derechos reservados.</p>", unsafe_allow_html=True)
