import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Big Five Cluster App", layout="wide")

st.title("🎯 Clasificador de Personalidad Big Five con K-Means")
st.write("Introduce tus puntajes del test de personalidad Big Five para descubrir tu clúster y compararte con famosos de tu grupo.")

# 1. Cargar datos y artefactos del modelo
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df = pd.read_csv(url)

    # Mapeo de categorías profesionales
    map_categorias = {
        1: 'Actor/actriz',
        2: 'Cantante',
        3: 'Modelo',
        4: 'Tv, series',
        5: 'Radio',
        6: 'Tecnología',
        7: 'Deportes',
        8: 'Politica',
        9: 'Escritor'
    }
    df['categoria_lbl'] = df['categoria'].map(map_categorias)
    return df

@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return kmeans, scaler

try:
    df = load_data()
    kmeans_cargado, scaler_cargado = load_models()

    # Asegurar etiquetas de clusters en el dataframe
    df['cluster'] = kmeans_cargado.labels_
except Exception as e:
    st.error(f"Error cargando los archivos necesarios. Asegúrate de tener 'kmeans_model.joblib' y 'scaler.joblib' en el mismo directorio. Detalle: {e}")
    st.stop()

# 2. Sidebar para entradas del usuario
st.sidebar.header("📝 Ingresa tus Resultados")
op = st.sidebar.slider("Apertura (op)", min_value=0.0, max_value=120.0, value=69.6, step=0.1)
co = st.sidebar.slider("Conciencia (co)", min_value=0.0, max_value=120.0, value=76.9, step=0.1)
ex = st.sidebar.slider("Extroversión (ex)", min_value=0.0, max_value=120.0, value=50.9, step=0.1)
ag = st.sidebar.slider("Amabilidad (ag)", min_value=0.0, max_value=120.0, value=67.7, step=0.1)
ne = st.sidebar.slider("Neuroticismo (ne)", min_value=0.0, max_value=120.0, value=102.7, step=0.1)

# Crear dataframe de entrada
mis_datos = pd.DataFrame([{'op': op, 'co': co, 'ex': ex, 'ag': ag, 'ne': ne}])

# 3. Predicción del clúster
mis_datos_escalados = scaler_cargado.transform(mis_datos)
mi_cluster = kmeans_cargado.predict(mis_datos_escalados)[0]

# Filtrar subconjunto del clúster asignado
cluster_df = df[df['cluster'] == mi_cluster]

# Layout de Streamlit para mostrar resultados
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### 🏆 ¡Tu Clúster Asignado es el **Clúster {mi_cluster}**!")
    st.write(f"Personas en este clúster comparten rasgos de personalidad similares a los tuyos.")

    st.markdown("#### 🌟 Famosos en este mismo Clúster:")
    st.dataframe(cluster_df[['usuario', 'categoria_lbl']].head(10), use_container_width=True)

with col2:
    st.markdown("### 📊 Visualización de Rasgos de Personalidad")
    tab1, tab2 = st.tabs(["📈 Gráfico de Radar (Superpuesto)", "📊 Gráfico de Barras (Clúster)"])

    features = ['op', 'co', 'ex', 'ag', 'ne']
    promedios_c = cluster_df[features].mean().values.tolist()
    mis_valores = [op, co, ex, ag, ne]

    with tab1:
        # Preparación de datos para gráfico de radar (cerrando el ciclo)
        promedios_radar = promedios_c + [promedios_c[0]]
        mis_valores_radar = mis_valores + [mis_valores[0]]

        angulos = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angulos += angulos[:1]

        fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Dibujar promedio del clúster
        ax_radar.plot(angulos, promedios_radar, color='#1f77b4', linewidth=2, label=f'Promedio Clúster {mi_cluster}')
        ax_radar.fill(angulos, promedios_radar, color='#1f77b4', alpha=0.25)

        # Dibujar valores del usuario sobrepuestos
        ax_radar.plot(angulos, mis_valores_radar, color='#d62728', linewidth=2, linestyle='dashed', label='Mis Resultados')
        ax_radar.fill(angulos, mis_valores_radar, color='#d62728', alpha=0.15)

        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_thetagrids(np.degrees(angulos[:-1]), features)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Comparativa: Mis Resultados vs Promedio del Clúster', y=1.1, fontsize=12, fontweight='bold')
        st.pyplot(fig_radar)

    with tab2:
        # Gráfico de barras del clúster asignado
        df_barras = pd.DataFrame({
            'Rasgo': features,
            'Promedio': promedios_c
        })

        fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=df_barras, x='Rasgo', y='Promedio', palette='coolwarm', ax=ax_bar)
        ax_bar.set_title(f'Promedio de Rasgos en Escala Original - Clúster {mi_cluster}', fontsize=12, fontweight='bold')
        ax_bar.set_ylabel('Valor Promedio')
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_bar)
