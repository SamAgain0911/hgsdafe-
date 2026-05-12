
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Load the trained model
model = joblib.load('modelo_desercion.pkl')

# --- Streamlit App --- (main function for organization)
def main():
    st.set_page_config(layout="wide")
    st.title('Sistema de Predicción de Deserción Estudiantil 🎓')

    st.markdown("""
    Esta aplicación utiliza un modelo de Machine Learning para predecir la probabilidad de deserción de un estudiante
    basado en diversas características. Ingresa los datos del estudiante para obtener una predicción.
    """)

    # Sidebar for input features
    st.sidebar.header('Datos del Estudiante')

    # Input fields for student data
    edad = st.sidebar.slider('Edad', 18, 30, 22)
    promedio = st.sidebar.slider('Promedio (GPA)', 2.0, 4.0, 3.0, 0.01)
    asistencia = st.sidebar.slider('Porcentaje de Asistencia', 0.6, 1.0, 0.85, 0.01)
    horas_estudio = st.sidebar.slider('Horas de Estudio Semanales', 5, 40, 20)
    uso_plataforma = st.sidebar.slider('Porcentaje de Uso de Plataforma', 30, 100, 70)
    materias_perdidas = st.sidebar.slider('Materias Perdidas', 0, 5, 1)
    nivel_socioeconomico = st.sidebar.selectbox('Nivel Socioeconómico', [1, 2, 3], format_func=lambda x: {1: 'Bajo', 2: 'Medio', 3: 'Alto'}[x])
    trabaja = st.sidebar.selectbox('¿Trabaja?', [0, 1], format_func=lambda x: {0: 'No', 1: 'Sí'}[x])
    acceso_internet = st.sidebar.selectbox('Acceso a Internet', [0, 1], format_func=lambda x: {0: 'No', 1: 'Sí'}[x])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[edad, promedio, asistencia, horas_estudio, uso_plataforma,
                                  materias_perdidas, nivel_socioeconomico, trabaja, acceso_internet]],
                                columns=['edad', 'promedio', 'asistencia', 'horas_estudio', 'uso_plataforma',
                                         'materias_perdidas', 'nivel_socioeconomico', 'trabaja', 'acceso_internet'])

    st.sidebar.subheader('Datos Ingresados:')
    st.sidebar.write(input_data)

    # Make prediction
    if st.sidebar.button('Predecir Deserción'):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[:, 1][0] # Probability of dropout

        st.subheader('Resultado de la Predicción:')
        if prediction == 1:
            st.error(f'**¡ALERTA!** El estudiante tiene una alta probabilidad de desertar ({prediction_proba:.2f}).')
            st.markdown("""
            **Recomendaciones:**
            *   Ofrecer tutorías personalizadas.
            *   Explorar opciones de ayuda financiera.
            *   Proveer acceso a recursos de apoyo psicológico y académico.
            """)
        else:
            st.success(f'El estudiante tiene una baja probabilidad de desertar ({prediction_proba:.2f}).')
            st.markdown("""
            **Recomendaciones:**
            *   Mantener el seguimiento académico y bienestar general.
            *   Fomentar la participación en actividades extracurriculares.
            """)

        st.markdown(f"Probabilidad de Deserción: **{prediction_proba:.2f}**")

        # Visualize probability
        prob_df = pd.DataFrame({
            'Categoría': ['Probabilidad de Deserción', 'Probabilidad de No Deserción'],
            'Valor': [prediction_proba, 1 - prediction_proba]
        })
        fig = px.bar(prob_df, x='Categoría', y='Valor', 
                     color='Categoría', 
                     title='Probabilidad de Deserción vs No Deserción',
                     color_discrete_map={'Probabilidad de Deserción': 'red', 'Probabilidad de No Deserción': 'green'})
        st.plotly_chart(fig)
        
    st.subheader('Importancia de las Variables (Modelo)')
    feature_importances = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    fig_importance = px.bar(feature_importances, x='feature', y='importance',
                            title='Importancia de las Variables en el Modelo',
                            labels={'feature': 'Variable', 'importance': 'Importancia'})
    st.plotly_chart(fig_importance)

if __name__ == '__main__':
    main()
