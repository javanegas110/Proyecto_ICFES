
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
# Ensure KNeighborsRegressor is imported as it's part of the loaded model's dependency
from sklearn.neighbors import KNeighborsRegressor


# Load the trained model and the OneHotEncoder
try:
    best_knn_model = joblib.load('best_knn_model.joblib')
    encoder = joblib.load('one_hot_encoder.joblib')
    # st.success("Modelo y codificador cargados exitosamente.") # Commented out for cleaner app UI
except Exception as e:
    st.error(f"Error al cargar el modelo o el codificador: {e}")
    st.stop()

# Define the original categorical column names (these were used to fit the encoder)
original_categorical_feature_names = [
    'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_JORNADA', 'COLE_NATURALEZA',
    'ESTU_GENERO', 'FAMI_EDUCACIONMADRE', 'FAMI_EDUCACIONPADRE',
    'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR',
    'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA'
]

# Streamlit App Title
st.title("Predicción de Puntaje Global Saber 11")
st.write("Selecciona las características para predecir el puntaje global:")

# Dictionary to store user inputs
user_inputs = {}

# Create select boxes for each categorical variable
for i, col_name in enumerate(original_categorical_feature_names):
    if i < len(encoder.categories_):
        categories = encoder.categories_[i].tolist()
        user_inputs[col_name] = st.selectbox(f"**{col_name.replace('_', ' ')}**", categories)
    else:
        st.warning(f"No se encontraron categorías para la columna {col_name}. Verifique la configuración del codificador.")

if st.button("Predecir Puntaje Global"):
    # Create a DataFrame from user inputs. Ensure column order matches the encoder's expectation.
    input_data = {col: [user_inputs[col]] for col in original_categorical_feature_names}
    input_df = pd.DataFrame(input_data)

    try:
        # Transform the input using the loaded encoder
        encoded_input_array = encoder.transform(input_df)

        # Create a DataFrame with the encoded features, ensuring column names match the training data (X_final)
        encoded_feature_names = encoder.get_feature_names_out(original_categorical_feature_names)
        input_encoded = pd.DataFrame(encoded_input_array, columns=encoded_feature_names)

        # Make prediction
        prediction = best_knn_model.predict(input_encoded)[0]

        st.success(f"El puntaje global predicho es: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.write("Asegúrate de que todas las selecciones son válidas y que el codificador se aplicó correctamente.")
