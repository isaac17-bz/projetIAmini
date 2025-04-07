import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gdown
import os

# Chemin local du modèle
model_path = "car_price_model.pkl"


brands = ['BMW', 'Mercedes', 'Audi', 'Toyota', 'Ford']  # Les marques de ton dataset
brand = st.selectbox('Marque', brands)


# Téléchargement automatique si le modèle n'existe pas
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1r99eYWZsVIHAn4hhSC1Qt6UxFYeWXvnf"  # Lien direct de téléchargement
    gdown.download(url, model_path, quiet=False)

# Charger le modèle sauvegardé
model = joblib.load(model_path)

# Dictionnaire pour encoder les colonnes catégorielles
label_encoders = {}

# Fonction pour prédire le prix
def predict_price(brand, model_name, year, engine_size, fuel_type, transmission, mileage, doors, owner_count):
    # Créer un DataFrame avec les valeurs entrées
    new_car = {
        'Brand': [brand],
        'Model': [model_name],
        'Year': [year],
        'Engine_Size': [engine_size],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Mileage': [mileage],
        'Doors': [doors],
        'Owner_Count': [owner_count]
    }
    new_car_df = pd.DataFrame(new_car)
    
    # Encoder les colonnes catégorielles
    for col in new_car_df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            new_car_df[col] = label_encoders[col].transform(new_car_df[col])

    # Prédire le prix
    predicted_price = model.predict(new_car_df)
    return predicted_price[0]

# Interface Streamlit
st.title("Prédiction du Prix d'une Voiture")

# Saisie des caractéristiques de la voiture
brand = st.text_input('Marque')
model_name = st.text_input('Modèle')
year = st.number_input('Année', min_value=1900, max_value=2025, step=1)
engine_size = st.number_input('Taille du moteur (L)', min_value=0.5, max_value=10.0, step=0.1)
fuel_type = st.selectbox('Type de carburant', ['Petrol', 'Diesel', 'Electric', 'Hybrid'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
mileage = st.number_input('Kilométrage (en km)', min_value=0, step=1000)
doors = st.number_input('Nombre de portes', min_value=2, max_value=5)
owner_count = st.number_input('Nombre de propriétaires précédents', min_value=1, max_value=5)

# Bouton pour prédire le prix
if st.button('Prédire le Prix'):
    predicted_price = predict_price(brand, model_name, year, engine_size, fuel_type, transmission, mileage, doors, owner_count)
    st.write(f"Le prix prédit pour cette voiture est : {predicted_price:.2f}€")
