import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("car_price_model.pkl")

st.title("🚗 Prédiction du Prix d'une Voiture")

# Collecter les entrées de l'utilisateur
brand = st.selectbox("Marque", ["Toyota", "Renault", "Peugeot", "BMW", "Audi", "Hyundai"])  # personnalise
model_name = st.text_input("Modèle")
engine_size = st.number_input("Cylindrée du moteur (en L)", min_value=0.0, step=0.1)
fuel_type = st.selectbox("Type de carburant", ["Petrol", "Diesel", "Electric", "Hybrid"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic", "CVT"])
mileage = st.number_input("Kilométrage", min_value=0)
doors = st.selectbox("Nombre de portes", [2, 3, 4, 5])
owner_count = st.selectbox("Nombre de propriétaires précédents", [0, 1, 2, 3])

# Préparation de l'entrée utilisateur
input_data = pd.DataFrame({
    "Brand": [brand],
    "Model": [model_name],
    "Engine_Size": [engine_size],
    "Fuel_Type": [fuel_type],
    "Transmission": [transmission],
    "Milleage": [mileage],
    "Doors": [doors],
    "Owner_count": [owner_count]
})

# Prédiction
if st.button("Prédire le prix"):
    prediction = model.predict(input_data)
    st.success(f"💰 Prix estimé : {int(prediction[0]):,} DZD")
