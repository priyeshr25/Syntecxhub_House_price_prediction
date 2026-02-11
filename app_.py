import streamlit as st
import pandas as pd
import pickle
import datetime

# -------------------------------
# Load Model and Dataset
# -------------------------------
pipe = pickle.load(open('model@.pkl', 'rb'))
df = pickle.load(open('dataset_@.pkl', 'rb'))

# -------------------------------
# App Title
# -------------------------------
st.title("🏠 House Price Prediction App")
st.write("Enter property details to predict the estimated price.")

st.sidebar.header("Property Details")

# -------------------------------
# Categorical Inputs
# -------------------------------
country = st.sidebar.selectbox("Country", sorted(df["country"].unique()))

city = st.sidebar.selectbox("City", sorted(df["city"].unique()))

property_type = st.sidebar.selectbox(
    "Property Type",
    sorted(df["property_type"].unique())
)

furnishing_status = st.sidebar.selectbox(
    "Furnishing Status",
    sorted(df["furnishing_status"].unique())
)

# -------------------------------
# Numerical Inputs
# -------------------------------
property_size_sqft = st.sidebar.number_input(
    "Property Size (sqft)", min_value=100, step=50
)

constructed_year = st.sidebar.number_input(
    "Constructed Year",
    min_value=1900,
    max_value=datetime.datetime.now().year,
    step=1
)

previous_owners = st.sidebar.number_input(
    "Previous Owners",
    min_value=0,
    step=1
)

rooms = st.sidebar.number_input(
    "Number of Rooms",
    min_value=1,
    step=1
)

bathrooms = st.sidebar.number_input(
    "Number of Bathrooms",
    min_value=1,
    step=1
)

garage = st.sidebar.selectbox("Garage Available?", [0, 1])

garden = st.sidebar.selectbox("Garden Available?", [0, 1])

crime_cases_reported = st.sidebar.number_input(
    "Crime Cases Reported",
    min_value=0,
    step=1
)

legal_cases_on_property = st.sidebar.number_input(
    "Legal Cases on Property",
    min_value=0,
    step=1
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("Predict Price"):

    input_data = pd.DataFrame([[
        country,
        city,
        property_type,
        furnishing_status,
        property_size_sqft,
        constructed_year,
        previous_owners,
        rooms,
        bathrooms,
        garage,
        garden,
        crime_cases_reported,
        legal_cases_on_property
    ]],
    columns=[
        "country",
        "city",
        "property_type",
        "furnishing_status",
        "property_size_sqft",
        "constructed_year",
        "previous_owners",
        "rooms",
        "bathrooms",
        "garage",
        "garden",
        "crime_cases_reported",
        "legal_cases_on_property"
    ])

    try:
        prediction = pipe.predict(input_data)
        st.success(f"💰 Estimated House Price: ₹ {prediction[0]:,.2f}")
    except Exception:
        st.error("Prediction failed. Please check input values or model compatibility.")
