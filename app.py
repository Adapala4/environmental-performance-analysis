import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

merged_data = pd.read_csv("data/merged_data.csv")
model = joblib.load("models/rf_pipeline.joblib")
st.title('Environmental Performance & Socioeconomic Development')
page = st.sidebar.radio("Navigation", ["Overview", "Country Explorer", "World Map", "Model Performance", "What-if Simulator"])
if page == "Overview":
    st.header("Overview")
    st.write('This dashboard explores the relationship between socioeconomic indicators and environmental performance across countries.')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Countries", merged_data.shape[0])
    with col2:
        st.metric("Average EPI", round(merged_data['EPI.new'].mean(), 2))
    with col3:
        st.metric("Best Model R²", 0.695)

elif page == "Country Explorer":
    st.header("Country Explorer")
    selected_country = st.selectbox("Select a country", merged_data["country"])
    country_data = merged_data[merged_data["country"] == selected_country]
    epi_score = country_data["EPI.new"].iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("EPI Score", round(epi_score, 2))
    with col2:
        st.metric("GDP", round(country_data["GDP"].iloc[0], 2))
    with col3:
        secondary = country_data["Secondary_enrollment"].iloc[0]
        if pd.isna(secondary):
            secondary = "N/A"
        else:
            secondary = round(secondary, 2)
        st.metric("Secondary Enrollment", secondary)
    with col4:
        st.metric("Urban Population", round(country_data["Urban Population"].iloc[0], 2))

    log_gdp = np.log(country_data["GDP"].iloc[0])
    country_features = pd.DataFrame({
        "log_GDP": [log_gdp],
        "Secondary_enrollment": [country_data["Secondary_enrollment"].iloc[0]],
        "Urban Population": [country_data["Urban Population"].iloc[0]],
        "Employment_rate": [country_data["Employment_rate"].iloc[0]],
        "Unemployment": [country_data["Unemployment"].iloc[0]]})

    predicted_epi = model.predict(country_features)[0]
    prediction_error = epi_score - predicted_epi
    actual_col, predicted_col, error_col = st.columns(3)

    with actual_col:
        st.metric("Actual EPI", round(epi_score, 2))

    with predicted_col:
        st.metric("Predicted EPI", round(predicted_epi, 2))

    with error_col:
        st.metric("Prediction Error", round(prediction_error, 2))

elif page == "World Map":
    st.header("World Map")
    selected_indicator = st.selectbox("Select an environmental indicator",
    ["Environmental Performance", "Air Quality", "Waste Management", "Environmental Health"])
    indicator_columns = {"Environmental Performance" : "EPI.new", "Air Quality" : "AIR.new",  "Waste Management" : "WMG.new",
      "Environmental Health"  : "HLT.new"}
    selected_column = indicator_columns[selected_indicator]

    fig = px.choropleth(data_frame = merged_data,  locations = "iso", color = selected_column, hover_name = "country")
    fig.update_geos(
    projection_type="orthographic",
    bgcolor="rgba(0,0,0,0)", showocean = True, oceancolor= "#1C7099")
    fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", height = 600)
    st.plotly_chart(fig)

